# -*- coding: utf-8 -*-
"""
===================================
股票筛选器 - Stock Screener
===================================

职责：
1. 获取指数成分股（沪深300/中证500等）
2. 使用 StockTrendAnalyzer 完整打分逻辑进行评分
3. 输出最值得买入的股票候选列表

筛选逻辑：
- 使用完整的 StockTrendAnalyzer 打分系统（趋势+乖离率+量能+MACD+RSI+支撑）
- 只返回买入信号为 STRONG_BUY 或 BUY 的股票
- 按得分从高到低排序

数据源策略：
- 使用 DataFetcherManager 统一管理多数据源
- 支持自动故障切换（akshare_em → efinance → tencent → sina）
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import akshare as ak
import pandas as pd

from data_provider import DataFetcherManager

# 导入完整的趋势分析器
from src.stock_analyzer import (
    StockTrendAnalyzer,
    TrendAnalysisResult,
    TrendStatus,
    BuySignal,
)

logger = logging.getLogger(__name__)


@dataclass
class StockScore:
    """股票评分数据类（兼容旧代码）"""
    code: str
    name: str
    price: float
    change_pct: float
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    bias_ma5: Optional[float] = None
    volume_ratio: Optional[float] = None
    turnover_rate: Optional[float] = None
    chip_concentration: Optional[float] = None
    pe_ratio: Optional[float] = None

    # 筛选条件得分
    bullish_alignment: int = 0
    bias_safe: int = 0
    volume_valid: int = 0
    chip_healthy: int = 0
    trend_strength: float = 0.0  # 改为 float，与 TrendAnalysisResult 一致
    total_score: int = 0

    # 标记
    is_bullish: bool = False
    is_strong_trend: bool = False

    # 新增：完整的分析结果
    analysis_result: Optional[TrendAnalysisResult] = None
    buy_signal: BuySignal = BuySignal.WAIT


class StockScreener:
    """股票筛选器

    使用 StockTrendAnalyzer 完整打分逻辑，筛选最值得买入的股票。

    筛选条件：
    1. 买入信号为 STRONG_BUY 或 BUY
    2. 按得分从高到低排序
    """

    # 指数代码映射
    ZZ1000_INDEX = "000852"  # 中证1000

    def __init__(self):
        """初始化筛选器"""
        self.fetcher_manager = DataFetcherManager()
        self.trend_analyzer = StockTrendAnalyzer()

    def get_index_constituents(self, index_code: str = ZZ1000_INDEX) -> List[str]:
        """
        获取指数成分股代码列表

        Args:
            index_code: 指数代码

        Returns:
            成分股代码列表
        """
        try:
            logger.info(f"正在获取指数 {index_code} 成分股...")

            df = None

            # 方法1: 使用 ak.index_stock_cons()
            try:
                df = ak.index_stock_cons(symbol=index_code)
                logger.info(f"[AkShare] index_stock_cons 成功获取 {len(df)} 只成分股")
            except Exception as e:
                logger.debug(f"[AkShare] index_stock_cons 失败: {e}")

            # 方法2: 使用 ak.index_zh_cons()（中证指数）
            if df is None:
                try:
                    df = ak.index_zh_cons(symbol=index_code)
                    logger.info(f"[AkShare] index_zh_cons 成功获取 {len(df)} 只成分股")
                except Exception as e:
                    logger.debug(f"[AkShare] index_zh_cons 失败: {e}")

            if df is None or df.empty:
                logger.error(f"无法获取指数 {index_code} 的成分股数据")
                return []

            # 提取股票代码
            if '品种代码' in df.columns:
                codes = df['品种代码'].tolist()
            elif '股票代码' in df.columns:
                codes = df['股票代码'].tolist()
            elif 'code' in df.columns:
                codes = df['code'].tolist()
            else:
                logger.error(f"成分股数据中没有股票代码字段: {df.columns.tolist()}")
                return []

            logger.info(f"成功获取 {len(codes)} 只成分股")
            return codes

        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")
            return []

    def get_realtime_data_batch(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        批量获取实时行情数据（使用多数据源协同）

        策略（按优先级）：
        1. AkShare 批量接口（stock_zh_a_spot_em）- 一次请求获取全部 A 股
        2. Efinance 批量接口 - 备选
        3. DataFetcherManager 逐个获取 - 兜底

        Args:
            stock_codes: 股票代码列表

        Returns:
            包含实时行情的 DataFrame
        """
        # 方案1: 尝试 AkShare 批量接口（一次请求获取全部，最快）
        df = self._try_akshare_batch(stock_codes)
        if not df.empty:
            return df

        # 方案2: 尝试 Efinance 批量接口
        df = self._try_efinance_batch(stock_codes)
        if not df.empty:
            return df

        # 方案3: 使用 DataFetcherManager 逐个获取（可靠但较慢）
        logger.info("批量接口全部失败，切换到逐个获取模式（可能较慢）...")
        return self._fetch_with_multi_sources(stock_codes)

    def _try_akshare_batch(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        尝试使用 AkShare 批量接口获取行情（一次请求获取全部 A 股，最快）
        """
        max_retries = 2
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                logger.info(f"[AkShare批量] 正在获取 A 股实时行情... (尝试 {attempt + 1}/{max_retries})")

                df = ak.stock_zh_a_spot_em()

                if df is None or df.empty:
                    logger.warning(f"[AkShare批量] 返回数据为空 (尝试 {attempt + 1}/{max_retries})")
                    continue

                # 筛选出目标股票
                code_col = None
                for col in ['代码', 'code', '股票代码']:
                    if col in df.columns:
                        code_col = col
                        break

                if code_col is None:
                    logger.error(f"[AkShare批量] 实时行情数据中没有代码字段: {df.columns.tolist()}")
                    return pd.DataFrame()

                df_filtered = df[df[code_col].astype(str).isin(stock_codes)]
                logger.info(f"[AkShare批量] 成功获取 {len(df_filtered)} 只股票的实时行情")
                return df_filtered

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[AkShare批量] 获取失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")

                if 'Connection' in error_msg or 'RemoteDisconnected' in error_msg or 'timeout' in error_msg.lower():
                    if attempt < max_retries - 1:
                        logger.info(f"[AkShare批量] 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue

                return pd.DataFrame()

        return pd.DataFrame()

    def _try_efinance_batch(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        尝试使用 Efinance 批量接口获取行情
        """
        try:
            import efinance as ef

            logger.info("[Efinance批量] 正在获取 A 股实时行情...")

            df = ef.stock.get_realtime_quotes()

            if df is None or df.empty:
                logger.warning("[Efinance批量] 返回数据为空")
                return pd.DataFrame()

            # 筛选出目标股票
            code_col = None
            for col in ['股票代码', '代码', 'code', '股票名']:
                if col in df.columns:
                    code_col = col
                    break

            if code_col is None:
                logger.error(f"[Efinance批量] 无法找到代码列: {df.columns.tolist()}")
                return pd.DataFrame()

            df_filtered = df[df[code_col].astype(str).isin(stock_codes)]
            logger.info(f"[Efinance批量] 成功获取 {len(df_filtered)} 只股票的实时行情")
            return df_filtered

        except ImportError:
            logger.debug("[Efinance批量] efinance 未安装")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"[Efinance批量] 获取失败: {e}")
            return pd.DataFrame()

    def _fetch_with_multi_sources(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        使用 DataFetcherManager 多数据源逐个获取实时行情（兜底方案）
        """
        logger.info(f"[多数据源] 开始逐个获取 {len(stock_codes)} 只股票的实时行情...")

        quotes_data = []
        success_count = 0
        fail_count = 0

        for i, code in enumerate(stock_codes):
            try:
                quote = self.fetcher_manager.get_realtime_quote(code)

                if quote and quote.has_basic_data():
                    quotes_data.append({
                        '代码': code,
                        '名称': quote.name,
                        '最新价': quote.price,
                        '涨跌幅': quote.change_pct,
                        '涨跌额': quote.change_amount,
                        '成交量': quote.volume,
                        '成交额': quote.amount,
                        '量比': quote.volume_ratio,
                        '换手率': quote.turnover_rate,
                        '振幅': quote.amplitude,
                        '最高': quote.high,
                        '最低': quote.low,
                        '今开': quote.open_price,
                        '市盈率-动态': quote.pe_ratio,
                    })
                    success_count += 1
                else:
                    fail_count += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"[多数据源] 进度: {i + 1}/{len(stock_codes)}, 成功: {success_count}, 失败: {fail_count}")

            except Exception as e:
                fail_count += 1
                logger.warning(f"[多数据源] {code} 获取异常: {e}")
                continue

        if not quotes_data:
            logger.error("[多数据源] 未能获取任何股票数据")
            return pd.DataFrame()

        df = pd.DataFrame(quotes_data)
        logger.info(f"[多数据源] 完成! 成功: {success_count}, 失败: {fail_count}")
        return df

    def _get_stock_history(self, code: str, days: int = 120) -> Optional[pd.DataFrame]:
        """
        获取股票历史 K 线数据

        Args:
            code: 股票代码
            days: 获取天数

        Returns:
            包含 OHLCV 的 DataFrame，失败返回 None
        """
        try:
            # 使用 DataFetcherManager.get_daily_data() 获取历史数据
            df, _ = self.fetcher_manager.get_daily_data(code, days=days)
            if df is None or df.empty:
                logger.debug(f"[历史数据] {code} 获取失败")
                return None
            return df
        except Exception as e:
            logger.debug(f"[历史数据] {code} 获取异常: {e}")
            return None

    def analyze_stocks(self, stock_codes: List[str], top_n: int = 2) -> List[StockScore]:
        """
        使用完整的 StockTrendAnalyzer 分析股票并打分

        Args:
            stock_codes: 股票代码列表
            top_n: 返回前 N 名

        Returns:
            按得分排序的 StockScore 列表（仅包含买入信号为 STRONG_BUY 或 BUY 的股票）
        """
        logger.info(f"[完整分析] 开始分析 {len(stock_codes)} 只股票...")

        scores = []
        success_count = 0
        fail_count = 0

        # 先批量获取实时行情（用于过滤 ST 股和获取名称）
        realtime_df = self.get_realtime_data_batch(stock_codes)
        if realtime_df.empty:
            logger.error("[完整分析] 无法获取实时行情")
            return []

        # 过滤 ST 股
        name_col = '名称' if '名称' in realtime_df.columns else 'name'
        code_col = '代码' if '代码' in realtime_df.columns else 'code'
        amount_col = '成交额' if '成交额' in realtime_df.columns else 'amount'

        filtered_df = realtime_df[
            (~realtime_df[name_col].str.contains('ST', na=False)) &
            (realtime_df[amount_col] > 0)
        ]

        valid_codes = filtered_df[code_col].astype(str).tolist()
        logger.info(f"[完整分析] 过滤后剩余 {len(valid_codes)} 只股票（已排除 ST 股）")

        # 逐只股票进行完整分析
        for i, code in enumerate(valid_codes):
            try:
                # 获取历史 K 线数据
                df = self._get_stock_history(code, days=120)
                if df is None or len(df) < 30:
                    fail_count += 1
                    continue

                # 使用 StockTrendAnalyzer 进行完整分析
                result = self.trend_analyzer.analyze(df, code)

                # 从实时行情获取额外信息
                realtime_row = filtered_df[filtered_df[code_col].astype(str) == code]
                name = ""
                change_pct = 0.0
                if not realtime_row.empty:
                    row = realtime_row.iloc[0]
                    name = row.get('名称', row.get('name', ''))
                    change_pct = float(row.get('涨跌幅', row.get('change_percent', 0)))

                # 创建 StockScore 对象
                score = StockScore(
                    code=code,
                    name=name or result.code,
                    price=result.current_price,
                    change_pct=change_pct,
                    ma5=result.ma5,
                    ma10=result.ma10,
                    ma20=result.ma20,
                    bias_ma5=result.bias_ma5,
                    total_score=result.signal_score,
                    buy_signal=result.buy_signal,
                    analysis_result=result,
                    # 标记
                    is_bullish=result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL],
                    trend_strength=result.trend_strength,
                )

                scores.append(score)
                success_count += 1

                # 每 20 只股票打印一次进度
                if (i + 1) % 20 == 0:
                    logger.info(f"[完整分析] 进度: {i + 1}/{len(valid_codes)}, 成功: {success_count}, 失败: {fail_count}")

            except Exception as e:
                fail_count += 1
                logger.debug(f"[完整分析] {code} 分析异常: {e}")
                continue

        logger.info(f"[完整分析] 完成! 成功: {success_count}, 失败: {fail_count}")

        if not scores:
            logger.warning("[完整分析] 未能分析任何股票")
            return []

        # 筛选买入信号为 STRONG_BUY 或 BUY 的股票
        buy_signals = [
            BuySignal.STRONG_BUY,
            BuySignal.BUY,
        ]

        buyable_scores = [s for s in scores if s.buy_signal in buy_signals]
        logger.info(f"[完整分析] 筛选出 {len(buyable_scores)} 只值得买入的股票")

        if not buyable_scores:
            logger.warning("[完整分析] 没有股票满足买入条件")
            # 如果没有买入信号，返回得分最高的几只（但标记为不推荐）
            scores_sorted = sorted(scores, key=lambda x: x.total_score, reverse=True)
            return scores_sorted[:top_n]

        # 按买入信号优先级 + 得分排序
        # STRONG_BUY 优先于 BUY，同级别按得分排序
        def sort_key(s: StockScore):
            signal_priority = 0 if s.buy_signal == BuySignal.STRONG_BUY else 1
            return (signal_priority, -s.total_score)

        buyable_scores.sort(key=sort_key)
        return buyable_scores[:top_n]

    def screen(self, index_code: Optional[str] = None, top_n: int = 2) -> Tuple[List[StockScore], str]:
        """
        执行股票筛选流程

        Args:
            index_code: 指数代码，默认中证1000
            top_n: 返回前 N 名

        Returns:
            (筛选结果, 指数名称)
        """
        if index_code is None:
            index_code = self.ZZ1000_INDEX

        # 确定指数名称
        index_names = {
            self.ZZ1000_INDEX: "中证1000",
            "000857": "中证800",
            "000903": "中证200",
            "000300": "沪深300",
            "000905": "中证500",
        }
        index_name = index_names.get(index_code, f"指数{index_code}")

        logger.info(f"==========================================")
        logger.info(f"🎯 股票筛选器启动（完整分析模式）")
        logger.info(f"📊 筛选范围: {index_name} ({index_code})")
        logger.info(f"🏆 返回数量: {top_n}")
        logger.info(f"==========================================")

        # 1. 获取成分股
        stock_codes = self.get_index_constituents(index_code)
        if not stock_codes:
            logger.error(f"无法获取 {index_name} 成分股，筛选终止")
            return [], index_name

        logger.info(f"开始从 {len(stock_codes)} 只成分股中筛选...")

        # 2. 使用完整分析模式进行筛选
        top_stocks = self.analyze_stocks(stock_codes, top_n=top_n)

        # 3. 输出结果
        logger.info(f"\n==========================================")
        logger.info(f"📋 筛选结果 TOP {len(top_stocks)}")
        logger.info(f"==========================================")

        for i, stock in enumerate(top_stocks, 1):
            signal_emoji = "🔥" if stock.buy_signal == BuySignal.STRONG_BUY else "✅"
            logger.info(
                f"\n#{i} {signal_emoji} {stock.name}({stock.code})\n"
                f"   买入信号: {stock.buy_signal.value}\n"
                f"   综合得分: {stock.total_score}/100\n"
                f"   现价: {stock.price:.2f} | 涨跌: {stock.change_pct:+.2f}%\n"
                f"   MA5: {stock.ma5 or 'N/A':.2f} | MA10: {stock.ma10 or 'N/A':.2f} | MA20: {stock.ma20 or 'N/A':.2f}\n"
                f"   乖离率: {stock.bias_ma5 or 0:+.2f}% | 趋势强度: {stock.trend_strength}/100"
            )

            # 如果有完整分析结果，打印更多信息
            if stock.analysis_result:
                result = stock.analysis_result
                logger.info(f"   MACD: {result.macd_status.value} | RSI: {result.rsi_status.value}")
                if result.signal_reasons:
                    logger.info(f"   买入理由: {'; '.join(result.signal_reasons[:3])}")
                if result.risk_factors:
                    logger.info(f"   风险提示: {'; '.join(result.risk_factors[:2])}")

        return top_stocks, index_name


def format_screener_report(stocks: List[StockScore], index_name: str) -> str:
    """
    格式化筛选结果报告

    Args:
        stocks: 筛选出的股票列表
        index_name: 指数名称

    Returns:
        Markdown 格式的报告
    """
    if not stocks:
        return f"# 📊 {index_name} 筛选结果\n\n未筛选出符合条件的股票。"

    # 检查是否有买入信号
    has_buy_signal = any(s.buy_signal in [BuySignal.STRONG_BUY, BuySignal.BUY] for s in stocks)

    lines = [
        f"# 🎯 {index_name} 每日优选股票",
        f"",
        f"## 📋 筛选结果",
        f"",
    ]

    if has_buy_signal:
        lines.extend([
            f"**使用完整的 StockTrendAnalyzer 打分系统（100分制）**",
            f"",
            f"评分维度：",
            f"- 趋势（30分）：均线排列状态",
            f"- 乖离率（20分）：价格与MA5偏离度",
            f"- 量能（15分）：量价配合程度",
            f"- MACD（15分）：金叉/死叉/多空状态",
            f"- RSI（10分）：超买/超卖/强势",
            f"- 支撑（10分）：均线支撑有效性",
            f"",
            f"**买入信号说明：**",
            f"- 🔥 强烈买入：得分≥75 + 多头排列",
            f"- ✅ 买入：得分≥60 + 多头/弱多头",
            f"",
            f"---",
            f"",
        ])
    else:
        lines.extend([
            f"⚠️ **今日没有股票满足买入条件**",
            f"",
            f"以下为得分最高的股票（仅供参考）：",
            f"",
            f"---",
            f"",
        ])

    for i, stock in enumerate(stocks, 1):
        signal_emoji = "🔥" if stock.buy_signal == BuySignal.STRONG_BUY else "✅" if stock.buy_signal == BuySignal.BUY else "⏸️"

        lines.extend([
            f"### #{i} {signal_emoji} {stock.name}({stock.code})",
            f"",
            f"| 指标 | 数值 | 状态 |",
            f"|------|------|------|",
            f"| **买入信号** | | **{stock.buy_signal.value}** |",
            f"| **综合得分** | | **{stock.total_score}/100** |",
            f"| 现价 | {stock.price:.2f} 元 | {'🟢' if stock.change_pct > 0 else '🔴'} {stock.change_pct:+.2f}% |",
            f"| MA5 | {stock.ma5 or 'N/A':.2f} | |",
            f"| MA10 | {stock.ma10 or 'N/A':.2f} | |",
            f"| MA20 | {stock.ma20 or 'N/A':.2f} | |",
            f"| 乖离率(MA5) | {stock.bias_ma5 or 0:+.2f}% | {'✅安全' if abs(stock.bias_ma5 or 0) <= 5 else '⚠️偏高'} |",
            f"| 趋势强度 | {stock.trend_strength}/100 | |",
            f"",
        ])

        # 如果有完整分析结果，添加更多信息
        if stock.analysis_result:
            result = stock.analysis_result
            lines.extend([
                f"**技术指标：**",
                f"- 趋势状态: {result.trend_status.value}",
                f"- MACD: {result.macd_status.value}",
                f"- RSI: {result.rsi_status.value}",
                f"- 量能: {result.volume_status.value}",
                f"",
            ])

            if result.signal_reasons:
                lines.append(f"**买入理由：**")
                for reason in result.signal_reasons[:5]:
                    lines.append(f"- {reason}")
                lines.append(f"")

            if result.risk_factors:
                lines.append(f"**风险提示：**")
                for risk in result.risk_factors[:3]:
                    lines.append(f"- {risk}")
                lines.append(f"")

        lines.extend([
            f"---",
            f"",
        ])

    # 添加免责声明
    lines.extend([
        f"## ⚠️ 免责声明",
        f"",
        f"本报告仅供参考，不构成投资建议。股市有风险，投资需谨慎。",
        f"",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    screener = StockScreener()

    # 解析命令行参数
    index_code = sys.argv[1] if len(sys.argv) > 1 else None
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # 执行筛选
    stocks, index_name = screener.screen(index_code=index_code, top_n=top_n)

    # 输出报告
    report = format_screener_report(stocks, index_name)
    print(report)
