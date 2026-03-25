# -*- coding: utf-8 -*-
"""
===================================
股票筛选器 - Stock Screener
===================================

职责：
1. 获取指数成分股（沪深300/中证500等）
2. 使用 StockTrendAnalyzer 完整打分逻辑进行评分
3. 输出最值得买入的股票候选列表

设计理念（盘后分析）：
- 使用历史日线数据（get_daily_data），而非实时行情
- 适合盘后分析场景，不依赖实时行情接口
- 完整利用 StockTrendAnalyzer 的技术分析能力

筛选逻辑：
- 使用完整的 StockTrendAnalyzer 打分系统（趋势+乖离率+量能+MACD+RSI+支撑）
- 只返回买入信号为 STRONG_BUY 或 BUY 的股票
- 按买入信号优先级 + 得分排序
"""

import logging
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
    """股票评分数据类"""
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
    trend_strength: float = 0.0
    total_score: int = 0
    is_bullish: bool = False
    analysis_result: Optional[TrendAnalysisResult] = None
    buy_signal: BuySignal = BuySignal.WAIT


class StockScreener:
    """股票筛选器（盘后分析模式）

    使用历史日线数据 + StockTrendAnalyzer 完整打分逻辑。

    优势：
    - 不依赖实时行情接口，适合盘后分析
    - 完整的技术分析能力（MA/MACD/RSI等）
    - 按买入信号筛选，确保返回的是"值得买入"的股票
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

    def _get_stock_daily_data(self, code: str, days: int = 120) -> Optional[pd.DataFrame]:
        """
        获取股票历史日线数据

        Args:
            code: 股票代码
            days: 获取天数

        Returns:
            包含 OHLCV 的 DataFrame，失败返回 None
        """
        try:
            df, _ = self.fetcher_manager.get_daily_data(code, days=days)
            if df is None or df.empty:
                logger.debug(f"[日线数据] {code} 获取失败")
                return None
            return df
        except Exception as e:
            logger.debug(f"[日线数据] {code} 获取异常: {e}")
            return None

    def analyze_stocks(self, stock_codes: List[str], top_n: int = 2) -> List[StockScore]:
        """
        使用 StockTrendAnalyzer 分析股票并打分

        Args:
            stock_codes: 股票代码列表
            top_n: 返回前 N 名

        Returns:
            按买入信号优先级+得分排序的 StockScore 列表
        """
        logger.info(f"[完整分析] 开始分析 {len(stock_codes)} 只股票...")

        scores = []
        success_count = 0
        fail_count = 0

        for i, code in enumerate(stock_codes):
            try:
                # 获取历史日线数据（用于技术分析）
                df = self._get_stock_daily_data(code, days=120)
                if df is None or len(df) < 30:
                    fail_count += 1
                    continue

                # 使用 StockTrendAnalyzer 进行完整分析
                result = self.trend_analyzer.analyze(df, code)

                # 从分析结果获取股票名称
                name = result.code  # StockTrendAnalyzer 不返回名称，使用代码

                # 创建 StockScore 对象
                score = StockScore(
                    code=code,
                    name=name,
                    price=result.current_price,
                    change_pct=result.bias_ma5,  # 近似涨跌幅
                    ma5=result.ma5,
                    ma10=result.ma10,
                    ma20=result.ma20,
                    bias_ma5=result.bias_ma5,
                    total_score=result.signal_score,
                    buy_signal=result.buy_signal,
                    analysis_result=result,
                    is_bullish=result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL],
                    trend_strength=result.trend_strength,
                )

                scores.append(score)
                success_count += 1

                # 每 20 只股票打印一次进度
                if (i + 1) % 20 == 0:
                    logger.info(f"[完整分析] 进度: {i + 1}/{len(stock_codes)}, 成功: {success_count}, 失败: {fail_count}")

            except Exception as e:
                fail_count += 1
                logger.debug(f"[完整分析] {code} 分析异常: {e}")
                continue

        logger.info(f"[完整分析] 完成! 成功: {success_count}, 失败: {fail_count}")

        if not scores:
            logger.warning("[完整分析] 未能分析任何股票")
            return []

        # 筛选买入信号为 STRONG_BUY 或 BUY 的股票
        buy_signals = [BuySignal.STRONG_BUY, BuySignal.BUY]
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
        logger.info(f"🎯 股票筛选器启动（盘后分析模式）")
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
                f"   收盘价: {stock.price:.2f}\n"
                f"   MA5: {stock.ma5 or 'N/A':.2f} | MA10: {stock.ma10 or 'N/A':.2f} | MA20: {stock.ma20 or 'N/A':.2f}\n"
                f"   乖离率: {stock.bias_ma5 or 0:+.2f}% | 趋势强度: {stock.trend_strength:.0f}/100"
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
        f"## 📋 筛选结果（盘后分析）",
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
            f"| 收盘价 | {stock.price:.2f} 元 | |",
            f"| MA5 | {stock.ma5 or 'N/A':.2f} | |",
            f"| MA10 | {stock.ma10 or 'N/A':.2f} | |",
            f"| MA20 | {stock.ma20 or 'N/A':.2f} | |",
            f"| 乖离率(MA5) | {stock.bias_ma5 or 0:+.2f}% | {'✅安全' if abs(stock.bias_ma5 or 0) <= 5 else '⚠️偏高'} |",
            f"| 趋势强度 | {stock.trend_strength:.0f}/100 | |",
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
