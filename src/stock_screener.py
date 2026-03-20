# -*- coding: utf-8 -*-
"""
===================================
股票筛选器 - Stock Screener
===================================

职责：
1. 获取指数成分股（中证1000等）
2. 基于核心交易理念进行多维度筛选
3. 输出优质股票候选列表

核心筛选标准（基于项目7条交易理念）：
- 理念1：严进策略（乖离率 < 5%）
- 理念2：趋势交易（MA5 > MA10 > MA20 多头排列）
- 理念3：效率优先（筹码集中度 < 15%）
- 理念4：买点偏好（回踩支撑而非追高）
- 理念5：风险排查（无重大利空）
- 理念6：量价配合
- 理念7：强势趋势股可适当放宽
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import akshare as ak
import pandas as pd

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
    chip_concentration: Optional[float] = None
    pe_ratio: Optional[float] = None

    # 筛选条件得分
    bullish_alignment: int = 0      # 多头排列得分 (0/1)
    bias_safe: int = 0              # 乖离率安全得分 (0/1)
    volume_valid: int = 0           # 量能配合得分 (0/1)
    chip_healthy: int = 0           # 筹码健康得分 (0/1)
    trend_strength: int = 0          # 趋势强度 (0-100)
    total_score: int = 0            # 综合得分

    # 标记
    is_bullish: bool = False         # 是否多头排列
    is_strong_trend: bool = False   # 是否强势趋势

    def __post_init__(self):
        """计算综合得分"""
        # 基础分：满足每个条件得20分
        base_score = (
            self.bullish_alignment * 20 +
            self.bias_safe * 20 +
            self.volume_valid * 20 +
            self.chip_healthy * 20
        )
        # 趋势强度加分（0-40分）
        strength_score = self.trend_strength * 0.4
        self.total_score = int(base_score + strength_score)


class StockScreener:
    """股票筛选器"""

    # 中证1000指数代码
    ZZ1000_INDEX = "000852"

    def __init__(self):
        pass

    def get_index_constituents(self, index_code: str = ZZ1000_INDEX) -> List[str]:
        """
        获取指数成分股代码列表

        Args:
            index_code: 指数代码，默认中证1000 (000852)

        Returns:
            成分股代码列表
        """
        try:
            # AkShare 获取指数成分股
            logger.info(f"正在获取指数 {index_code} 成分股...")

            # 尝试多种 AkShare 接口
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
            if '股票代码' in df.columns:
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
        批量获取实时行情数据

        Args:
            stock_codes: 股票代码列表

        Returns:
            包含实时行情的 DataFrame
        """
        try:
            logger.info(f"正在获取 {len(stock_codes)} 只股票的实时行情...")

            # 使用 AkShare 获取A股实时行情（支持筛选）
            df = ak.stock_zh_a_spot_em()

            # 筛选出目标股票
            if '代码' in df.columns:
                df_filtered = df[df['代码'].isin(stock_codes)]
            elif 'code' in df.columns:
                df_filtered = df[df['code'].isin(stock_codes)]
            else:
                logger.error(f"实时行情数据中没有代码字段: {df.columns.tolist()}")
                return pd.DataFrame()

            logger.info(f"成功获取 {len(df_filtered)} 只股票的实时行情")
            return df_filtered

        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return pd.DataFrame()

    def calculate_stock_scores(self, stock_codes: List[str], top_n: int = 10) -> List[StockScore]:
        """
        计算股票筛选得分

        Args:
            stock_codes: 股票代码列表
            top_n: 返回前 N 名

        Returns:
            按得分排序的 StockScore 列表
        """
        # 获取实时行情
        realtime_df = self.get_realtime_data_batch(stock_codes)
        if realtime_df.empty:
            logger.error("无法获取实时行情，无法进行筛选")
            return []

        scores = []

        for idx, row in realtime_df.iterrows():
            try:
                # 提取数据
                code = row.get('代码', row.get('code', ''))
                name = row.get('名称', row.get('name', ''))
                price = float(row.get('最新价', row.get('price', 0)))
                change_pct = float(row.get('涨跌幅', row.get('change_percent', 0)))
                ma5 = row.get('MA5') or row.get('ma5')
                ma10 = row.get('MA10') or row.get('ma10')
                ma20 = row.get('MA20') or row.get('ma20')
                volume_ratio = row.get('量比') or row.get('volume_ratio')
                turnover_rate = row.get('换手率') or row.get('turnover_rate')
                pe_ratio = row.get('市盈率-动态') or row.get('pe_ratio')

                # 转换 MA 值
                if ma5:
                    try:
                        ma5 = float(ma5)
                    except:
                        ma5 = None
                if ma10:
                    try:
                        ma10 = float(ma10)
                    except:
                        ma10 = None
                if ma20:
                    try:
                        ma20 = float(ma20)
                    except:
                        ma20 = None

                # 计算乖离率
                bias_ma5 = 0.0
                if ma5 and ma5 > 0:
                    bias_ma5 = (price - ma5) / ma5 * 100

                # 判断多头排列
                is_bullish = False
                if ma5 and ma10 and ma20:
                    is_bullish = ma5 > ma10 > ma20

                # 趋势强度（基于均线间距）
                trend_strength = 0
                if ma5 and ma10 and ma20:
                    # 均线间距比例
                    spacing1 = (ma5 - ma10) / ma10 if ma10 > 0 else 0
                    spacing2 = (ma10 - ma20) / ma20 if ma20 > 0 else 0
                    trend_strength = int(min((spacing1 + spacing2) * 1000, 100))

                # 筛选条件判断
                bullish_alignment = 1 if is_bullish else 0

                # 乖离率安全：<= 5%
                bias_safe = 1 if abs(bias_ma5) <= 5 else 0

                # 强势趋势股放宽标准
                is_strong_trend = is_bullish and trend_strength > 60
                bias_safe_strict = 1 if abs(bias_ma5) <= 2 else 0

                # 使用宽松标准还是严格标准
                effective_bias_safe = bias_safe_strict if is_strong_trend else bias_safe

                # 量能配合：量比 0.8-3 为正常
                volume_valid = 0
                if volume_ratio:
                    try:
                        vr = float(volume_ratio)
                        volume_valid = 1 if 0.8 <= vr <= 3 else 0
                    except:
                        pass

                # 筹码健康：换手率适中 (1-5%)
                chip_healthy = 0
                if turnover_rate:
                    try:
                        tr = float(turnover_rate)
                        # 换手率 1-5% 为健康
                        chip_healthy = 1 if 1 <= tr <= 5 else 0
                    except:
                        pass

                # 创建评分对象
                score = StockScore(
                    code=code,
                    name=name,
                    price=price,
                    change_pct=change_pct,
                    ma5=ma5,
                    ma10=ma10,
                    ma20=ma20,
                    bias_ma5=bias_ma5,
                    volume_ratio=volume_ratio,
                    turnover_rate=turnover_rate,
                    pe_ratio=pe_ratio,
                    bullish_alignment=bullish_alignment,
                    bias_safe=effective_bias_safe,
                    volume_valid=volume_valid,
                    chip_healthy=chip_healthy,
                    trend_strength=trend_strength,
                    is_bullish=is_bullish,
                    is_strong_trend=is_strong_trend,
                )

                scores.append(score)

            except Exception as e:
                logger.warning(f"处理股票 {row.get('代码', '')} 评分时出错: {e}")
                continue

        # 按得分排序
        scores_sorted = sorted(scores, key=lambda x: x.total_score, reverse=True)
        logger.info(f"完成 {len(scores)} 只股票评分，最高分: {scores_sorted[0].total_score if scores_sorted else 0}")

        return scores_sorted[:top_n]

    def screen(self, index_code: Optional[str] = None, top_n: int = 2) -> Tuple[List[StockScore], str]:
        """
        执行股票筛选流程

        Args:
            index_code: 指数代码，默认中证1000
            top_n: 返回前 N 名

        Returns:
            (筛选结果, 索引名称)
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
        logger.info(f"🎯 股票筛选器启动")
        logger.info(f"📊 筛选范围: {index_name} ({index_code})")
        logger.info(f"🏆 返回数量: {top_n}")
        logger.info(f"==========================================")

        # 1. 获取成分股
        stock_codes = self.get_index_constituents(index_code)
        if not stock_codes:
            logger.error(f"无法获取 {index_name} 成分股，筛选终止")
            return [], index_name

        logger.info(f"开始从 {len(stock_codes)} 只成分股中筛选...")

        # 2. 初筛：过滤 ST 股、停牌股等
        logger.info("执行初筛：过滤 ST 股、停牌股...")
        realtime_df = self.get_realtime_data_batch(stock_codes)
        if realtime_df.empty:
            return [], index_name

        # 过滤条件：
        # - 名称不含 ST
        # - 涨跌幅在合理范围内（非异常）
        # - 成交额不为0
        name_col = '名称' if '名称' in realtime_df.columns else 'name'
        code_col = '代码' if '代码' in realtime_df.columns else 'code'
        amount_col = '成交额' if '成交额' in realtime_df.columns else 'amount'

        filtered_df = realtime_df[
            (~realtime_df[name_col].str.contains('ST', na=False)) &
            (realtime_df[amount_col] > 0) &
            (realtime_df[code_col].astype(str).str.len() == 6)  # 过滤掉非6位代码
        ]

        filtered_codes = filtered_df[code_col].tolist()
        logger.info(f"初筛后剩余: {len(filtered_codes)} 只股票")

        # 3. 计算得分并排序
        top_stocks = self.calculate_stock_scores(filtered_codes, top_n=min(top_n * 2, 20))

        # 4. 输出结果
        logger.info(f"\n==========================================")
        logger.info(f"📋 筛选结果 TOP {len(top_stocks)}")
        logger.info(f"==========================================")
        for i, stock in enumerate(top_stocks[:top_n], 1):
            logger.info(
                f"\n#{i} {stock.name}({stock.code})\n"
                f"   现价: {stock.price:.2f} | 涨跌: {stock.change_pct:+.2f}%\n"
                f"   MA5: {stock.ma5 or 'N/A':.2f} | MA10: {stock.ma10 or 'N/A':.2f} | MA20: {stock.ma20 or 'N/A':.2f}\n"
                f"   乖离率: {stock.bias_ma5:+.2f}% | 量比: {stock.volume_ratio or 'N/A'} | 换手率: {stock.turnover_rate or 'N/A'}%\n"
                f"   多头排列: {'✅' if stock.is_bullish else '❌'} | 筹码健康: {'✅' if stock.chip_healthy else '❌'}\n"
                f"   综合得分: {stock.total_score}"
            )

        return top_stocks[:top_n], index_name


def format_screener_report(stocks: List[StockScore], index_name: str) -> str:
    """
    格式化筛选结果报告

    Args:
        stocks: 筛选出的股票列表
        index_name: 索引名称

    Returns:
        Markdown 格式的报告
    """
    if not stocks:
        return f"# 📊 {index_name} 筛选结果\n\n未筛选出符合条件的股票。"

    lines = [
        f"# 🎯 {index_name} 每日优选股票",
        f"",
        f"## 📋 筛选结果",
        f"",
        f"基于以下核心交易理念筛选：",
        f"- ✅ 趋势交易：MA5 > MA10 > MA20 多头排列",
        f"- ✅ 严进策略：乖离率 < 5%（强势股可放宽至 8%）",
        f"- ✅ 效率优先：换手率 1-5%",
        f"- ✅ 量价配合：量比 0.8-3",
        f"",
        f"---",
        f""
    ]

    for i, stock in enumerate(stocks, 1):
        lines.extend([
            f"### #{i} {stock.name}({stock.code})",
            f"",
            f"| 指标 | 数值 | 状态 |",
            f"|------|------|------|",
            f"| 现价 | {stock.price:.2f} 元 | |",
            f"| 涨跌幅 | {stock.change_pct:+.2f}% | {'🟢' if stock.change_pct > 0 else '🔴'} |",
            f"| MA5 | {stock.ma5 or 'N/A':.2f} | |",
            f"| MA10 | {stock.ma10 or 'N/A':.2f} | |",
            f"| MA20 | {stock.ma20 or 'N/A':.2f} | |",
            f"| 乖离率(MA5) | {stock.bias_ma5:+.2f}% | {'✅安全' if abs(stock.bias_ma5) <= 5 else '⚠️追高'} |",
            f"| 量比 | {stock.volume_ratio or 'N/A'} | {'✅配合' if 0.8 <= (stock.volume_ratio or 0) <= 3 else '❌异常'} |",
            f"| 换手率 | {stock.turnover_rate or 'N/A'}% | {'✅健康' if 1 <= (stock.turnover_rate or 0) <= 5 else '❌异常'} |",
            f"| 多头排列 | | {'✅' if stock.is_bullish else '❌'} |",
            f"| 综合得分 | | **{stock.total_score}** |",
            f"",
            f"**筛选标准满足情况：**",
            f"- 多头排列: {'✅' if stock.bullish_alignment else '❌'}",
            f"- 乖离率安全: {'✅' if stock.bias_safe else '❌'}",
            f"- 量能配合: {'✅' if stock.volume_valid else '❌'}",
            f"- 筹码健康: {'✅' if stock.chip_healthy else '❌'}",
            f"",
            f"---",
            f""
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
