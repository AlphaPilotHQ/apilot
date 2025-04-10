"""
性能计算模块

专注于计算 Overview 和 Key Metrics 指标:
- Overview: Backtest Period, Initial Capital, Final Capital, Total Return, Win Rate, Profit/Loss Ratio
- Key Metrics: Annualized Return, Max Drawdown, Sharpe Ratio, Turnover
"""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from apilot.core import TradeData
from apilot.utils import get_logger

logger = get_logger()


def calculate_daily_results(
    trades: dict[str, TradeData], daily_data: dict[date, Any], sizes: dict[str, float]
) -> pd.DataFrame:
    """
    计算每日交易结果

    Args:
        trades: 交易数据字典
        daily_data: 每日行情数据字典
        sizes: 合约乘数字典

    Returns:
        包含每日结果的DataFrame
    """
    if not trades or not daily_data:
        logger.info("计算每日结果所需数据不足")
        return pd.DataFrame()

    # 将交易分配到对应的交易日
    daily_trades = {}
    for trade in trades.values():
        trade_date = trade.datetime.date()
        if trade_date not in daily_trades:
            daily_trades[trade_date] = []
        daily_trades[trade_date].append(trade)

    # 计算每日结果
    daily_results = []

    for current_date in sorted(daily_data.keys()):
        # 初始化当天结果
        result = {
            "date": current_date,
            "trades": daily_trades.get(current_date, []),
            "trade_count": len(daily_trades.get(current_date, [])),
            "turnover": 0.0,
            "pnl": 0.0,
        }

        # 计算交易盈亏和资金变化
        # 这里简化处理，具体实现应根据实际需求完善

        # 添加到结果列表
        daily_results.append(result)

    # 转换为DataFrame
    df = pd.DataFrame(daily_results)
    if not df.empty:
        df.set_index("date", inplace=True)

    return df


def calculate_trade_metrics(trades: list[TradeData]) -> dict[str, float]:
    """
    计算交易相关指标

    Args:
        trades: 交易列表

    Returns:
        交易指标字典
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
        }

    # 根据交易方向和开平标志分析交易
    # 先按照交易对和方向分组
    position_trades = {}  # {symbol: {direction: [trades]}}

    # 组织交易配对
    for trade in trades:
        symbol = trade.symbol
        # 处理方向，支持中英文和枚举值
        if hasattr(trade.direction, "value"):
            direction = trade.direction.value
        else:
            direction = str(trade.direction)

        # 将中文方向转换为英文（兼容处理）
        if direction in ["多", "买"]:
            direction = "LONG"
        elif direction in ["空", "卖"]:
            direction = "SHORT"

        if symbol not in position_trades:
            position_trades[symbol] = {"LONG": [], "SHORT": []}

        position_trades[symbol][direction].append(trade)

    # 收集所有交易
    closed_trades = []

    for _symbol, directions in position_trades.items():
        # 分析多头和空头交易
        for direction, trades_list in directions.items():
            # 按时间排序交易
            sorted_trades = sorted(trades_list, key=lambda t: t.datetime)
            
            for trade in sorted_trades:
                # 检查交易是否有盈亏记录
                if not hasattr(trade, "profit"):
                    # 没有profit属性的交易设置为0（中性）
                    trade.profit = 0
                
                closed_trades.append(trade)
                
                # 记录交易的盈亏情况
                profit_value = getattr(trade, "profit", 0)
                if profit_value != 0:
                    logger.debug(f"交易 {trade.tradeid} 盈亏: {profit_value:.2f}")

    # 计算胜负统计
    winning_trades = [t for t in closed_trades if getattr(t, "profit", 0) > 0]
    losing_trades = [t for t in closed_trades if getattr(t, "profit", 0) < 0]
    neutral_trades = [t for t in closed_trades if getattr(t, "profit", 0) == 0]

    # 识别开仓和平仓交易
    # 在这个系统中，profit为0的大多是开仓交易
    opening_trades = len(neutral_trades)
    closing_trades = len(winning_trades) + len(losing_trades)
    
    # 只考虑有实际盈亏的交易（排除开仓交易）
    true_total_trades = closing_trades
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    # 详细记录日志
    logger.info(
        f"交易统计: 总交易 {len(closed_trades)}, 盈利 {win_count}, 亏损 {loss_count}, "
        f"开仓/持平 {opening_trades}, 有效交易 {true_total_trades}"
    )

    # 胜率只考虑有盈亏的交易
    win_rate = (win_count / true_total_trades * 100) if true_total_trades > 0 else 0

    # 计算盈利因子(总盈利/总亏损)
    total_profit = sum(getattr(t, "profit", 0) for t in winning_trades)
    total_loss = abs(sum(getattr(t, "profit", 0) for t in losing_trades))

    # 计算盈亏指标
    # 计算总盈亏比 - 总盈利/总亏损
    profit_loss_ratio = (total_profit / total_loss) if total_loss > 0 else float("inf")
    
    # 计算平均值（仅用于报告）
    avg_profit = total_profit / win_count if win_count > 0 else 0
    avg_loss = total_loss / loss_count if loss_count > 0 else 0

    return {
        "total_trades": len(trades),
        "closed_trades": len(closed_trades),
        "effective_trades": true_total_trades,
        "opening_trades": opening_trades,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "total_profit": total_profit,
        "total_loss": -total_loss,
    }


def calculate_statistics(
    df: pd.DataFrame | None = None,
    trades: list[TradeData] | None = None,
    capital: float = 0,
    annual_days: int = 240,
) -> dict[str, Any]:
    """
    计算并返回完整的性能统计指标

    Args:
        df: 包含每日结果的DataFrame
        trades: 交易列表
        capital: 初始资金
        annual_days: 年交易日数

    Returns:
        包含性能统计指标的字典
    """
    # 初始化统计指标
    stats = {
        # Overview部分
        "start_date": "",
        "end_date": "",
        "total_days": 0,
        "initial_capital": capital,
        "final_capital": 0,
        "total_return": 0,
        "win_rate": 0,
        "profit_factor": 0,
        # Key Metrics部分
        "annual_return": 0,
        "max_drawdown": 0,
        "sharpe_ratio": 0,
        "total_turnover": 0,
        # 额外的盈亏分析
        "total_profit": 0,
        "total_loss": 0,
        "avg_profit": 0,
        "avg_loss": 0,
        "profit_days": 0,
        "loss_days": 0,
    }

    # 如果没有数据，返回空结果
    if df is None or df.empty:
        logger.warning("没有可用的交易数据")
        return stats

    # 计算基本统计数据

    # 1. 时间范围
    stats["start_date"] = df.index[0]
    stats["end_date"] = df.index[-1]
    stats["total_days"] = len(df)

    # 2. 资金变化
    if "balance" in df.columns:
        stats["final_capital"] = df["balance"].iloc[-1]
        stats["total_return"] = ((stats["final_capital"] / capital) - 1) * 100

    # 3. 回报指标
    if "return" in df.columns:
        daily_returns = df["return"].values
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            stats["sharpe_ratio"] = (
                np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(annual_days)
            )
            stats["annual_return"] = (
                stats["total_return"] / stats["total_days"] * annual_days
            )

        # 计算盈利天数和亏损天数
        if "net_pnl" in df.columns:
            stats["profit_days"] = (df["net_pnl"] > 0).sum()
            stats["loss_days"] = (df["net_pnl"] < 0).sum()

    # 4. 回撤
    if "ddpercent" in df.columns:
        stats["max_drawdown"] = df["ddpercent"].min()

    # 5. 交易相关指标
    if "turnover" in df.columns:
        stats["total_turnover"] = df["turnover"].sum()
        # 计算周转率（总成交金额/初始资金）
        if capital > 0:
            stats["turnover_ratio"] = stats["total_turnover"] / capital
        else:
            stats["turnover_ratio"] = 0.0

    # 6. 交易分析
    if trades:
        # 计算交易相关指标
        trade_metrics = calculate_trade_metrics(trades)

        # 更新统计数据
        stats.update(trade_metrics)

        # 确保胜率和盈亏比显示在Overview部分
        stats["win_rate"] = trade_metrics.get("win_rate", 0)
        stats["profit_loss_ratio"] = trade_metrics.get("profit_loss_ratio", 0)

    # 清理无效值
    stats = {
        k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in stats.items()
    }

    return stats
