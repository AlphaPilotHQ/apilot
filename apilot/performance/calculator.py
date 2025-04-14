"""
Performance Calculation Module

Focuses on calculating Overview and Key Metrics indicators:
- Overview: Backtest Period, Initial Capital, Final Capital, Total Return, Win Rate, Profit/Loss Ratio
- Key Metrics: Annualized Return, Max Drawdown, Sharpe Ratio, Turnover
"""

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from apilot.core import TradeData

logger = logging.getLogger(__name__)


def calculate_daily_results(
    trades: dict[str, TradeData], daily_data: dict[date, Any], sizes: dict[str, float]
) -> pd.DataFrame:
    """
    Calculate daily trading results

    Args:
        trades: Dictionary of trade data
        daily_data: Dictionary of daily market data
        sizes: Dictionary of contract multipliers

    Returns:
        DataFrame containing daily results
    """
    if not trades or not daily_data:
        logger.info("Insufficient data for calculating daily results")
        return pd.DataFrame()

    # Assign trades to their corresponding trading days
    daily_trades = {}
    for trade in trades.values():
        trade_date = trade.datetime.date()
        if trade_date not in daily_trades:
            daily_trades[trade_date] = []
        daily_trades[trade_date].append(trade)

    # Calculate daily results
    daily_results = []

    for current_date in sorted(daily_data.keys()):
        # Initialize results for the current day
        result = {
            "date": current_date,
            "trades": daily_trades.get(current_date, []),
            "trade_count": len(daily_trades.get(current_date, [])),
            "turnover": 0.0,
            "pnl": 0.0,
        }

        # Calculate trade profits and capital changes
        # Simplified handling here, specific implementation should be refined based on actual requirements

        # Add to the results list
        daily_results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(daily_results)
    if not df.empty:
        df.set_index("date", inplace=True)

    return df


def calculate_trade_metrics(trades: list[TradeData]) -> dict[str, float]:
    """
    Calculate trade-related metrics

    Args:
        trades: List of trades

    Returns:
        Dictionary of trade metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
        }

    # Analyze trades based on direction and opening/closing flags
    # First group by trading pair and direction
    position_trades = {}  # {symbol: {direction: [trades]}}

    # Organize trade pairing
    for trade in trades:
        symbol = trade.symbol
        # Process direction, supporting Chinese, English and enum values
        if hasattr(trade.direction, "value"):
            direction = trade.direction.value
        else:
            direction = str(trade.direction)

        # Convert Chinese direction terms to English (for compatibility)
        # "多"/"买" = long/buy, "空"/"卖" = short/sell
        if direction in ["多", "买"]:  # Chinese: "long", "buy"
            direction = "LONG"
        elif direction in ["空", "卖"]:  # Chinese: "short", "sell"
            direction = "SHORT"

        if symbol not in position_trades:
            position_trades[symbol] = {"LONG": [], "SHORT": []}

        position_trades[symbol][direction].append(trade)

    # Collect all trades
    closed_trades = []

    for _symbol, directions in position_trades.items():
        # Analyze long and short trades
        for _direction, trades_list in directions.items():
            # Sort trades by time
            sorted_trades = sorted(trades_list, key=lambda t: t.datetime)

            for trade in sorted_trades:
                # Check if the trade has profit/loss record
                if not hasattr(trade, "profit"):
                    # Set trade without profit attribute to 0 (neutral)
                    trade.profit = 0

                closed_trades.append(trade)

                # Record trade's profit/loss situation
                profit_value = getattr(trade, "profit", 0)
                if profit_value != 0:
                    logger.debug(
                        f"Trade {trade.tradeid} profit/loss: {profit_value:.2f}"
                    )

    # Calculate win/loss statistics
    winning_trades = [t for t in closed_trades if getattr(t, "profit", 0) > 0]
    losing_trades = [t for t in closed_trades if getattr(t, "profit", 0) < 0]
    neutral_trades = [t for t in closed_trades if getattr(t, "profit", 0) == 0]

    # Identify opening and closing trades
    # In this system, trades with profit=0 are mostly opening trades
    opening_trades = len(neutral_trades)
    closing_trades = len(winning_trades) + len(losing_trades)

    # Only consider trades with actual profit/loss (exclude opening trades)
    true_total_trades = closing_trades
    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    # Detailed logging TODO: Change to parameters + results
    logger.info(
        f"Trade statistics: Total trades {len(closed_trades)}, Profitable {win_count}, Loss {loss_count}, "
        f"Opening/Neutral {opening_trades}, Effective trades {true_total_trades}"
    )

    # Win rate only considers trades with profit/loss
    win_rate = (win_count / true_total_trades * 100) if true_total_trades > 0 else 0

    # Calculate profit factor (total profit / total loss)
    total_profit = sum(getattr(t, "profit", 0) for t in winning_trades)
    total_loss = abs(sum(getattr(t, "profit", 0) for t in losing_trades))

    # Calculate profit/loss metrics
    # Calculate total profit/loss ratio - total profit/total loss
    profit_loss_ratio = (total_profit / total_loss) if total_loss > 0 else float("inf")

    # Calculate averages (for reporting only)
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
    Calculate and return complete performance statistics

    Args:
        df: DataFrame containing daily results
        trades: List of trades
        capital: Initial capital
        annual_days: Number of trading days per year

    Returns:
        Dictionary containing performance statistics
    """
    # Initialize statistics
    stats = {
        # Overview section
        "start_date": "",
        "end_date": "",
        "total_days": 0,
        "initial_capital": capital,
        "final_capital": 0,
        "total_return": 0,
        "win_rate": 0,
        "profit_factor": 0,
        # Key Metrics section
        "annual_return": 0,
        "max_drawdown": 0,
        "sharpe_ratio": 0,
        "total_turnover": 0,
        # Additional profit/loss analysis
        "total_profit": 0,
        "total_loss": 0,
        "avg_profit": 0,
        "avg_loss": 0,
        "profit_days": 0,
        "loss_days": 0,
    }

    # If there's no data, return empty results
    if df is None or df.empty:
        logger.warning("No available trading data")
        return stats

    # Calculate basic statistics

    # 1. Time range
    stats["start_date"] = df.index[0]
    stats["end_date"] = df.index[-1]
    stats["total_days"] = len(df)

    # 2. Capital changes
    if "balance" in df.columns:
        stats["final_capital"] = df["balance"].iloc[-1]
        stats["total_return"] = ((stats["final_capital"] / capital) - 1) * 100

    # 3. Return metrics
    if "return" in df.columns:
        daily_returns = df["return"].values
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            stats["sharpe_ratio"] = (
                np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(annual_days)
            )
            stats["annual_return"] = (
                stats["total_return"] / stats["total_days"] * annual_days
            )

        # Calculate profitable days and loss days
        if "net_pnl" in df.columns:
            stats["profit_days"] = (df["net_pnl"] > 0).sum()
            stats["loss_days"] = (df["net_pnl"] < 0).sum()

    # 4. Drawdown
    if "ddpercent" in df.columns:
        stats["max_drawdown"] = df["ddpercent"].min()

    # 5. Trade-related metrics
    if "turnover" in df.columns:
        stats["total_turnover"] = df["turnover"].sum()
        # Calculate turnover ratio (total transaction amount/initial capital)
        if capital > 0:
            stats["turnover_ratio"] = stats["total_turnover"] / capital
        else:
            stats["turnover_ratio"] = 0.0

    # 6. Trade analysis
    if trades:
        # Calculate trade-related metrics
        trade_metrics = calculate_trade_metrics(trades)

        # Update statistics
        stats.update(trade_metrics)

        # Ensure win rate and profit/loss ratio display in Overview section
        stats["win_rate"] = trade_metrics.get("win_rate", 0)
        stats["profit_loss_ratio"] = trade_metrics.get("profit_loss_ratio", 0)

    # Clean invalid values
    stats = {
        k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in stats.items()
    }

    return stats
