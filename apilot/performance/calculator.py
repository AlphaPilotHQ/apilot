"""
Performance calculation module for strategy backtesting.
"""

from datetime import date

import numpy as np
from pandas import DataFrame

from apilot.core import Direction, TradeData
from apilot.utils import get_logger

logger = get_logger()


class DailyResult:
    def __init__(self, date: date) -> None:
        """
        Initialize daily result
        """
        self.date: date = date
        self.close_prices: dict[str, float] = {}
        self.pre_closes: dict[str, float] = {}

        self.trades: list[TradeData] = []
        self.trade_count: int = 0

        self.start_poses: dict[str, float] = {}
        self.end_poses: dict[str, float] = {}

        self.turnover: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """
        Add trade to daily result
        """
        self.trades.append(trade)

    def add_close_price(self, symbol: str, price: float) -> None:
        """
        Add closing price for a symbol
        """
        self.close_prices[symbol] = price

    def calculate_pnl(
        self,
        pre_closes: dict[str, float],
        start_poses: dict[str, float],
        sizes: dict[str, float],
    ) -> None:
        """
        Calculate profit and loss
        """
        self.pre_closes = pre_closes
        self.start_poses = start_poses.copy()
        self.end_poses = start_poses.copy()

        # Calculate holding PnL
        self.holding_pnl = 0
        for symbol in self.pre_closes.keys():
            pre_close = self.pre_closes.get(symbol, 0)
            if not pre_close:
                pre_close = 1  # Avoid division by zero

            start_pos = self.start_poses.get(symbol, 0)
            size = sizes.get(symbol, 1)
            close_price = self.close_prices.get(symbol, pre_close)

            symbol_holding_pnl = start_pos * (close_price - pre_close) * size
            self.holding_pnl += symbol_holding_pnl

        # Calculate trading PnL
        self.trade_count = len(self.trades)
        self.trading_pnl = 0
        self.turnover = 0

        for trade in self.trades:
            symbol = trade.symbol
            pos_change = (
                trade.volume if trade.direction == Direction.LONG else -trade.volume
            )

            if symbol in self.end_poses:
                self.end_poses[symbol] += pos_change
            else:
                self.end_poses[symbol] = pos_change

            size = sizes.get(symbol, 1)
            close_price = self.close_prices.get(symbol, trade.price)

            turnover = trade.volume * size * trade.price
            self.trading_pnl += pos_change * (close_price - trade.price) * size

            self.turnover += turnover

        # Calculate net PnL (no commission fees)
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl


class PerformanceCalculator:
    def __init__(self, capital: float = 0, annual_days: int = 240):
        """
        Initialize performance calculator

        Args:
            capital: Initial capital for backtesting
            annual_days: Trading days in a year
        """
        self.capital = capital
        self.annual_days = annual_days
        self.daily_df = None

    def calculate_result(
        self, trades: dict, daily_results: dict, sizes: dict
    ) -> DataFrame:
        """
        Calculate backtest results

        Args:
            trades: Dictionary of trades
            daily_results: Dictionary of daily results
            sizes: Dictionary of contract sizes

        Returns:
            DataFrame with daily performance data
        """
        if not trades:
            logger.info("Backtest trade records are empty")
            return DataFrame()

        # Add trade data to daily results
        for trade in trades.values():
            d = trade.datetime.date()
            daily_result = daily_results.get(d, None)
            if daily_result:
                daily_result.add_trade(trade)

        # Iterate and calculate daily results
        pre_closes = {}
        start_poses = {}
        for daily_result in daily_results.values():
            daily_result.calculate_pnl(
                pre_closes,
                start_poses,
                sizes,
            )
            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        # Generate DataFrame
        first_result = next(iter(daily_results.values()))
        results = {
            key: [getattr(dr, key) for dr in daily_results.values()]
            for key in first_result.__dict__
        }

        self.daily_df = DataFrame.from_dict(results).set_index("date")
        logger.info("Daily mark-to-market P&L calculation completed")
        return self.daily_df


class TradeAnalyzer:
    """Analyze trade performance"""
    
    def __init__(self, trades=None):
        """
        Initialize trade analyzer
        
        Args:
            trades: List of TradeData objects
        """
        self.trades = trades or []
        self.trade_stats = {}
        
    def analyze(self):
        """Analyze trades and calculate statistics"""
        if not self.trades:
            return {}
            
        # TODO: Implement trade analysis
        return {}


def calculate_statistics(
    df: DataFrame = None,
    capital: float = 0,
    annual_days: int = 240,
    output: bool = True,
) -> tuple:
    """
    Calculate statistics from performance data

    Args:
        df: DataFrame with daily performance data
        capital: Initial capital for backtesting
        annual_days: Trading days in a year
        output: Whether to print statistics to log

    Returns:
        Dictionary with calculated statistics and modified DataFrame
    """
    stats = {
        "start_date": "",
        "end_date": "",
        "total_days": 0,
        "profit_days": 0,
        "loss_days": 0,
        "capital": capital,
        "end_balance": 0,
        "max_ddpercent": 0,
        "total_turnover": 0,
        "total_trade_count": 0,
        "total_return": 0,
        "annual_return": 0,
        "sharpe_ratio": 0,
        "return_drawdown_ratio": 0,
    }

    # Return early if no data
    if df is None or df.empty:
        logger.warning("No trading data available")
        return stats, df

    # Make a copy to avoid modifying original data
    df = df.copy()

    # Calculate balance
    df["balance"] = df["net_pnl"].cumsum() + capital

    # Calculate daily returns
    pre_balance = df["balance"].shift(1)
    pre_balance.iloc[0] = capital
    x = df["balance"] / pre_balance
    x[x <= 0] = np.nan
    df["return"] = np.log(x).fillna(0)

    # Calculate drawdown
    df["highlevel"] = df["balance"].cummax()
    df["ddpercent"] = (df["balance"] - df["highlevel"]) / df["highlevel"] * 100

    # Check for bankruptcy
    if not (df["balance"] > 0).all():
        logger.warning("Bankruptcy detected during backtest")
        return stats, df

    # Calculate basic statistics
    stats.update(
        {
            "start_date": df.index[0],
            "end_date": df.index[-1],
            "total_days": len(df),
            "profit_days": (df["net_pnl"] > 0).sum(),
            "loss_days": (df["net_pnl"] < 0).sum(),
            "end_balance": df["balance"].iloc[-1],
            "max_ddpercent": df["ddpercent"].min(),
            "total_turnover": df["turnover"].sum(),
            "total_trade_count": df["trade_count"].sum(),
        }
    )

    # Calculate return metrics
    stats["total_return"] = (stats["end_balance"] / capital - 1) * 100
    stats["annual_return"] = stats["total_return"] / stats["total_days"] * annual_days

    # Calculate risk-adjusted metrics
    daily_returns = df["return"].values
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        stats["sharpe_ratio"] = (
            np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(annual_days)
        )

    if stats["max_ddpercent"] < 0:
        stats["return_drawdown_ratio"] = -stats["total_return"] / stats["max_ddpercent"]

    # Clean up invalid values
    stats = {
        k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in stats.items()
    }

    if output:
        logger.info(f"Trade day:\t{stats['start_date']} - {stats['end_date']}")
        logger.info(f"Profit days:\t{stats['profit_days']}")
        logger.info(f"Loss days:\t{stats['loss_days']}")
        logger.info(f"Initial capital:\t{capital:.2f}")
        logger.info(f"Final capital:\t{stats['end_balance']:.2f}")
        logger.info(f"Total return:\t{stats['total_return']:.2f}%")
        logger.info(f"Annual return:\t{stats['annual_return']:.2f}%")
        logger.info(f"Max drawdown:\t{stats['max_ddpercent']:.2f}%")
        logger.info(f"Total turnover:\t{stats['total_turnover']:.2f}")
        logger.info(f"Total trades:\t{stats['total_trade_count']}")
        logger.info(f"Sharpe ratio:\t{stats['sharpe_ratio']:.2f}")
        logger.info(f"Return/Drawdown:\t{stats['return_drawdown_ratio']:.2f}")

    return stats, df
