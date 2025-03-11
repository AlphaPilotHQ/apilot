"""
Test script for backtesting functionality using local data.
"""

import os
from datetime import datetime
import pandas as pd
from typing import Any, List

from vnpy.trader.constant import Interval, Direction, Exchange
from vnpy.trader.object import BarData, TickData
from vnpy.trader.utility import BarGenerator
from vnpy.cta_strategy.template import CtaTemplate
from vnpy.cta_strategy.backtesting import BacktestingEngine


class TestStrategy(CtaTemplate):

    # Strategy parameters
    fast_window = 20     # 调整为更短周期，增加交易频率
    slow_window = 200    # 保持中等长度的慢速均线
    
    # Variables
    fast_ma = 0.0
    slow_ma = 0.0
    
    fast_ma_array = []
    slow_ma_array = []
    
    parameters = ["fast_window", "slow_window"]
    variables = ["fast_ma", "slow_ma"]
    
    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict,
    ):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.fast_ma_array = []
        self.slow_ma_array = []
        self.bg = BarGenerator(self.on_bar)  # 添加BarGenerator初始化
    
    def on_init(self):
        """Called when strategy is initialized."""
        self.write_log("策略初始化")
        
    def on_start(self):
        """Called when strategy is started."""
        self.write_log("策略启动")
    
    def on_stop(self):
        """Called when strategy is stopped."""
        self.write_log("策略停止")
    
    def on_tick(self, tick: TickData):
        """Called on every tick update."""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """Called on every bar update."""
        # Update fast MA
        self.fast_ma_array.append(bar.close_price)
        if len(self.fast_ma_array) > self.fast_window:
            self.fast_ma_array.pop(0)
        self.fast_ma = sum(self.fast_ma_array) / len(self.fast_ma_array)

        # Update slow MA
        self.slow_ma_array.append(bar.close_price)
        if len(self.slow_ma_array) > self.slow_window:
            self.slow_ma_array.pop(0)
        self.slow_ma = sum(self.slow_ma_array) / len(self.slow_ma_array)

        # No trading until both MAs are calculated
        if len(self.fast_ma_array) < self.fast_window or len(self.slow_ma_array) < self.slow_window:
            return

        # 全仓交易策略 - 先检查引擎状态
        if hasattr(self, 'inited') and not self.inited:
            return

        # 使用账户价值计算交易量
        if self.fast_ma > self.slow_ma and not self.pos:  # 金叉做多
            # 获取账户可用资金
            # 注意：在回测中，我们通过cta_engine访问资金状态
            capital = 0
            if hasattr(self.cta_engine, 'capital'):
                capital = self.cta_engine.capital
            
            # 如果没有可用资金信息，则使用默认值(初始100万)
            if not capital:
                capital = 1000000
                
            # 计算可以买入的最大数量 (全仓)
            # 预留1%资金作为缓冲，避免因计算精度问题导致的下单失败
            max_volume = capital * 0.5 / bar.close_price
            
            # 买入
            self.buy(bar.close_price, max_volume)
            self.write_log(f"全仓买入: {max_volume:.2f} 手，价格: {bar.close_price:.2f}, 金额: {max_volume * bar.close_price:.2f}")
        
        elif self.fast_ma < self.slow_ma and self.pos > 0:  # 死叉卖出
            # 全部卖出
            self.sell(bar.close_price, abs(self.pos))
            self.write_log(f"全仓卖出: {abs(self.pos):.2f} 手，价格: {bar.close_price:.2f}, 金额: {abs(self.pos) * bar.close_price:.2f}")


def load_bar_data_from_csv(csv_path, symbol, exchange, interval, start, end):
    """
    Load bar data from csv file.
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return []
    
    # Read csv file
    df = pd.read_csv(csv_path)
    
    # Print column names for debugging
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Check and convert datetime column - try different possible column names
    datetime_col = None
    for col_name in ["datetime", "date", "time", "timestamp", "Date", "Time", "Datetime", "TimeStamp", "candle_begin_time"]:
        if col_name in df.columns:
            datetime_col = col_name
            break
    
    if datetime_col is None:
        print("Error: No datetime column found in CSV file")
        print("Available columns:", df.columns.tolist())
        return []
    
    # Convert datetime column
    df["datetime"] = pd.to_datetime(df[datetime_col])
    
    # Filter by date range
    df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
    
    # Map column names to expected names if necessary
    price_cols = {
        "open": ["open", "Open", "OPEN"],
        "high": ["high", "High", "HIGH"],
        "low": ["low", "Low", "LOW"],
        "close": ["close", "Close", "CLOSE"],
        "volume": ["volume", "Volume", "VOLUME", "vol", "Vol"]
    }
    
    col_mapping = {}
    for target, possibilities in price_cols.items():
        for possibility in possibilities:
            if possibility in df.columns:
                col_mapping[target] = possibility
                break
        
        if target not in col_mapping:
            print(f"Warning: Column '{target}' not found in CSV")
            return []
    
    # Create bar data objects
    bars = []
    for _, row in df.iterrows():
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            datetime=row["datetime"],
            open_price=row[col_mapping["open"]],
            high_price=row[col_mapping["high"]],
            low_price=row[col_mapping["low"]],
            close_price=row[col_mapping["close"]],
            volume=row[col_mapping["volume"]],
            gateway_name="CSV"
        )
        bars.append(bar)
    
    print(f"Loaded {len(bars)} bars from CSV")
    return bars


def test_single_backtest():
    """Test running a single backtest with local data"""
    print("Testing single backtest with local data...")
    
    engine = BacktestingEngine()
    
    try:
        # Set parameters
        symbol = "SOL-USDT"
        exchange = Exchange.BINANCE
        interval = Interval.MINUTE
        start = datetime(2023, 1, 1)  # 修改为2023年1月1日
        end = datetime(2023, 1, 31)  # 修改为2023年12月31日
        
        engine.set_parameters(
            vt_symbol=f"{symbol}.{exchange.value}",
            interval=interval,
            start=start,
            end=end,
            rate=0.0001,      # 降低手续费率
            slippage=0.0001,  # 大幅降低滑点
            size=1,           # 合约乘数保持为1
            pricetick=0.01,   # 价格精度
            capital=1000000   # 增加初始资金
        )
        
        # Load data from CSV
        csv_path = "/Users/bobbyding/Documents/GitHub/vnpy/SOL-USDT.csv"
        bars = load_bar_data_from_csv(csv_path, symbol, exchange, interval, start, end)
        
        if not bars:
            print("No data loaded, aborting test")
            return False
        
        # Add bars to engine's history_data
        engine.history_data = bars
        
        # Add strategy
        engine.add_strategy(TestStrategy, {
            "fast_window": 5,
            "slow_window": 20
        })
        
        # Run backtest
        engine.run_backtesting()
        
        # Calculate result
        engine.calculate_result()
        stats = engine.calculate_statistics()  # 使用calculate_statistics方法的返回值
        
        # Get results
        df = engine.daily_df  # 使用 daily_df 属性
        
        print("Single backtest completed!")
        print("Statistics:")
        if stats:
            for key, value in stats.items():
                print(f"{key}: {value}")
        else:
            print("No statistics available - likely no trades were made")
        
        # 输出详细的交易记录
        print("\n=== 交易记录 ===")
        if engine.trades:
            for i, (vt_tradeid, trade) in enumerate(engine.trades.items()):
                print(f"交易 #{i+1}: {'买入' if trade.direction == Direction.LONG else '卖出'} {trade.volume} 手, 价格: {trade.price}, 时间: {trade.datetime}")
        else:
            print("没有交易记录")
            
        return True
    except Exception as e:
        print(f"Single backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting backtesting tests...\n")
    
    single_test_result = test_single_backtest()
    
    if single_test_result:
        print("\nTest passed successfully!")
    else:
        print("\nTest failed. Please check the error messages above.")
