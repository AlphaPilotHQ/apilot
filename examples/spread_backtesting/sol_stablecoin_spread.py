import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict

# 添加项目根目录到Python路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# from apilot.trader.optimize import OptimizationSetting
from apilot.trader.object import BarData, TradeData
from apilot.trader.constant import Exchange, Interval
from apilot.spread_strategy.base import SpreadData, LegData
from apilot.spread_strategy.template import SpreadAlgoTemplate, SpreadStrategyTemplate
from apilot.spread_strategy.strategies.statistical_arbitrage_strategy import StatisticalArbitrageStrategy
from apilot.spread_strategy.backtesting import BacktestingEngine, BacktestingMode, DailyResult, OptimizationSetting


# 直接从CSV文件加载K线数据
def load_bar_data_from_csv(file_path: str, symbol: str, exchange: Exchange = Exchange.LOCAL) -> List[BarData]:
    """
    从CSV文件直接加载K线数据
    """
    df = pd.read_csv(file_path)
    bars = []
    
    # 确保时间列是datetime格式
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    
    for _, row in df.iterrows():
        # 创建BarData对象
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=row['candle_begin_time'],
            interval=Interval.MINUTE,
            volume=float(row['volume']),
            open_price=float(row['open']),
            high_price=float(row['high']),
            low_price=float(row['low']),
            close_price=float(row['close']),
            gateway_name="CSV"
        )
        bars.append(bar)
    
    return bars


# 分析价差
def analyze_spread(sol_busd_file, sol_usdt_file):
    """
    分析SOL-BUSD和SOL-USDT之间的价差
    """
    df_busd = pd.read_csv(sol_busd_file)
    df_usdt = pd.read_csv(sol_usdt_file)
    
    # 确保时间列是datetime格式
    df_busd['candle_begin_time'] = pd.to_datetime(df_busd['candle_begin_time'])
    df_usdt['candle_begin_time'] = pd.to_datetime(df_usdt['candle_begin_time'])
    
    # 设置索引方便合并
    df_busd.set_index('candle_begin_time', inplace=True)
    df_usdt.set_index('candle_begin_time', inplace=True)
    
    # 合并数据
    merged = pd.DataFrame()
    merged['busd_close'] = df_busd['close']
    merged['usdt_close'] = df_usdt['close']
    
    # 计算价差
    merged['spread'] = merged['busd_close'] - merged['usdt_close']
    
    # 计算价差的统计特性
    mean_spread = merged['spread'].mean()
    std_spread = merged['spread'].std()
    
    # 绘制价差图
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(merged.index, merged['busd_close'], label='SOL-BUSD')
    plt.plot(merged.index, merged['usdt_close'], label='SOL-USDT')
    plt.title('SOL Price Comparison: BUSD vs USDT')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(merged.index, merged['spread'])
    plt.axhline(y=mean_spread, color='r', linestyle='-', label=f'Mean: {mean_spread:.6f}')
    plt.axhline(y=mean_spread + 2*std_spread, color='g', linestyle='--', label=f'Mean+2*Std: {mean_spread + 2*std_spread:.6f}')
    plt.axhline(y=mean_spread - 2*std_spread, color='g', linestyle='--', label=f'Mean-2*Std: {mean_spread - 2*std_spread:.6f}')
    plt.title('SOL-BUSD and SOL-USDT Spread')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sol_spread_analysis.png')
    plt.close()
    
    print(f"价差均值: {mean_spread:.6f}")
    print(f"价差标准差: {std_spread:.6f}")
    print(f"上轨(+2σ): {mean_spread + 2*std_spread:.6f}")
    print(f"下轨(-2σ): {mean_spread - 2*std_spread:.6f}")
    
    # 获取时间范围
    df_busd = pd.read_csv(sol_busd_file)
    df_usdt = pd.read_csv(sol_usdt_file)
    
    df_busd['candle_begin_time'] = pd.to_datetime(df_busd['candle_begin_time'])
    df_usdt['candle_begin_time'] = pd.to_datetime(df_usdt['candle_begin_time'])
    
    start_time = min(
        df_busd['candle_begin_time'].min(),
        df_usdt['candle_begin_time'].min()
    )
    
    end_time = max(
        df_busd['candle_begin_time'].max(),
        df_usdt['candle_begin_time'].max()
    )
    
    return merged, start_time, end_time


# 定制回测引擎，允许直接加载CSV数据
class CustomBacktestingEngine(BacktestingEngine):
    
    def load_data_from_csv(self, sol_busd_bars: List[BarData], sol_usdt_bars: List[BarData]):
        """
        直接从CSV加载的数据进行回测
        """
        # 清除之前的数据
        self.history_data.clear()
        
        # 合并和排序所有K线数据
        bars = sol_busd_bars + sol_usdt_bars
        bars.sort(key=lambda x: x.datetime)
        
        # 过滤日期范围内的数据
        for bar in bars:
            if self.start <= bar.datetime <= self.end:
                self.history_data.append(bar)
        
        # 初始化所有交易日的日期数据
        self.daily_results.clear()
        current_date = self.start.date()
        end_date = self.end.date()
        
        while current_date <= end_date:
            if current_date not in self.daily_results:
                # 使用0作为初始价格，后续会在update_daily_close中更新
                self.daily_results[current_date] = DailyResult(current_date, 0)
            current_date += timedelta(days=1)
        
        self.output(f"数据加载完成，总共 {len(self.history_data)} 条K线")


# 运行回测
def run_backtest(sol_busd_file, sol_usdt_file, start_time, end_time, boll_window=20, boll_dev=2.0, print_trades=False):
    """
    运行价差回归策略回测
    """
    # 加载K线数据
    sol_busd_bars = load_bar_data_from_csv(sol_busd_file, "SOL-BUSD")
    sol_usdt_bars = load_bar_data_from_csv(sol_usdt_file, "SOL-USDT")
    
    print(f"已加载 {len(sol_busd_bars)} 条 SOL-BUSD 数据")
    print(f"已加载 {len(sol_usdt_bars)} 条 SOL-USDT 数据")
    
    # 定义价差
    spread = SpreadData(
        name="SOL-Spread",
        legs=[
            LegData("SOL-BUSD.LOCAL"), 
            LegData("SOL-USDT.LOCAL")
        ],
        variable_symbols={"BUSD": "SOL-BUSD.LOCAL", "USDT": "SOL-USDT.LOCAL"},
        variable_directions={"BUSD": 1, "USDT": -1},  # 做多BUSD，做空USDT
        price_formula="BUSD-USDT",  # SOL-BUSD价格减去SOL-USDT价格
        trading_multipliers={"SOL-BUSD.LOCAL": 1, "SOL-USDT.LOCAL": 1},
        active_symbol="SOL-BUSD.LOCAL",
        min_volume=1,
        compile_formula=False
    )

    # 设置回测引擎
    engine = CustomBacktestingEngine()
    engine.set_parameters(
        spread=spread,
        interval="1m",
        start=start_time,
        end=end_time,
        rate=0.0000,  # 币安现货交易所手续费约为0.02%
        slippage=0,
        size=1,  # 每笔交易1个SOL
        pricetick=0.001,  # SOL交易最小价格变动
        capital=10000,  # 初始资金10000 USDT
    )
    
    # 添加策略，设置布林带参数
    engine.add_strategy(
        StatisticalArbitrageStrategy, 
        {
            "boll_window": boll_window,
            "boll_dev": boll_dev,
            "max_pos": 10
        }
    )

    # 执行回测
    engine.load_data_from_csv(sol_busd_bars, sol_usdt_bars)
    engine.run_backtesting()
    df = engine.calculate_result()
    stats = engine.calculate_statistics()
    
    # 打印回测结果
    print("====== 回测统计 ======")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 绘制回测结果
    engine.show_chart()
    
    # 显示交易记录
    if print_trades:
        print("\n====== 交易记录 ======")
        for trade in engine.trades.values():
            print(trade)
    else:
        print(f"\n总交易次数: {len(engine.trades)}")
        print("交易记录显示已禁用。设置 print_trades=True 可显示完整交易记录。")
    
    # 保存回测结果
    df.to_csv("sol_spread_backtest_results.csv", index=False)
    print("回测完成，结果已保存到 sol_spread_backtest_results.csv")
    
    return engine, df, stats


if __name__ == "__main__":
    # 文件路径
    sol_busd_file = "/Users/bobbyding/Documents/GitHub/apilot/SOL-BUSD_LOCAL_1m.csv"
    sol_usdt_file = "/Users/bobbyding/Documents/GitHub/apilot/SOL-USDT_LOCAL_1m.csv"
    
    # 分析价差并获取时间范围
    print("分析SOL-BUSD和SOL-USDT价差...")
    spread_data, start_time, end_time = analyze_spread(sol_busd_file, sol_usdt_file)
    print(f"回测时间范围: {start_time} 至 {end_time}")
    
    # 执行回测
    print("开始回测...")
    engine, df, stats = run_backtest(sol_busd_file, sol_usdt_file, start_time, end_time, print_trades=False)
