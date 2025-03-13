"""
加密货币投资组合回测示例 - 使用BTC和SOL
特别适合在crypto市场的量化交易分析
"""
from init_env import *
from datetime import datetime

# 导入所需模块和图形库
from vnpy.trader.constant import Interval, Exchange

# 导入portfolio_strategy模块 - 使用已移植的模块
from vnpy.portfolio_strategy.backtesting import BacktestingEngine 
from vnpy.portfolio_strategy.strategies.trend_following_strategy import TrendFollowingStrategy
from vnpy.trader.optimize import OptimizationSetting


def run_crypto_portfolio_backtest():
    """
    运行加密货币投资组合回测
    """
    print("开始运行加密货币投资组合回测...")
    
    # 创建回测引擎
    engine = BacktestingEngine()
    
    # 设置回测参数
    engine.set_parameters(
        vt_symbols=["BTC-USDT.LOCAL", "SOL-USDT.LOCAL"],  # BTC和SOL使用LOCAL交易所
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 30),
        rates={                    # 手续费率
            "BTC-USDT.LOCAL": 0.0005,
            "SOL-USDT.LOCAL": 0.0005
        },
        slippages={                # 滑点
            "BTC-USDT.LOCAL": 0.0001,
            "SOL-USDT.LOCAL": 0.0001
        },
        sizes={                    # 合约乘数
            "BTC-USDT.LOCAL": 1,
            "SOL-USDT.LOCAL": 1
        },
        priceticks={               # 价格最小变动
            "BTC-USDT.LOCAL": 0.01,
            "SOL-USDT.LOCAL": 0.001
        },
        capital=100000,            # 初始资金
    )
    
    # 添加趋势跟踪策略
    setting = {
        "window": 20,              # 趋势窗口
        "open_std": 2.0,           # 开仓标准差
        "close_std": 1.0,          # 平仓标准差
        "fixed_size": 1            # 固定头寸
    }
    
    engine.add_strategy(TrendFollowingStrategy, setting)
    
    # 加载历史数据
    engine.load_data()
    
    # 运行回测
    engine.run_backtesting()
    
    # 计算并输出回测结果
    df = engine.calculate_result()
    engine.calculate_statistics()
    
    # 显示回测图表
    engine.show_chart()
    
    # 参数优化示例 (注释掉，需要时可以解开使用)
    """
    # 设置优化参数
    setting = OptimizationSetting()
    setting.set_target("sharpe_ratio")   # 优化目标 - 夏普比率
    setting.add_parameter("window", 10, 30, 5)  # 参数范围
    setting.add_parameter("open_std", 1.0, 3.0, 0.5)
    setting.add_parameter("close_std", 0.5, 2.0, 0.5)
    
    # 执行参数优化
    result = engine.run_optimization(setting)
    
    # 输出优化结果
    for strategy_setting, target_value in result:
        print(f"参数: {strategy_setting}, 目标值: {target_value}")
    """


if __name__ == "__main__":
    run_crypto_portfolio_backtest()
