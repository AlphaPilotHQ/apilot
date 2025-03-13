
from init_env import *

def main():
    """
    主函数，运行回测
    """
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="SOL-USDT.LOCAL",  # 使用加密货币符号和LOCAL交易所
        interval="1m",               # 使用小时线，加密货币常用
        start=datetime(2023, 1, 1),  # 更新为较近的日期
        end=datetime(2023, 6, 30),
        rate=0.001,                  # 加密货币交易费率
        slippage=0.5,                # 滑点
        size=1,                      # 每次交易1个BTC
        pricetick=0.1,               # 最小价格变动
        capital=100_000,             # 初始资金
    )
    engine.add_strategy(AtrRsiStrategy, {})
    
    engine.load_data()
    engine.run_backtesting()
    df = engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()
    

    
if __name__ == "__main__":
    main()
