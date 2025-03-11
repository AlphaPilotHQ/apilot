from datetime import datetime
import os
import pandas as pd
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.utility import BarGenerator, ArrayManager
from vnpy.vnpy_ctastrategy import CtaTemplate, StopOrder
from vnpy.trader.object import TickData, BarData, ContractData

# 导入回测引擎
from vnpy.strategy.cta_backtester.engine import BacktesterEngine


class SimpleMaStrategy(CtaTemplate):
    """
    简单双均线交叉策略
    """
    author = "VeighNa"

    # 策略参数
    fast_window = 10  # 快速均线窗口
    slow_window = 20  # 慢速均线窗口

    # 策略变量
    fast_ma0 = 0.0  # 当前快速均线
    fast_ma1 = 0.0  # 上一周期快速均线
    slow_ma0 = 0.0  # 当前慢速均线
    slow_ma1 = 0.0  # 上一周期慢速均线

    # 参数列表，保存了参数的名称
    parameters = ["fast_window", "slow_window"]

    # 变量列表，保存了变量的名称
    variables = ["fast_ma0", "fast_ma1", "slow_ma0", "slow_ma1"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """
        初始化策略
        """
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # K线合成器：从Tick合成分钟K线用于交易
        self.bg = BarGenerator(self.on_bar)
        # 时间序列管理器：计算技术指标用
        self.am = ArrayManager()

    def on_init(self):
        """
        策略初始化
        """
        self.write_log("策略初始化")
        # 加载历史数据用于初始化回放
        self.load_bar(10)

    def on_start(self):
        """
        策略启动
        """
        self.write_log("策略启动")
        # 策略启动时发出的下单
        self.put_event()

    def on_stop(self):
        """
        策略停止
        """
        self.write_log("策略停止")
        # 策略停止时发出的下单
        self.put_event()

    def on_tick(self, tick: TickData):
        """
        Tick数据更新
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        K线数据更新
        """
        am = self.am
        am.update_bar(bar)

        # 更新均线计算，至少需要满足计算周期才能计算
        if not am.inited:
            self.put_event()
            return

        # 保存当前均线值
        self.fast_ma1 = self.fast_ma0
        self.slow_ma1 = self.slow_ma0

        # 计算当前均线值
        self.fast_ma0 = am.sma(self.fast_window)
        self.slow_ma0 = am.sma(self.slow_window)

        # 判断均线交叉
        cross_over = (self.fast_ma0 > self.slow_ma0 and
                      self.fast_ma1 < self.slow_ma1)  # 金叉
        cross_below = (self.fast_ma0 < self.slow_ma0 and
                       self.fast_ma1 > self.slow_ma1)  # 死叉

        # 交易信号
        if cross_over:
            # 如果有空头持仓，先平空
            if self.pos < 0:
                self.cover(bar.close_price, abs(self.pos))
            # 做多
            self.buy(bar.close_price, 1)
        elif cross_below:
            # 如果有多头持仓，先平多
            if self.pos > 0:
                self.sell(bar.close_price, abs(self.pos))
            # 做空
            self.short(bar.close_price, 1)

        # 更新图形界面
        self.put_event()

    def on_order(self, order):
        """
        报单更新
        """
        pass

    def on_trade(self, trade):
        """
        成交更新
        """
        self.put_event()

    def on_stop_order(self, stop_order):
        """
        停止单更新
        """
        pass


def run_backtest():
    """
    运行简单回测脚本
    """
    # 创建并初始化引擎
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    backtester_engine = BacktesterEngine(main_engine, event_engine)
    backtester_engine.init_engine()
    
    # 修改日志输出方式
    original_write_log = backtester_engine.write_log
    backtester_engine.write_log = lambda msg: print(f"[回测] {msg}") or original_write_log(msg)
    
    # 手动注册策略类
    backtester_engine.classes["SimpleMaStrategy"] = SimpleMaStrategy
    print(f"[回测] 已手动注册策略类: SimpleMaStrategy")
    
    # 指定本地CSV文件路径
    csv_file = "/Users/bobbyding/Documents/GitHub/vnpy/SOL-USDT.csv"
    
    # 设置交易对参数
    symbol = "SOL-USDT"
    exchange = Exchange.BINANCE
    interval = Interval.MINUTE
    
    # 加载CSV数据
    bar_data = backtester_engine.load_bar_data_from_csv(
        csv_path=csv_file,
        symbol=symbol,
        exchange=exchange,
        interval=interval.value
    )
    
    if not bar_data:
        print("加载数据失败，无法继续回测")
        return
    
    print(f"成功加载数据，共{len(bar_data)}条记录")
    
    # 获取时间范围
    start_date = bar_data[0].datetime
    end_date = bar_data[-1].datetime
    print(f"数据时间范围: {start_date} 到 {end_date}")
    
    # 设置完整的vt_symbol
    vt_symbol = f"{symbol}.{exchange.value}"
    
    # 注入数据到回测引擎
    print("正在将数据注入到回测引擎...")
    backtester_engine.backtesting_engine.history_data = bar_data
    
    # 运行回测
    try:
        backtester_engine.run_backtesting(
            class_name="SimpleMaStrategy",
            vt_symbol=vt_symbol,
            interval=interval.value,
            start=start_date,
            end=end_date,
            rate=0.0003,         # 手续费率
            slippage=0,          # 滑点
            size=1,              # 合约乘数
            pricetick=0.01,      # 价格最小变动
            capital=100000,      # 起始资金
            setting={            # 策略参数
                "fast_window": 10,
                "slow_window": 20
            }
        )
        
        # 获取结果
        stats = backtester_engine.get_result_statistics()
        if stats:
            print("\n===== 回测结果统计 =====")
            for key, value in stats.items():
                print(f"{key}: {value}")
        else:
            print("\n回测未能生成结果统计")
            
    except Exception as e:
        print(f"\n回测过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    run_backtest()
