from init_env import *
from datetime import datetime

from vnpy.trader.constant import Direction, Interval
from vnpy.trader.object import BarData, TickData, OrderData, TradeData
from vnpy.cta_strategy.base import StopOrder
from vnpy.cta_strategy.template import CtaTemplate
from vnpy.cta_strategy.backtesting import BacktestingEngine
from vnpy.trader.utility import BarGenerator, ArrayManager


class TurtleSignalStrategy(CtaTemplate):
    """"""
    author = "用Python的交易员"

    entry_window = 20
    exit_window = 10
    atr_window = 20
    fixed_size = 1

    entry_up = 0
    entry_down = 0
    exit_up = 0
    exit_down = 0
    atr_value = 0

    long_entry = 0
    short_entry = 0
    long_stop = 0
    short_stop = 0

    parameters = ["entry_window", "exit_window", "atr_window", "fixed_size"]
    variables = ["entry_up", "entry_down", "exit_up", "exit_down", "atr_value"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        # 使用引擎的日志方法记录初始化信息
        self.cta_engine.write_log("策略初始化")
        
        # 对于回测，数据已通过CSV加载，无需调用load_bar
        # self.load_bar(20)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.cta_engine.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.cta_engine.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # Only calculates new entry channel when no position holding
        if not self.pos:
            self.entry_up, self.entry_down = self.am.donchian(
                self.entry_window
            )

        self.exit_up, self.exit_down = self.am.donchian(self.exit_window)

        if not self.pos:
            self.atr_value = self.am.atr(self.atr_window)

            self.long_entry = 0
            self.short_entry = 0
            self.long_stop = 0
            self.short_stop = 0

            self.send_buy_orders(self.entry_up)
            self.send_short_orders(self.entry_down)
        elif self.pos > 0:
            self.send_buy_orders(self.entry_up)

            sell_price = max(self.long_stop, self.exit_down)
            self.sell(sell_price, abs(self.pos), True)

        elif self.pos < 0:
            self.send_short_orders(self.entry_down)

            cover_price = min(self.short_stop, self.exit_up)
            self.cover(cover_price, abs(self.pos), True)
        
        # 移除对put_event的调用，该方法在回测环境下不可用
        # self.put_event()

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        if trade.direction == Direction.LONG:
            self.long_entry = trade.price
            self.long_stop = self.long_entry - 2 * self.atr_value
        else:
            self.short_entry = trade.price
            self.short_stop = self.short_entry + 2 * self.atr_value

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass

    def send_buy_orders(self, price):
        """"""
        t = self.pos / self.fixed_size

        if t < 1:
            self.buy(price, self.fixed_size, True)

        if t < 2:
            self.buy(price + self.atr_value * 0.5, self.fixed_size, True)

        if t < 3:
            self.buy(price + self.atr_value, self.fixed_size, True)

        if t < 4:
            self.buy(price + self.atr_value * 1.5, self.fixed_size, True)

    def send_short_orders(self, price):
        """"""
        t = self.pos / self.fixed_size

        if t > -1:
            self.short(price, self.fixed_size, True)

        if t > -2:
            self.short(price - self.atr_value * 0.5, self.fixed_size, True)

        if t > -3:
            self.short(price - self.atr_value, self.fixed_size, True)

        if t > -4:
            self.short(price - self.atr_value * 1.5, self.fixed_size, True)


def run_backtesting(show_chart=True):
    """
    运行海龟信号策略回测
    """
    # 初始化回测引擎
    engine = BacktestingEngine()
    
    # 设置回测参数
    engine.set_parameters(
        vt_symbol="SOL-USDT.LOCAL",  # 修改为与CSV文件名匹配的交易对
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        rate=0.0001,
        slippage=0,
        size=1,
        pricetick=0.01,
        capital=100000,
    )
    
    # 添加策略
    engine.add_strategy(
        TurtleSignalStrategy, 
        {
            "entry_window": 20,
            "exit_window": 10,
            "atr_window": 20,
            "fixed_size": 1
        }
    )
    
    # 添加数据 - 确保文件路径正确
    engine.add_data(
        database_type="csv",
        data_path="/Users/bobbyding/Documents/GitHub/apilot/SOL-USDT_LOCAL_1m.csv",
        datetime="candle_begin_time",  # 修改为与CSV文件列名匹配
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume"
    )
    
    # 运行回测
    engine.run_backtesting()
    
    # 计算结果和统计指标
    df = engine.calculate_result()
    stats = engine.calculate_statistics()
    
    # 打印统计结果
    print(f"起始资金: {100000:.2f}")
    print(f"结束资金: {stats.get('end_balance', 0):.2f}")
    print(f"总收益率: {stats.get('total_return', 0)*100:.2f}%")  
    print(f"年化收益: {stats.get('annual_return', 0)*100:.2f}%")  
    
    # 添加错误处理，避免某些指标不存在
    max_drawdown = stats.get('max_drawdown', 0)
    if isinstance(max_drawdown, (int, float)):
        print(f"最大回撤: {max_drawdown*100:.2f}%")  
    else:
        print(f"最大回撤: 0.00%")
        
    print(f"夏普比率: {stats.get('sharpe_ratio', 0):.2f}")
    
    # 显示图表
    if show_chart:
        engine.show_chart()
    
    return df, stats, engine


if __name__ == "__main__":
    # 运行回测
    df, stats, engine = run_backtesting(show_chart=True)
