from vnpy.trader.utility import BarGenerator, ArrayManager
from vnpy.vnpy_ctastrategy import CtaTemplate, StopOrder
from vnpy.trader.object import TickData, BarData
from vnpy.trader.constant import Interval


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
