"""
策略模块

包含各种交易策略的模板和实现,支持PA策略和目标持仓策略.

主要组件:
- PATemplate: PA策略基类,提供标准的策略框架
- TargetPosTemplate: 目标持仓策略模板,用于管理目标持仓

使用示例:
    from apilot.strategy import PATemplate

    class MyStrategy(PATemplate):
        # 定义您的策略
        pass

默认导出:
__all__ = [
    "PATemplate",
    "TargetPosTemplate",
]

推荐用法:
    from apilot.strategy import PATemplate
    # 继承策略基类

    class MyStrategy(PATemplate):
        ...

"""

# 从模板导入基类
from .template import PATemplate, TargetPosTemplate

# 导入示例策略
# from .examples.pair_trading_strategy import PairTradingStrategy
# from .examples.pcp_arbitrage_strategy import PcpArbitrageStrategy
# from .examples.portfolio_boll_channel_strategy import PortfolioBollChannelStrategy
# from .examples.trend_following_strategy import TrendFollowingStrategy

# 定义公共API
__all__ = [
    "PATemplate",
    "TargetPosTemplate",
]
