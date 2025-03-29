"""
策略模块

包含各种交易策略的模板和实现,支持CTA策略和目标仓位策略.

主要组件:
- CtaTemplate: CTA策略基类,提供标准的策略框架
- TargetPosTemplate: 目标仓位策略基类,简化仓位管理
- 各种示例策略实现,可作为自定义策略的参考

推荐用法:
    from apilot.strategy import CtaTemplate
    # 继承策略基类

    class MyStrategy(CtaTemplate):
        ...
"""

# 定义公共API
__all__ = [
    "CtaTemplate",
    "PairTradingStrategy",
    "PcpArbitrageStrategy",
    "PortfolioBollChannelStrategy",
    "TargetPosTemplate",
    "TrendFollowingStrategy",
]

# 从模板导入基类
from .template import CtaTemplate, TargetPosTemplate

# 导入示例策略
# from .examples.pair_trading_strategy import PairTradingStrategy
# from .examples.pcp_arbitrage_strategy import PcpArbitrageStrategy
# from .examples.portfolio_boll_channel_strategy import PortfolioBollChannelStrategy
# from .examples.trend_following_strategy import TrendFollowingStrategy
