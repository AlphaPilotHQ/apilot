"""
策略模块

包含各种交易策略的模板和实现，支持CTA策略和目标仓位策略。
"""

# 从模板导入基类
from .template import CtaTemplate, TargetPosTemplate


from .examples.pair_trading_strategy import PairTradingStrategy
from .examples.pcp_arbitrage_strategy import PcpArbitrageStrategy
from .examples.portfolio_boll_channel_strategy import PortfolioBollChannelStrategy
from .examples.trend_following_strategy import TrendFollowingStrategy
