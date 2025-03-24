"""
量化交易策略模块
"""

# 导出策略模板
from .template import CtaTemplate, TargetPosTemplate

# 导出示例策略
from .examples.pair_trading_strategy import PairTradingStrategy
from .examples.pcp_arbitrage_strategy import PcpArbitrageStrategy
from .examples.portfolio_boll_channel_strategy import PortfolioBollChannelStrategy
from .examples.trend_following_strategy import TrendFollowingStrategy