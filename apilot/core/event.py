from apilot.event import EVENT_TIMER  # noqa

EVENT_TICK = "eTick."
EVENT_TRADE = "eTrade."
EVENT_ORDER = "eOrder."
EVENT_POSITION = "ePosition."
EVENT_ACCOUNT = "eAccount."
EVENT_QUOTE = "eQuote."
EVENT_CONTRACT = "eContract."
EVENT_LOG = "eLog"

# CTA策略事件
EVENT_CTA_LOG: str = "EVENT_CTA_LOG"
EVENT_CTA_STRATEGY = "EVENT_CTA_STRATEGY"

# 算法交易事件
EVENT_ALGO_LOG = "eAlgoLog"
EVENT_ALGO_UPDATE = "eAlgoUpdate"
