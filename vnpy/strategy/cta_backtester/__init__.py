"""
CTA Strategy Backtester Module.
"""

from pathlib import Path

from .engine import BacktesterEngine, APP_NAME


class CtaBacktesterApp:
    """"""
    app_name: str = APP_NAME
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "CTA回测"
    engine_class = BacktesterEngine
