from pathlib import Path

# 使用 Python 3.8+ 标准库
from importlib.metadata import version, PackageNotFoundError

from vnpy.trader.app import BaseApp

from .engine import AlgoEngine, APP_NAME


try:
    __version__ = version("vnpy_algotrading")
except PackageNotFoundError:
    __version__ = "dev"


class AlgoTradingApp(BaseApp):
    """"""

    app_name: str = APP_NAME
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "算法交易"
    engine_class: AlgoEngine = AlgoEngine
    widget_name: str = "AlgoManager"
    icon_name: str = str(app_path.joinpath("ui", "algo.ico"))
