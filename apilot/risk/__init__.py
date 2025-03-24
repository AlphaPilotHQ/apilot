from pathlib import Path
from typing import Type

from apilot.core.app import BaseApp

from .engine import RiskEngine, APP_NAME


class RiskManagerApp(BaseApp):
    """"""
    app_name: str = APP_NAME
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "交易风控"
    engine_class: Type[RiskEngine] = RiskEngine
    widget_name: str = "RiskManager"
    icon_name: str = str(app_path.joinpath("ui", "rm.ico"))
