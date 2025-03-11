"""
Global settings for the APilot trading platform.
Customized from VeighNa to focus on cryptocurrency trading.
"""

from logging import INFO
from typing import Dict, Any
from tzlocal import get_localzone_name

from .utility import load_json


SETTINGS: Dict[str, Any] = {
    # 日志设置
    "log.active": True,
    "log.level": INFO,      # 改为INFO以获取更多交易信息
    "log.console": True,
    "log.file": True,
    
    # 通知设置
    "email.active": False,  # 默认关闭邮件通知
    "email.server": "smtp.gmail.com",
    "email.port": 587,
    "email.username": "",
    "email.password": "",  # 建议使用应用专用密码
    "email.sender": "",
    "email.receiver": "",
    
    # Telegram通知设置
    "telegram.active": False,
    "telegram.token": "",
    "telegram.chat_id": "",
    
    # 数据库设置
    "database.timezone": get_localzone_name(),
    "database.name": "csv",  # 默认使用CSV数据库，直接从CSV文件读取数据
    "database.data_path": "csv_database",  # CSV数据文件存储路径
    
    # 币安API设置
    "binance.key": "", 
    "binance.secret": "",
    "binance.proxy_host": "",  # 需要代理时设置
    "binance.proxy_port": 0,
    
    # 回测设置
    "backtest.risk_free": 0.0,
    "backtest.slippage": 0.0,
    "backtest.size": 1,
    "backtest.pricetick": 0.0,
    "backtest.capital": 1000000,
}


# 从JSON文件加载配置
SETTING_FILENAME: str = "apilot_setting.json"
SETTINGS.update(load_json(SETTING_FILENAME))


def get_settings(prefix: str = "") -> Dict[str, Any]:
    """获取指定前缀的所有设置项"""
    prefix_length: int = len(prefix)
    settings = {k[prefix_length:]: v for k, v in SETTINGS.items() if k.startswith(prefix)}
    return settings
