"""
通用工具函数模块

提供常用的工具函数和辅助方法
"""

import os
from pathlib import Path


def get_data_dir() -> str:
    """
    获取数据目录的绝对路径

    优先使用环境变量 APILOT_DATA_DIR 定义的路径,
    否则使用项目根目录下的 data 文件夹
    """
    # 优先使用环境变量
    env_data_dir = os.environ.get("APILOT_DATA_DIR")
    if env_data_dir and os.path.exists(env_data_dir):
        return env_data_dir

    # 获取项目根目录 (apilot 项目根目录)
    root_path = Path(__file__).parent.parent.parent.absolute()
    data_dir = os.path.join(root_path, "data")

    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    return data_dir
