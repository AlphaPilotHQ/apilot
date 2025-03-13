"""
初始化环境，确保能够找到vnpy包
在Jupyter Notebook中，使用以下代码导入：
from init_env import *
"""

import os
import sys

# 添加项目根目录到Python路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 常用导入预加载 
from datetime import datetime
from vnpy.trader.optimize import OptimizationSetting
from vnpy.cta_strategy import BacktestingEngine
from vnpy.cta_strategy.strategies.atr_rsi_strategy import AtrRsiStrategy
