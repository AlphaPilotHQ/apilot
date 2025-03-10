"""
工具函数(Utility)单元测试
"""
import pytest
import os
from pathlib import Path
from datetime import datetime

from vnpy.trader.utility import (
    extract_vt_symbol,
    generate_vt_symbol,
    load_json,
    save_json,
    round_to,
    get_folder_path,
    get_file_path,
    get_temp_path,
    get_csv_path
)

# 测试函数将在此处实现

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
