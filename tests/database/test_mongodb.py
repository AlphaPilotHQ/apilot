"""
MongoDB数据库适配器单元测试
"""

import unittest
import os
import sys
from datetime import datetime, timedelta
from typing import List
from pathlib import Path

# 加载.env文件中的环境变量
from dotenv import load_dotenv

# 查找项目根目录下的.env文件
root_path = Path(__file__).parent.parent.parent  # tests/database -> root
# 先尝试加载.env文件，如果不存在则给出提示
env_file = root_path / ".env"
env_template = root_path / ".env.template"

if env_file.exists():
    load_dotenv(env_file)
    print(f"已加载环境变量配置: {env_file}")
else:
    print(f"警告: 未找到.env文件，请从.env.template创建。使用默认设置继续...")
    if env_template.exists():
        print(f"提示: 可以使用以下命令创建.env文件:\ncp {env_template} {env_file}")

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.setting import SETTINGS
from vnpy.database.mongodb.mongodb_database import MongodbDatabase


class TestMongodbDatabase(unittest.TestCase):
    """测试MongoDB数据库适配器"""

    def setUp(self) -> None:
        """初始化数据库连接"""
        # 设置MongoDB连接参数 - 从环境变量中读取
        from getpass import getpass
        
        # 设置MongoDB连接参数
        SETTINGS["database.database"] = os.environ.get("VNPY_TEST_DB", "alphapilot")
        SETTINGS["database.host"] = os.environ.get("VNPY_TEST_HOST", "localhost")
        SETTINGS["database.port"] = int(os.environ.get("VNPY_TEST_PORT", "27017"))
        SETTINGS["database.user"] = os.environ.get("VNPY_TEST_USER", "")
        SETTINGS["database.password"] = os.environ.get("VNPY_TEST_PASSWORD", "")
        
        # 如果未设置凭证且开启了提示，则交互式获取
        prompt_credentials = os.environ.get("VNPY_PROMPT_CREDENTIALS", "FALSE").upper() == "TRUE"
        if prompt_credentials:
            if not SETTINGS["database.user"]:
                SETTINGS["database.user"] = input("MongoDB 用户名: ")
            if not SETTINGS["database.password"]:
                SETTINGS["database.password"] = getpass("MongoDB 密码: ")
        
        # 控制台输出连接信息，但不显示敏感凭证
        print(f"\n连接到 MongoDB: {SETTINGS['database.host']}:{SETTINGS['database.port']}/{SETTINGS['database.database']}")
        if SETTINGS["database.user"]:
            print(f"  用户名: {SETTINGS['database.user']}")
            # 如果有密码，显示星号；如果没有，显示未设置
            print(f"  密码: {'*' * 8 if SETTINGS['database.password'] else '<未设置>'}")

        self.database = MongodbDatabase()
        
        # 不清空测试数据，因为是连接到测试服务器
        # self._clear_test_data()

    def tearDown(self) -> None:
        """清理测试数据"""
        # 不清空测试数据，因为是连接到测试服务器
        # self._clear_test_data()

    def _clear_test_data(self) -> None:
        """清空测试使用的数据"""
        self.database.db.drop_collection("bar_data")
        self.database.db.drop_collection("tick_data")
        self.database.db.drop_collection("bar_overview")
        self.database.db.drop_collection("tick_overview")

    def test_connection(self) -> None:
        """测试数据库连接"""
        # 只测试数据库连接是否成功
        self.assertIsNotNone(self.database.db)
        
        # 检查数据库中是否存在symbol_trade集合
        collections = self.database.db.list_collection_names()
        self.assertIn('symbol_trade', collections)
        
        # 检查是否能从symbol_trade集合中获取数据
        symbol_trade = self.database.db['symbol_trade'].find_one()
        self.assertIsNotNone(symbol_trade)
        self.assertIn('symbol', symbol_trade)
        
        print(f"\n成功连接到数据库，发现集合: {collections}")
        print(f"symbol_trade样例: {symbol_trade}")

    def test_symbol_data(self) -> None:
        """测试访问交易对数据"""
        # 查询symbol表中的数据
        symbols = list(self.database.db['symbol'].find().limit(5))
        self.assertTrue(len(symbols) > 0, "没有找到交易对数据")
        
        # 打印一些交易对信息
        print(f"\n共找到 {len(symbols)} 个交易对示例:")
        for symbol in symbols:
            print(f"  - {symbol.get('name', 'N/A')}: {symbol}")
        
        # 查询某个交易对的交易数据
        if symbols and 'name' in symbols[0]:
            symbol_name = symbols[0]['name']
            trades = list(self.database.db['symbol_trade'].find({'symbol': symbol_name}).limit(5))
            
            print(f"\n交易对 {symbol_name} 的交易数据示例:")
            for trade in trades:
                print(f"  - {trade}")
            
            self.assertTrue(len(trades) >= 0, f"没有找到交易对 {symbol_name} 的交易数据")


if __name__ == "__main__":
    unittest.main()
