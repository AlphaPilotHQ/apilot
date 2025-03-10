"""
MySQL数据库适配器测试 (pytest版本)
"""
import pytest
import os
from datetime import datetime, timedelta
from pathlib import Path

# 加载.env文件中的环境变量
from dotenv import load_dotenv

# 查找项目根目录下的.env文件
root_path = Path(__file__).parent.parent.parent  # tests/database -> root
env_file = root_path / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"已加载环境变量配置: {env_file}")
else:
    print(f"警告: 未找到.env文件，请从.env.template创建。使用默认设置继续...")

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.setting import SETTINGS

# 设置默认数据库参数，防止模块导入时出错
SETTINGS["database.database"] = os.environ.get("VNPY_TEST_MYSQL_DB", "vnpy_test")
SETTINGS["database.host"] = os.environ.get("VNPY_TEST_MYSQL_HOST", "localhost")
SETTINGS["database.port"] = int(os.environ.get("VNPY_TEST_MYSQL_PORT", "3306"))
SETTINGS["database.user"] = os.environ.get("VNPY_TEST_MYSQL_USER", "root")
SETTINGS["database.password"] = os.environ.get("VNPY_TEST_MYSQL_PASSWORD", "")

# 先检查pymysql是否安装
try:
    import pymysql
    has_pymysql = True
except ImportError:
    print("警告: pymysql 模块未安装，如果需要运行 MySQL 测试，请安装: pip install pymysql")
    has_pymysql = False
    MysqlDatabase = None

# 在pymysql可用的情况下导入MySQL数据库模块
if has_pymysql:
    try:
        from vnpy.database.mysql.mysql_database import MysqlDatabase
    except Exception as e:
        print(f"导入MySQL模块时出错: {e}")
        MysqlDatabase = None


# pytest fixture - 每个测试函数前都会执行
@pytest.fixture
def mysql_db():
    """初始化数据库连接"""
    # 检查pymysql和MySQL模块是否可用
    if not has_pymysql:
        pytest.skip("pymysql 驱动程序未安装，跳过MySQL测试，请安装: pip install pymysql")
    if MysqlDatabase is None:
        pytest.skip("无法加载MySQL模块，跳过测试")
        
    # 检查环境变量是否设置
    if not SETTINGS["database.user"] or not SETTINGS["database.database"]:
        pytest.skip("缺少MySQL配置参数，跳过测试")
    
    # 控制台输出连接信息
    print(f"\n连接到 MySQL: {SETTINGS['database.host']}:{SETTINGS['database.port']}/{SETTINGS['database.database']}")
    
    db = MysqlDatabase()
    
    # 清空测试数据
    _clear_test_data(db)
    
    # yield将db对象传递给测试函数
    yield db
    
    # 测试结束后清理
    _clear_test_data(db)


def _clear_test_data(db):
    """清空测试使用的数据"""
    # 删除测试用的K线数据
    db.delete_bar_data(
        symbol="test_symbol",
        exchange=Exchange.BINANCE,
        interval=Interval.MINUTE
    )
    
    # 删除测试用的TICK数据
    db.delete_tick_data(
        symbol="test_symbol",
        exchange=Exchange.BINANCE
    )


def _create_test_bars(n=10):
    """创建测试用的K线数据"""
    bars = []
    start = datetime.now()
    
    for i in range(n):
        dt = start + timedelta(minutes=i)
        bar = BarData(
            symbol="test_symbol",
            exchange=Exchange.BINANCE,
            datetime=dt,
            interval=Interval.MINUTE,
            volume=100.0 + i,
            open_price=10.0 + i,
            high_price=11.0 + i,
            low_price=9.0 + i,
            close_price=10.5 + i,
            gateway_name="test"
        )
        bars.append(bar)
    
    return bars, start


def _create_test_ticks(n=10):
    """创建测试用的TICK数据"""
    ticks = []
    start = datetime.now()
    
    for i in range(n):
        dt = start + timedelta(seconds=i)
        tick = TickData(
            symbol="test_symbol",
            exchange=Exchange.BINANCE,
            datetime=dt,
            name="Test Instrument",
            volume=100.0 + i,
            last_price=10.0 + i * 0.1,
            bid_price_1=9.9 + i * 0.1,
            ask_price_1=10.1 + i * 0.1,
            bid_volume_1=50.0,
            ask_volume_1=50.0,
            gateway_name="test"
        )
        ticks.append(tick)
    
    return ticks, start


def test_connection(mysql_db):
    """测试数据库连接"""
    assert mysql_db.db is not None


def test_save_and_load_bar_data(mysql_db):
    """测试保存和加载K线数据"""
    # 创建测试数据
    bars, start = _create_test_bars(10)
    
    # 保存测试数据
    mysql_db.save_bar_data(bars)
    
    # 加载测试数据并验证
    loaded_bars = mysql_db.load_bar_data(
        symbol="test_symbol",
        exchange=Exchange.BINANCE,
        interval=Interval.MINUTE,
        start=start,
        end=start + timedelta(minutes=15)
    )
    
    assert len(loaded_bars) == len(bars)
    
    # 验证数据内容
    for original, loaded in zip(bars, loaded_bars):
        assert original.symbol == loaded.symbol
        assert original.exchange == loaded.exchange
        assert original.interval == loaded.interval
        assert original.volume == loaded.volume
        assert original.open_price == loaded.open_price
        assert original.close_price == loaded.close_price


def test_save_and_load_tick_data(mysql_db):
    """测试保存和加载TICK数据"""
    # 创建测试数据
    ticks, start = _create_test_ticks(10)
    
    # 保存测试数据
    mysql_db.save_tick_data(ticks)
    
    # 加载测试数据并验证
    loaded_ticks = mysql_db.load_tick_data(
        symbol="test_symbol",
        exchange=Exchange.BINANCE,
        start=start,
        end=start + timedelta(seconds=15)
    )
    
    assert len(loaded_ticks) == len(ticks)
    
    # 验证数据内容
    for original, loaded in zip(ticks, loaded_ticks):
        assert original.symbol == loaded.symbol
        assert original.exchange == loaded.exchange
        assert original.datetime.date() == loaded.datetime.date()
        assert original.last_price == loaded.last_price
        assert original.volume == loaded.volume


def test_get_bar_overview(mysql_db):
    """测试获取K线数据概览"""
    # 创建并保存测试数据
    bars, _ = _create_test_bars(10)
    mysql_db.save_bar_data(bars)
    
    # 获取概览
    overviews = mysql_db.get_bar_overview()
    
    # 至少应该有我们刚添加的数据
    assert len(overviews) > 0
    
    # 查找我们添加的测试数据概览
    test_overview = None
    for overview in overviews:
        if (overview.symbol == "test_symbol" and 
            overview.exchange == Exchange.BINANCE and
            overview.interval == Interval.MINUTE):
            test_overview = overview
            break
    
    assert test_overview is not None
    assert test_overview.count >= 10  # 至少有10条记录


def test_get_tick_overview(mysql_db):
    """测试获取TICK数据概览"""
    # 创建并保存测试数据
    ticks, _ = _create_test_ticks(10)
    mysql_db.save_tick_data(ticks)
    
    # 获取概览
    overviews = mysql_db.get_tick_overview()
    
    # 至少应该有我们刚添加的数据
    assert len(overviews) > 0
    
    # 查找我们添加的测试数据概览
    test_overview = None
    for overview in overviews:
        if (overview.symbol == "test_symbol" and 
            overview.exchange == Exchange.BINANCE):
            test_overview = overview
            break
    
    assert test_overview is not None
    assert test_overview.count >= 10  # 至少有10条记录


@pytest.mark.skip(reason="此测试可能会影响数据库中的其他数据，谨慎运行")
def test_delete_bar_data(mysql_db):
    """测试删除K线数据"""
    # 创建并保存测试数据
    bars, _ = _create_test_bars(10)
    mysql_db.save_bar_data(bars)
    
    # 删除数据
    mysql_db.delete_bar_data(
        symbol="test_symbol",
        exchange=Exchange.BINANCE,
        interval=Interval.MINUTE
    )
    
    # 验证数据已删除
    loaded_bars = mysql_db.load_bar_data(
        symbol="test_symbol",
        exchange=Exchange.BINANCE,
        interval=Interval.MINUTE,
        start=datetime(2000, 1, 1),
        end=datetime(2100, 1, 1)
    )
    
    assert len(loaded_bars) == 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
