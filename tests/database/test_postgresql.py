"""
PostgreSQL数据库适配器测试 (pytest版本)
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
from vnpy.database.postgresql.postgresql_database import PostgresqlDatabase


# pytest fixture - 每个测试函数前都会执行
@pytest.fixture
def postgresql_db():
    """初始化数据库连接"""
    # 设置PostgreSQL连接参数 - 从环境变量中读取
    SETTINGS["database.database"] = os.environ.get("VNPY_TEST_POSTGRES_DB", "vnpy_test")
    SETTINGS["database.host"] = os.environ.get("VNPY_TEST_POSTGRES_HOST", "localhost")
    SETTINGS["database.port"] = int(os.environ.get("VNPY_TEST_POSTGRES_PORT", "5432"))
    SETTINGS["database.user"] = os.environ.get("VNPY_TEST_POSTGRES_USER", "postgres")
    SETTINGS["database.password"] = os.environ.get("VNPY_TEST_POSTGRES_PASSWORD", "")
    
    # 控制台输出连接信息
    print(f"\n连接到 PostgreSQL: {SETTINGS['database.host']}:{SETTINGS['database.port']}/{SETTINGS['database.database']}")
    
    db = PostgresqlDatabase()
    
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


def test_connection(postgresql_db):
    """测试数据库连接"""
    assert postgresql_db.db is not None


def test_save_and_load_bar_data(postgresql_db):
    """测试保存和加载K线数据"""
    # 创建测试数据
    bars, start = _create_test_bars(10)
    
    # 保存测试数据
    postgresql_db.save_bar_data(bars)
    
    # 加载测试数据并验证
    loaded_bars = postgresql_db.load_bar_data(
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


def test_save_and_load_tick_data(postgresql_db):
    """测试保存和加载TICK数据"""
    # 创建测试数据
    ticks, start = _create_test_ticks(10)
    
    # 保存测试数据
    postgresql_db.save_tick_data(ticks)
    
    # 加载测试数据并验证
    loaded_ticks = postgresql_db.load_tick_data(
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


def test_get_bar_overview(postgresql_db):
    """测试获取K线数据概览"""
    # 创建并保存测试数据
    bars, _ = _create_test_bars(10)
    postgresql_db.save_bar_data(bars)
    
    # 获取概览
    overviews = postgresql_db.get_bar_overview()
    
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


def test_get_tick_overview(postgresql_db):
    """测试获取TICK数据概览"""
    # 创建并保存测试数据
    ticks, _ = _create_test_ticks(10)
    postgresql_db.save_tick_data(ticks)
    
    # 获取概览
    overviews = postgresql_db.get_tick_overview()
    
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


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
