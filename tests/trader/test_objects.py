"""
交易对象和工具函数测试 (pytest版本)
"""
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from vnpy.trader.constant import Direction, Offset, Status, Exchange, Interval, Product
from vnpy.trader.object import (
    TickData, 
    BarData, 
    OrderData, 
    TradeData, 
    PositionData, 
    AccountData,
    ContractData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest
)
from vnpy.trader.utility import (
    extract_vt_symbol, 
    generate_vt_symbol, 
    load_json, 
    save_json,
    round_to
)


def test_tick_data_creation():
    """测试创建TICK数据对象"""
    tick = TickData(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        datetime=datetime.now(),
        name="BTC/USDT",
        volume=1000.0,
        last_price=50000.0,
        bid_price_1=49990.0,
        ask_price_1=50010.0,
        bid_volume_1=10.0,
        ask_volume_1=10.0,
        gateway_name="BINANCE"
    )
    
    assert tick.symbol == "BTCUSDT"
    assert tick.exchange == Exchange.BINANCE
    assert tick.vt_symbol == f"BTCUSDT.{Exchange.BINANCE.value}"
    assert isinstance(tick.datetime, datetime)
    assert tick.last_price == 50000.0


def test_bar_data_creation():
    """测试创建K线数据对象"""
    bar = BarData(
        symbol="ETHUSDT",
        exchange=Exchange.BINANCE,
        datetime=datetime.now(),
        interval=Interval.MINUTE,
        volume=500.0,
        open_price=3000.0,
        high_price=3100.0,
        low_price=2900.0,
        close_price=3050.0,
        gateway_name="BINANCE"
    )
    
    assert bar.symbol == "ETHUSDT"
    assert bar.exchange == Exchange.BINANCE
    assert bar.vt_symbol == f"ETHUSDT.{Exchange.BINANCE.value}"
    assert bar.interval == Interval.MINUTE
    assert bar.open_price == 3000.0
    assert bar.close_price == 3050.0


def test_order_data_creation():
    """测试创建委托订单对象"""
    order = OrderData(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        orderid="123456",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=50000.0,
        volume=1.0,
        status=Status.SUBMITTING,
        gateway_name="BINANCE"
    )
    
    assert order.symbol == "BTCUSDT"
    assert order.exchange == Exchange.BINANCE
    assert order.vt_symbol == f"BTCUSDT.{Exchange.BINANCE.value}"
    assert order.orderid == "123456"
    assert order.vt_orderid == f"BINANCE.123456"
    assert order.direction == Direction.LONG
    assert order.offset == Offset.OPEN
    assert order.price == 50000.0
    assert order.volume == 1.0
    assert order.status == Status.SUBMITTING


def test_trade_data_creation():
    """测试创建成交数据对象"""
    trade = TradeData(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        orderid="123456",
        tradeid="T123456",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=50000.0,
        volume=1.0,
        datetime=datetime.now(),
        gateway_name="BINANCE"
    )
    
    assert trade.symbol == "BTCUSDT"
    assert trade.exchange == Exchange.BINANCE
    assert trade.vt_symbol == f"BTCUSDT.{Exchange.BINANCE.value}"
    assert trade.orderid == "123456"
    assert trade.tradeid == "T123456"
    assert trade.vt_tradeid == f"BINANCE.T123456"
    assert trade.direction == Direction.LONG
    assert trade.offset == Offset.OPEN
    assert trade.price == 50000.0
    assert trade.volume == 1.0


def test_contract_data_creation():
    """测试创建合约对象"""
    contract = ContractData(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        name="BTC/USDT永续合约",
        product=Product.FUTURES,
        size=1.0,
        pricetick=0.01,
        min_volume=0.001,
        history_data=True,
        gateway_name="BINANCE"
    )
    
    assert contract.symbol == "BTCUSDT"
    assert contract.exchange == Exchange.BINANCE
    assert contract.vt_symbol == f"BTCUSDT.{Exchange.BINANCE.value}"
    assert contract.name == "BTC/USDT永续合约"
    assert contract.product == Product.FUTURES
    assert contract.size == 1.0
    assert contract.pricetick == 0.01
    assert contract.min_volume == 0.001
    assert contract.history_data is True


def test_order_request_creation():
    """测试创建委托请求对象"""
    req = OrderRequest(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=50000.0,
        volume=1.0,
        reference="demo"
    )
    
    assert req.symbol == "BTCUSDT"
    assert req.exchange == Exchange.BINANCE
    assert req.vt_symbol == f"BTCUSDT.{Exchange.BINANCE.value}"
    assert req.direction == Direction.LONG
    assert req.offset == Offset.OPEN
    assert req.price == 50000.0
    assert req.volume == 1.0
    assert req.reference == "demo"


def test_cancel_request_creation():
    """测试创建撤单请求对象"""
    req = CancelRequest(
        orderid="123456",
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
    )
    
    assert req.orderid == "123456"
    assert req.symbol == "BTCUSDT"
    assert req.exchange == Exchange.BINANCE


def test_subscribe_request_creation():
    """测试创建订阅请求对象"""
    req = SubscribeRequest(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
    )
    
    assert req.symbol == "BTCUSDT"
    assert req.exchange == Exchange.BINANCE
    assert req.vt_symbol == f"BTCUSDT.{Exchange.BINANCE.value}"


def test_vt_symbol_functions():
    """测试vt_symbol相关函数"""
    # 测试生成vt_symbol
    vt_symbol = generate_vt_symbol("BTCUSDT", Exchange.BINANCE)
    assert vt_symbol == "BTCUSDT.BINANCE"
    
    # 测试提取vt_symbol
    symbol, exchange = extract_vt_symbol(vt_symbol)
    assert symbol == "BTCUSDT"
    assert exchange == Exchange.BINANCE


def test_round_to_function():
    """测试round_to工具函数"""
    # 标准情况
    assert round_to(10.123, 0.01) == 10.12
    assert round_to(10.125, 0.01) == 10.13
    assert round_to(10.123, 0.1) == 10.1
    
    # 极端情况
    assert round_to(0, 0.01) == 0
    assert round_to(10.123, 1) == 10.0


def test_json_functions(tmpdir):
    """测试JSON文件读写函数"""
    # 创建临时文件路径
    temp_file = tmpdir.join("test_data.json")
    
    # 测试数据
    test_data = {
        "string": "hello",
        "number": 123,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2}
    }
    
    # 保存测试数据
    file_path = str(temp_file)
    save_json(file_path, test_data)
    
    # 读取测试数据
    loaded_data = load_json(file_path)
    
    # 验证数据一致
    assert loaded_data["string"] == test_data["string"]
    assert loaded_data["number"] == test_data["number"]
    assert loaded_data["list"] == test_data["list"]
    assert loaded_data["dict"] == test_data["dict"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
