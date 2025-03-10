import multiprocessing
import sys
from time import sleep
from datetime import datetime, time
from logging import INFO
import decimal

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import OrderRequest, SubscribeRequest, OrderData, TradeData
from vnpy.trader.constant import Direction, Exchange, Offset, OrderType, Status
from vnpy.trader.utility import round_to

# 设置基础配置
SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True

# 设置数据源配置 
SETTINGS["datafeed.name"] = "default"
SETTINGS["datafeed.username"] = ""
SETTINGS["datafeed.password"] = ""

from vnpy.gateway.binance import BinanceGateway
from vnpy_ctastrategy import CtaStrategyApp
from vnpy_ctastrategy.base import EVENT_CTA_LOG


# 日志和数据源配置已在上方设置


# 从环境变量或配置文件加载Binance配置
try:
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from vnpy.trader.utility import load_json
    
    # 尝试加载.env文件
    root_path = Path(__file__).parent.parent.parent  # examples/live_test -> root
    env_file = root_path / ".env"
    env_template = root_path / ".env.template"
    
    if env_file.exists():
        load_dotenv(env_file)
        print(f"已加载环境变量配置: {env_file}")
        
        # 从环境变量读取 Binance 配置
        api_key = os.environ.get("VNPY_BINANCE_API_KEY", "")
        secret_key = os.environ.get("VNPY_BINANCE_SECRET_KEY", "")
        session_number = int(os.environ.get("VNPY_BINANCE_SESSION_NUMBER", "3"))
        proxy_host = os.environ.get("VNPY_BINANCE_PROXY_HOST", "")
        proxy_port = int(os.environ.get("VNPY_BINANCE_PROXY_PORT", "0"))
        testnet = os.environ.get("VNPY_BINANCE_TESTNET", "false").lower() == "true"
        
        binance_setting = {
            "API Key": api_key,
            "Secret Key": secret_key,
            "Session Number": session_number,
            "Proxy Host": proxy_host,
            "Proxy Port": proxy_port,
            "Testnet": testnet
        }
        
        if api_key and secret_key:
            print(f"从环境变量加载了 Binance 配置: API Key={api_key[:4] if api_key else ''}...")
        else:
            # 如果环境变量中没有配置，则尝试从文件加载
            print("环境变量中没有发现 Binance 配置，尝试从文件加载...")
            # 使用当前文件的目录路径作为基准路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "connect_binance.json")
            print(f"尝试加载配置文件: {config_path}")
            try:
                binance_setting = load_json(config_path)
                print(f"从文件加载了 Binance 配置: API Key={binance_setting.get('API Key', '')[:4] if binance_setting.get('API Key') else ''}...")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
    else:
        # 如果.env文件不存在，继续使用原来的方式
        print(f"警告: 未找到.env文件，请从.env.template创建。尝试从文件加载...")
        # 使用当前文件的目录路径作为基准路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "connect_binance.json")
        print(f"尝试加载配置文件: {config_path}")
        try:
            binance_setting = load_json(config_path)
            print(f"从文件加载了 Binance 配置: API Key={binance_setting.get('API Key', '')[:4] if binance_setting.get('API Key') else ''}...")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            binance_setting = {
                "API Key": "",  # 请填写您的API Key
                "Secret Key": "",  # 请填写您的Secret Key
                "Session Number": 3,
                "Proxy Host": "",
                "Proxy Port": 0,
                "Testnet": True  # 设置为True使用测试网络
            }
except Exception as e:
    print(f"加载配置异常: {e}")
    binance_setting = {
        "API Key": "",  # 请填写您的API Key
        "Secret Key": "",  # 请填写您的Secret Key
        "Session Number": 3,
        "Proxy Host": "",
        "Proxy Port": 0,
        "Testnet": True  # 设置为True使用测试网络
    }


# Chinese futures market trading period (day/night)
DAY_START = time(8, 45)
DAY_END = time(15, 0)

NIGHT_START = time(20, 45)
NIGHT_END = time(2, 45)


def check_trading_period():
    """
    对于Binance加密货币交易所，不需要检查交易时间，因为它是24小时交易的
    """
    # 始终返回 True，即表示当前时间段可以交易
    return True


def place_test_order(main_engine, symbol, price, volume, direction=Direction.LONG, order_type=OrderType.LIMIT):
    """
    下单测试函数
    """
    main_engine.write_log(f"准备发送测试订单: {symbol}, 方向: {direction.value}, 价格: {price}, 数量: {volume}")
    
    # 先订阅市场行情
    req = SubscribeRequest(
        symbol=symbol,
        exchange=Exchange.BINANCE
    )
    main_engine.subscribe(req, "BINANCE")
    main_engine.write_log(f"订阅行情: {symbol}")
    
    # 创建订单请求
    order_req = OrderRequest(
        symbol=symbol,
        exchange=Exchange.BINANCE,
        direction=direction,
        type=order_type,
        volume=volume,
        price=price,
        offset=Offset.NONE
    )
    
    # 发送订单
    vt_orderid = main_engine.send_order(order_req, "BINANCE")
    main_engine.write_log(f"发送订单成功, 订单ID: {vt_orderid}")
    return vt_orderid


def on_order(order: OrderData):
    """
    订单状态更新回调函数
    """
    print(f"订单更新 - ID: {order.orderid}, 状态: {order.status.value}, 成交量: {order.traded}/{order.volume}")


def on_trade(trade: TradeData):
    """
    成交回调函数
    """
    print(f"订单成交 - ID: {trade.orderid}, 成交ID: {trade.tradeid}, 价格: {trade.price}, 数量: {trade.volume}")


def run_child():
    """
    Running in the child process.
    """
    # 修改为可以同时在文件和控制台输出日志
    SETTINGS["log.file"] = True
    SETTINGS["log.console"] = True

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(BinanceGateway)
    cta_engine = main_engine.add_app(CtaStrategyApp)
    main_engine.write_log("主引擎创建成功")
    
    # 注册回调函数
    event_engine.register("eOrder", on_order)
    event_engine.register("eTrade", on_trade)

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(binance_setting, "BINANCE")
    main_engine.write_log("连接Binance接口")

    # 等待连接建立
    sleep(5)
    
    # 检查交易所连接状态
    gateway_name = "BINANCE"
    gateway = main_engine.get_gateway(gateway_name)
    
    # 不直接检查status属性，而是通过功能性验证连接状态
    if gateway:
        main_engine.write_log(f"✅ {gateway_name}交易所连接成功!")
        
        # 获取账户余额
        main_engine.write_log("正在获取账户余额...")
        accounts = main_engine.get_all_accounts()
        if accounts:
            main_engine.write_log(f"✅ 成功获取账户信息，总共{len(accounts)}个账户")
            for account in accounts:
                main_engine.write_log(f"账户ID: {account.accountid}")
                main_engine.write_log(f"账户余额: {account.balance}")
                main_engine.write_log(f"可用资金: {account.available}")
                main_engine.write_log(f"冻结资金: {account.frozen}")
            
            # 对于Binance，通常使用USDT作为交易基础货币
            usdt_balance = 0
            for account in accounts:
                if account.accountid == "USDT":
                    usdt_balance = account.available
                    main_engine.write_log(f"可用USDT: {usdt_balance}")
                    break
        else:
            main_engine.write_log("❌ 未获取到账户信息，可能是连接尚未完全建立或API权限不足")
    else:
        main_engine.write_log(f"❌ {gateway_name}交易所连接失败!")

    # 继续等待，确保有足够时间完成初始化
    sleep(5)

    cta_engine.init_engine()
    main_engine.write_log("CTA策略初始化完成")

    cta_engine.init_all_strategies()
    sleep(10)   # 留出足够时间完成策略初始化
    main_engine.write_log("CTA策略全部初始化")

    cta_engine.start_all_strategies()
    main_engine.write_log("CTA策略全部启动")
    
    # 测试行情订阅功能
    try:
        main_engine.write_log("开始测试行情订阅功能...")
        sleep(2)  # 等待连接完全建立
        
        # 订阅BTC/USDT行情
        req = SubscribeRequest(
            symbol="BTCUSDT",
            exchange=Exchange.BINANCE
        )
        main_engine.subscribe(req, "BINANCE")
        main_engine.write_log("已订阅BTCUSDT行情")
        sleep(3)  # 等待行情更新
        
        # 检查是否收到行情数据
        ticks = main_engine.get_all_ticks()
        if ticks:
            main_engine.write_log(f"✅ 成功接收行情数据，共{len(ticks)}个交易对")
            for tick in ticks:
                if tick.symbol == "BTCUSDT":
                    main_engine.write_log(f"BTCUSDT 当前价格: {tick.last_price}")
                    main_engine.write_log(f"BTCUSDT 买一价: {tick.bid_price_1}, 卖一价: {tick.ask_price_1}")
                    main_engine.write_log(f"BTCUSDT 成交量: {tick.volume}")
                    break
        else:
            main_engine.write_log("❌ 未接收到行情数据，可能是连接问题或行情订阅失败")
            
        main_engine.write_log("行情订阅测试完成")
        
        # 注释掉实际下单操作，仅进行连接测试
        # 如果您想测试下单功能，请取消以下注释
        """
        # 买入 BTC 现货，使用市价单，金额10 USDT
        main_engine.write_log("准备买入BTC现货，使用市价单...")
        
        # 假设当前 BTC 价格大约60000 USDT，10 USDT 可买约 0.00016 BTC
        # 为了安全起见，使用稍小的数量
        spot_volume = 0.00016  # 大约相当于10 USDT
        
        # 下单现货
        spot_req = OrderRequest(
            symbol="BTCUSDT",  # Binance的现货交易对格式
            exchange=Exchange.BINANCE,
            direction=Direction.LONG,  # 买入
            type=OrderType.MARKET,     # 市价单
            volume=spot_volume,
            price=0,  # 市价单时价格为0
            offset=Offset.NONE
        )
        
        # 发送订单
        spot_orderid = main_engine.send_order(spot_req, "BINANCE")
        main_engine.write_log(f"已发送BTC现货买入订单，订单ID: {spot_orderid}")
        """
    except Exception as e:
        main_engine.write_log(f"测试过程中出错: {e}")

    # 简化循环，等待用户手动终止
    print("程序运行中，按Ctrl+C可以终止")
    try:
        while True:
            sleep(10)
    except KeyboardInterrupt:
        print("收到终止信号，关闭子进程")
        main_engine.close()
        sys.exit(0)


def run_parent():
    """
    Running in the parent process.
    """
    print("启动CTA策略守护父进程")

    child_process = None

    while True:
        trading = check_trading_period()

        # Start child process in trading period
        if trading and child_process is None:
            print("启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            if not child_process.is_alive():
                child_process = None
                print("子进程关闭成功")

        sleep(5)


if __name__ == "__main__":
    run_parent()
