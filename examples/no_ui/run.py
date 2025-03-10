import multiprocessing
import sys
from time import sleep
from datetime import datetime, time
from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy.gateway.binance import BinanceGateway
from vnpy_ctastrategy import CtaStrategyApp
from vnpy_ctastrategy.base import EVENT_CTA_LOG


SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True


# 从文件加载Binance配置，或使用默认设置
try:
    import os
    from vnpy.trader.utility import load_json
    # 使用当前文件的目录路径作为基准路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "connect_binance.json")
    print(f"加载配置文件: {config_path}")
    binance_setting = load_json(config_path)
    print(f"配置内容: API Key={binance_setting.get('API Key', '')[:4]}...")
except Exception as e:
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

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(binance_setting, "BINANCE")
    main_engine.write_log("连接Binance接口")

    # 等待连接建立
    sleep(5)
    
    # 获取账户余额
    main_engine.write_log("正在获取账户余额...")
    accounts = main_engine.get_all_accounts()
    if accounts:
        for account in accounts:
            main_engine.write_log(f"账户ID: {account.accountid}")
            main_engine.write_log(f"账户余额: {account.balance}")
            main_engine.write_log(f"可用资金: {account.available}")
            main_engine.write_log(f"冻结资金: {account.frozen}")
    else:
        main_engine.write_log("未获取到账户信息，可能是连接尚未完全建立")

    # 继续等待，确保有足够时间完成初始化
    sleep(5)

    cta_engine.init_engine()
    main_engine.write_log("CTA策略初始化完成")

    cta_engine.init_all_strategies()
    sleep(60)   # Leave enough time to complete strategy initialization
    main_engine.write_log("CTA策略全部初始化")

    cta_engine.start_all_strategies()
    main_engine.write_log("CTA策略全部启动")

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
