"""
VeighNa交易平台Binance实盘交易测试脚本
实现最简单的连接、账户查询和下单功能
"""
import os
import sys
from time import sleep
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件中的环境变量
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.constant import Direction, Offset, OrderType, Exchange
from vnpy.trader.object import OrderRequest
from vnpy.trader.setting import SETTINGS
from vnpy.gateway.binance_gateway import BinanceGateway

# 加载API密钥
api_key = os.environ.get("BINANCE_API_KEY", "")
api_secret = os.environ.get("BINANCE_SECRET_KEY", "")

# 创建连接设置
binance_setting = {
    "API Key": api_key,
    "Secret Key": api_secret,
    "Session Number": 3,
    "Proxy Host": os.environ.get("BINANCE_PROXY_HOST", ""),
    "Proxy Port": int(os.environ.get("BINANCE_PROXY_PORT", 0)),
}

# 主函数
def main() -> None:
    """主函数 - 启动Binance交易测试"""
    try:
        print("======= 开始Binance实盘交易测试 =======")
        
        # 创建引擎
        event_engine = EventEngine()
        main_engine = MainEngine(event_engine)
        
        # 添加Binance网关
        main_engine.add_gateway(BinanceGateway)
        
        # 连接Binance
        main_engine.connect(binance_setting, "BINANCE")
        print("正在连接到Binance...")
        
        # 简单等待几秒，让网关连接和数据加载
        sleep(5)
        
        # 查询现货账户余额
        print("-------- 查询现货账户余额 --------")
        accounts = main_engine.get_all_accounts()
        for account in accounts:
            print(f"账户 {account.accountid}: 余额 = {account.balance}, 可用 = {account.available}")
        
        # 创建下单请求
        req = OrderRequest(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            direction=Direction.LONG,
            type=OrderType.MARKET,
            volume= 0.001,
            price=0,  # 市价单价格填0
            offset=Offset.NONE  # 现货没有开平方向
        )
        
        # 下单
        print(f"下单: {req.symbol}, {'买入' if req.direction == Direction.LONG else '卖出'}, {req.type.value}, 数量 {req.volume}")
        vt_orderid = main_engine.send_order(req, 'BINANCE')
        
        if not vt_orderid:
            print("❌ 下单失败")
            main_engine.close()
            sys.exit(1)
        
        print(f"✅ 下单成功: {vt_orderid}")
        
        # 测试结束
        print("======= Binance实盘交易测试完成 =======")
        
        # 关闭引擎
        main_engine.close()
        
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
