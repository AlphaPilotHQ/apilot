"""
MongoDB数据转换为CSV示例脚本

此脚本从MongoDB数据库中提取数据，并转换为标准CSV格式，用于回测。
"""

import os
import argparse
from datetime import datetime
import pandas as pd
from pymongo import MongoClient


def convert_mongo_to_csv(
    mongo_uri=None,
    database="alphapilot",
    collection="symbol_trade",
    symbol="BTCUSDT",
    start_date="2023-01-01",
    end_date="2025-12-31",
    output_dir="data",
    symbol_field="symbol",
    datetime_field="kline_st",
    open_field="first_trade_price",
    high_field="high_price",
    low_field="low_price",
    close_field="last_trade_price",
    volume_field="trade_volume",
    timestamp_ms=True,
):
    """将MongoDB数据转换为标准CSV格式

    Args:
        mongo_uri: MongoDB连接字符串，默认从环境变量MONGO_URI获取
        database: 数据库名称
        collection: 集合名称
        symbol: 交易对符号（如BTCUSDT）
        start_date: 开始日期（格式：YYYY-MM-DD）
        end_date: 结束日期（格式：YYYY-MM-DD）
        output_dir: 输出目录
        symbol_field: 交易对字段名
        datetime_field: 日期时间字段名
        open_field: 开盘价字段名
        high_field: 最高价字段名
        low_field: 最低价字段名
        close_field: 收盘价字段名
        volume_field: 成交量字段名
        timestamp_ms: 是否使用毫秒时间戳
    """
    # 获取MongoDB URI
    if not mongo_uri:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("必须提供MongoDB连接URI，或设置MONGO_URI环境变量")

    # 解析日期
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    # 连接MongoDB
    client = MongoClient(mongo_uri)
    db = client[database]
    coll = db[collection]

    # 创建查询条件
    query = {symbol_field: symbol}
    
    # 添加日期范围条件
    if timestamp_ms:
        # 毫秒时间戳
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)
        query[datetime_field] = {"$gte": start_ts, "$lte": end_ts}
    else:
        # 日期时间对象
        query[datetime_field] = {"$gte": start, "$lte": end}
    
    # 设置投影，只获取需要的字段
    projection = {
        symbol_field: 1,
        datetime_field: 1,
        open_field: 1,
        high_field: 1,
        low_field: 1,
        close_field: 1,
        volume_field: 1,
        "_id": 0
    }
    
    # 查询数据并转换为列表
    print(f"正在查询 {symbol} 数据...")
    documents = list(coll.find(query, projection).sort(datetime_field, 1))
    print(f"找到 {len(documents)} 条记录")
    
    if not documents:
        print("没有找到匹配记录，请检查查询条件")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(documents)
    
    # 处理时间戳
    if timestamp_ms:
        df["datetime"] = pd.to_datetime(df[datetime_field], unit="ms")
    else:
        df["datetime"] = df[datetime_field]
    
    # 重命名列以符合标准格式
    renamed_df = pd.DataFrame()
    renamed_df["datetime"] = df["datetime"]
    renamed_df["open"] = df[open_field].astype(float)
    renamed_df["high"] = df[high_field].astype(float)
    renamed_df["low"] = df[low_field].astype(float)
    renamed_df["close"] = df[close_field].astype(float)
    renamed_df["volume"] = df[volume_field].astype(float)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV
    output_file = os.path.join(output_dir, f"{symbol}.csv")
    renamed_df.to_csv(output_file, index=False)
    print(f"数据已保存到 {output_file}")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="从MongoDB导出数据到标准CSV格式")
    parser.add_argument("--uri", help="MongoDB连接URI")
    parser.add_argument("--db", default="alphapilot", help="数据库名称")
    parser.add_argument("--collection", default="symbol_trade", help="集合名称")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易对符号")
    parser.add_argument("--start", default="2023-01-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--output", default="data", help="输出目录")
    args = parser.parse_args()
    
    convert_mongo_to_csv(
        mongo_uri=args.uri,
        database=args.db,
        collection=args.collection,
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )