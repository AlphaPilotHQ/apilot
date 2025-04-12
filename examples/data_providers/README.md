# 数据提供者示例

本目录包含了各种数据源的示例实现。APilot框架推荐使用CSV作为标准数据格式，但这些示例展示了如何实现自定义数据提供者。

## 推荐方式：使用CSV数据

APilot框架设计为优先使用CSV作为标准数据格式。这提供了最佳的稳定性、可靠性和易用性。

```python
from apilot import BacktestingEngine

# 创建回测引擎
engine = BacktestingEngine()
# ... 设置参数 ...

# 添加CSV数据
engine.add_csv_data(
    symbol="BTC-USDT.LOCAL", 
    filepath="data/BTCUSDT.csv"
)
```

## 标准CSV格式

建议的CSV文件格式包含以下列：
- `datetime`: 日期时间（ISO格式）
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `open_interest`: 持仓量（可选）

## MongoDB提供者示例

`mongodb_provider.py` 是一个示例实现，展示如何创建自定义数据提供者来连接MongoDB。**注意这仅作为参考**，不推荐在生产环境中使用，因为它需要大量的字段映射配置。

如果您使用MongoDB存储数据，我们建议使用 `examples/data_converters/mongo_to_csv.py` 脚本将数据转换为标准CSV格式，然后使用CSV数据提供者进行回测。

## 使用示例数据提供者

如果您仍然希望直接使用MongoDB数据提供者，可以：

1. 将示例代码复制到您的项目中
2. 手动注册数据提供者：

```python
from apilot.datafeed import register_provider
from your_module import MongoDBDatabase

# 注册提供者
register_provider("mongodb", MongoDBDatabase)

# 然后可以使用
engine.add_mongodb_data(
    symbol="BTC-USDT.LOCAL",
    database="your_db",
    collection="your_collection",
    # ... 其他配置参数 ...
)
```

## 创建自定义数据提供者

如果您需要实现自己的数据提供者，请遵循以下步骤：

1. 继承 `apilot.core.database.BaseDatabase` 类
2. 实现 `load_bar_data` 方法
3. 注册您的数据提供者