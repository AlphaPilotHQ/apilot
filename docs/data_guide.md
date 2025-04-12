# APilot 数据接入指南

## 推荐数据接入方式：CSV

APilot框架推荐使用**CSV格式**作为标准数据接入方式。这是因为：

1. **简单可靠**：CSV是通用格式，易于理解和使用
2. **控制在用户**：用户可以自行处理和准备数据
3. **稳定接口**：CSV格式长期稳定，不会频繁变化
4. **低维护成本**：框架开发者无需适配各种数据源
5. **高度兼容**：几乎所有数据源都可以导出为CSV

## 标准CSV格式

推荐的CSV文件格式如下：

| datetime | open | high | low | close | volume | open_interest |
|----------|------|------|-----|-------|--------|---------------|
| 2023-01-01 00:00:00 | 16500.0 | 16550.0 | 16480.0 | 16520.0 | 100.5 | 0 |
| 2023-01-01 00:01:00 | 16520.0 | 16530.0 | 16510.0 | 16525.0 | 98.3 | 0 |

- `datetime`: 日期时间（格式可通过参数设置）
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `open_interest`: 持仓量（可选）

## 使用CSV数据

```python
from apilot import BacktestingEngine

# 创建回测引擎
engine = BacktestingEngine()
# ... 设置参数 ...

# 添加CSV数据
engine.add_csv_data(
    symbol="BTC-USDT.LOCAL", 
    filepath="data/BTCUSDT.csv",
    # 可选参数
    dtformat="%Y-%m-%d %H:%M:%S",  # 日期时间格式
    datetime_index=0,  # datetime列索引
    open_index=1,      # open列索引
    high_index=2,      # high列索引
    low_index=3,       # low列索引
    close_index=4,     # close列索引
    volume_index=5,    # volume列索引
    openinterest_index=6  # open_interest列索引
)
```

## 从其他数据源转换

如果您使用其他数据源（如MongoDB、MySQL等），我们建议将数据转换为标准CSV格式后再进行回测。

APilot提供了示例转换脚本：
- MongoDB → CSV: `examples/data_converters/mongo_to_csv.py`

### 示例：MongoDB转CSV

```bash
# 使用转换脚本
python examples/data_converters/mongo_to_csv.py \
  --symbol BTCUSDT \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --output data
```

## 高级：自定义数据提供者

如果您有特殊需求，可以实现自己的数据提供者。参考 `examples/data_providers/` 目录中的示例。

但请注意，我们不建议在生产环境中使用复杂的自定义数据提供者，因为：

1. 需要维护大量配置和字段映射
2. 增加维护成本和出错风险
3. 可能导致不一致的行为

## 大数据量处理建议

对于特别大量的数据（如数千万条记录），我们建议：

1. 分段转换和处理数据
2. 考虑降采样（如从分钟级转为小时级）
3. 使用高效的CSV读取库（如pandas）
4. 预处理数据以减少回测加载时间