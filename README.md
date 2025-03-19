# APilot - AI-Driven Quant, Open to All

<p align="center">
    <img src ="https://img.shields.io/badge/version-0.1.2-blueviolet.svg"/>
    <img src ="https://img.shields.io/badge/python-3.10|3.11|3.12-blue.svg" />
    <img src ="https://img.shields.io/badge/license-MIT-green.svg" />
</p>

## 项目概述

APilot是一款专注于加密货币市场的高性能量化交易框架，由AlphaPilot.tech团队开发。
该框架可以回测，实盘。

官方网站：[www.alphapilot.tech](https://www.alphapilot.tech)

## 核心功能

- **内置多种交易策略**：cta_strategy, spread_strategy, factor_strategy[todo]
- **专业下单算法**：BestLimit、Iceberg、TWAP
- **安心睡觉的风控**：交易流控、持仓限制
- **高性能架构**：谁用谁知道

## Database

- **CSV**：本地CSV文件存储
- **MySQL**：本地MySQL数据库
- **PostgreSQL**：本地PostgreSQL数据库
- **TimescaleDB**：本地TimescaleDB数据库
- **MongoDB**：本地MongoDB数据库

## Gateway

### 加密货币

- **Binance**：币安现货和合约
- **OKX**：OKX现货和合约
- **Bybit**：Bybit现货和合约

### 证券/期货

- **Interactive Brokers**：盈透证券全球证券、期货、期权

## Strategy

- **[CTA策略]**：经典的趋势跟踪和均值回归策略
- **[价差交易]**：支持自定义价差，实时计算价差行情和持仓，支持价差算法交易以及自动价差策略
- **[投资组合策略]**：支持多资产投资组合回测和实盘交易

## 技术组件

- **REST Client**：基于协程异步IO的高性能REST API客户端
- **Websocket Client**：高性能Websocket API客户端，支持和REST Client共用事件循环
- **事件驱动引擎**：简洁易用的事件驱动系统，作为交易程序的核心
- **数据库适配器**：支持多种数据库，包括CSV、MySQL、PostgreSQL、TimescaleDB、MongoDB等

## 安装指南

### 依赖环境

- Python 3.10/3.11/3.12
- 推荐使用venv创建虚拟环境

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/AlphaPilotHQ/apilot.git

# 进入项目目录
cd apilot

# 安装依赖
pip install -e .
```

## 使用示例

```python
# 策略回测示例
from vnpy.trader.app.cta_strategy.backtesting import BacktestingEngine
from vnpy.trader.object import BarData
from datetime import datetime

# 创建回测引擎
engine = BacktestingEngine()

# 设置回测参数
engine.set_parameters(
    vt_symbol="BTC-USDT.BINANCE",
    interval="1h",
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31),
    rate=0.0003,
    slippage=0,
    size=1,
    pricetick=0.01,
    capital=100000,
)

# 加载策略
engine.add_strategy(MyStrategy, {})

# 运行回测
engine.run_backtesting()

# 计算结果
engine.calculate_result()
```


## 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件
