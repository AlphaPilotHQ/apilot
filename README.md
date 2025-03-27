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

- **MongoDB**：本地MongoDB数据库，需要装插件

## Gateway

### 加密货币

- **Binance**：币安现货和合约
- bybit okx 也即将支持


### 美股

- 还没做，但要做

## Strategy

- **[CTA策略]**：经典的趋势跟踪和均值回归策略

## 技术组件

- **REST Client**：基于协程异步IO的高性能REST API客户端
- **Websocket Client**：高性能Websocket API客户端，支持和REST Client共用事件循环
- **事件驱动引擎**：简洁易用的事件驱动系统，作为交易程序的核心


## 技术架构

### 具体架构
apilot
├── apilot                        # 主Python包



### 设计原则
Core 目录 - 包含系统的所有抽象接口和核心数据结构
    抽象基类（如 BaseEngine, BaseGateway）
    数据模型（如 OrderData, TickData）
    常量定义（如 Direction, Interval）
    基础事件系统


功能目录 - 包含各自领域的具体实现
    gateway/ - 各交易所接口实现
    engine/ - 各种引擎的具体实现
    等等

## TODO：
## 重要TODO：
- upload to pypi


## 可选TODO
- [ ] 添加更多单元测试
- [ ] 添加更多策略示例
- test riskmanager
- test algotrading & websocket
- test optimizer

- tzlocal库的删除
- OffsetConverter类删除



## 未来TODO
- 修改email成telegram
- 图表系统拆出来重新做
- 提供Docker部署方案
- 支持更多数据源



## 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件
