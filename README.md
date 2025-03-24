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
│   ├── __init__.py
│   ├── core                      # 核心组件
│   │   ├── __init__.py
│   │   ├── app.py                # 应用基类
│   │   ├── constant.py           # 常量定义
│   │   ├── converter.py          # 转换器
│   │   ├── csv_database.py       # CSV数据库
│   │   ├── database.py           # 数据库接口
│   │   ├── engine.py             # 引擎基类和主引擎
│   │   ├── event.py              # 事件类型和事件引擎
│   │   ├── gateway.py            # 接口基类
│   │   ├── object.py             # 数据对象
│   │   ├── setting.py            # 全局设置
│   │   └── utility.py            # 工具函数
│   ├── gateway                   # 交易所/券商接口
│   │   ├── __init__.py
│   │   └── binance.py            # 币安交易所接口
│   ├── engine                    # 引擎模块
│   │   ├── __init__.py
│   │   ├── backtest.py           # 回测引擎
│   │   ├── email_engine.py       # 邮件引擎
│   │   ├── live.py               # 实盘引擎
│   │   ├── log_engine.py         # 日志引擎
│   │   └── oms_engine.py         # 订单管理引擎
│   ├── strategy                  # 策略模块
│   │   ├── __init__.py
│   │   ├── template.py           # 策略模板
│   │   └── examples              # 策略示例
│   │       ├── pair_trading_strategy.py          # 配对交易策略
│   │       ├── pcp_arbitrage_strategy.py         # PCP套利策略
│   │       ├── portfolio_boll_channel_strategy.py # 投资组合布林通道策略
│   │       └── trend_following_strategy.py       # 趋势跟踪策略
│   ├── riskmanager               # 风控引擎
│   │   ├── __init__.py
│   │   └── engine.py             # 风控引擎实现
│   ├── algotrading               # 算法交易
│   │   ├── __init__.py
│   │   ├── algo_base.py          # 算法基类
│   │   ├── algo_engine.py        # 算法引擎
│   │   ├── algo_template.py      # 算法模板
│   │   ├── best_limit_algo.py    # 最优限价算法
│   │   ├── iceberg_algo.py       # 冰山算法
│   │   ├── sniper_algo.py        # 狙击算法
│   │   ├── stop_algo.py          # 止损算法
│   │   └── twap_algo.py          # 时间加权平均价格算法
│   └── optimizer                 # 策略优化
│       ├── __init__.py
│       └── optimizer.py          # 优化引擎
│
├── docs                          # 文档
├── examples                      # 示例
│   ├── cta_backtesting          # CTA策略回测示例
│   ├── spread_backtesting       # 价差交易回测示例
│   ├── algo_backtesting         # 算法交易回测示例
│   └── portfolio_backtesting    # 投资组合回测示例
├── tests                         # 测试
├── requirements.txt              # 依赖
├── setup.py                      # 打包配置
├── README.md                     # 项目说明
└── pyproject.toml                # 项目配置


### 设计原则
Core 目录 - 包含系统的所有抽象接口和核心数据结构
    抽象基类（如 BaseEngine, BaseGateway）
    数据模型（如 OrderData, TickData）
    常量定义（如 Direction, Interval）
    基础事件系统


功能目录 - 包含各自领域的具体实现
    gateway/ - 各交易所接口实现
    engine/ - 各种引擎的具体实现
    risk/ - 风险管理具体实现
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
- [ ] 完全移除CTA引擎中已弃用的write_log方法
- [ ] 更新所有策略代码，使用主引擎的log_xxx方法替代write_log
- [ ] 为回测引擎添加与主引擎一致的log_debug/log_info等方法，统一API
- tzlocal库的删除
- OffsetConverter类删除



## 未来TODO
- 修改email成telegram
- 图表系统拆出来重新做
- 提供Docker部署方案
- 支持更多数据源



## 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件
