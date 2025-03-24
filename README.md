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


### 证券/期货

- 还没做，但要做

## Strategy

- **[CTA策略]**：经典的趋势跟踪和均值回归策略
- **[价差交易]**：支持自定义价差，实时计算价差行情和持仓，支持价差算法交易以及自动价差策略
- **[投资组合策略]**：支持多资产投资组合回测和实盘交易

## 技术组件

- **REST Client**：基于协程异步IO的高性能REST API客户端
- **Websocket Client**：高性能Websocket API客户端，支持和REST Client共用事件循环
- **事件驱动引擎**：简洁易用的事件驱动系统，作为交易程序的核心


## 技术架构
apilot
├── src                           # 主Python包(核心框架)
│   ├── __init__.py
│   ├── event                     # 事件引擎
│   │   └── engine.py
│   ├── gateway                   # 交易所/券商接口
│   │   └── binance_gateway.py
│   ├── data                      # 数据模块(仅保留CSV读取等轻量功能)
│   │   └── csv_loader.py         # 内置对csv数据的读取支持
│   ├── strategy                  # 策略引擎
│   │   ├── cta                   # CTA策略
│   │   │   ├── base.py
│   │   │   ├── engine.py
│   │   │   ├── template.py
│   │   │   └── backtesting.py    # CTA策略回测
│   ├── risk                      # 风控引擎
│   │   └── engine.py
│   ├── execution                 # 交易执行(智能订单算法)
│   │   ├── engine.py             # 执行引擎管理入口
│   │   └── algos
│   │       ├── best_limit_algo.py
│   │       ├── iceberg_algo.py
│   │       ├── sniper_algo.py
│   │       ├── stop_algo.py
│   │       └── twap_algo.py
│   ├── order                     # 订单管理
│   │   └── manager.py
│   ├── account                   # 账户管理
│   │   └── manager.py
│   ├── backtester                # 通用回测/优化(可选)
│   │   ├── engine.py
│   │   ├── optimize.py           # 策略参数优化
│   │   └── performance.py        # 回测绩效评估
│   ├── logmonitor                # 日志 & 监控合并 (事件驱动)
│   │   ├── logger.py             # 自定义事件驱动Logger
│   │   └── notify.py             # 各种通知 (email, telegram 等)
│   └── utils                     # 通用工具/常量/对象
│       ├── constant.py
│       ├── object.py
│       ├── setting.py
│       └── utility.py
│
├── docs                          # 文档
├── examples                      # 示例
│   ├── cta_backtesting
│   └── spread_backtesting
├── tests                         # 测试
├── requirements.txt              # 依赖
├── README.md
└── pyproject.toml                # 发布到PyPI

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
- put event涉及到GUI部分删除
- tzlocal库的删除



## 未来TODO
- 修改email成telegram
- 图表系统拆出来重新做
- 提供Docker部署方案
- 支持更多数据源



## 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件
