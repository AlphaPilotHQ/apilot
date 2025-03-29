# APilot - AI-Driven Quant, Open to All

<p align="center">
    <img src ="https://img.shields.io/badge/version-0.1.2-blueviolet.svg"/>
    <img src ="https://img.shields.io/badge/python-3.10|3.11|3.12-blue.svg" />
    <img src ="https://img.shields.io/badge/license-MIT-green.svg" />
</p>

## 项目概述

APilot 是一款高性能量化交易框架，专注于加密货币和股票市场，由 AlphaPilot.tech 团队开发。
框架支持策略回测和实盘交易，为量化交易者提供全方位解决方案。

官方网站：[www.alphapilot.tech](https://www.alphapilot.tech)

## 核心功能

- **多种交易策略**：PA 策略 (价格行为)、因子策略 (开发中)
- **专业下单算法**：BestLimit、TWAP 算法
- **完善风控系统**：交易流量控制、持仓限制
- **高性能架构**：基于事件驱动的异步处理系统

## Database

- **CSV**：本地 CSV 文件存储，简单易用
- **MongoDB**：高性能本地数据库存储 (需安装插件)

## Gateway

### 加密货币

- **Binance**：支持币安现货和合约交易
- **即将支持**：Bybit、OKX 等主流交易所

### 股票市场

- **开发中**：美股、港股交易接口

## 策略类型

- **PA 策略**：支持趋势跟踪、均值回归等经典价格行为策略
- **因子策略**：基于多因子模型的量化策略 (开发中)


## 技术架构

### 目录结构
apilot/
├── apilot/                        # 主 Python 包
│   ├── analysis/                  # 分析工具
│   │   ├── __init__.py            # 模块初始化
│   │   └── performance.py         # 绩效分析工具
│   │
│   ├── core/                      # 核心组件
│   │   ├── __init__.py            # 模块初始化
│   │   ├── constant.py            # 常量定义
│   │   ├── database.py            # 数据库接口
│   │   ├── engine.py              # 引擎基类
│   │   ├── event.py               # 事件系统
│   │   ├── gateway.py             # 网关基类
│   │   ├── object.py              # 数据对象模型
│   │   └── utility.py             # 核心工具函数
│   │
│   ├── datafeed/                  # 数据馈送模块
│   │   ├── __init__.py            # 模块初始化
│   │   ├── csv_database.py        # CSV数据库
│   │   ├── data_manager.py        # 数据管理器
│   │   └── providers/             # 数据提供者
│   │       ├── __init__.py        # 模块初始化
│   │       ├── csv_provider.py    # CSV数据提供者
│   │       └── mongodb_provider.py # MongoDB数据提供者
│   │
│   ├── engine/                    # 引擎实现
│   │   ├── __init__.py            # 模块初始化
│   │   ├── backtest.py            # 回测引擎
│   │   ├── live.py                # 实盘引擎
│   │   └── oms_engine.py          # 订单管理系统引擎
│   │
│   ├── execution/                 # 执行模块
│   │   ├── __init__.py            # 模块初始化
│   │   ├── algo/                  # 算法交易
│   │   │   ├── __init__.py        # 模块初始化
│   │   │   ├── algo_engine.py     # 算法交易引擎
│   │   │   ├── algo_template.py   # 算法交易模板
│   │   │   ├── best_limit_algo.py # 最优限价算法
│   │   │   └── twap_algo.py       # 时间加权平均价格算法
│   │   └── gateway/               # 交易所网关
│   │       ├── __init__.py        # 模块初始化
│   │       └── binance.py         # 币安交易所接口
│   │
│   ├── optimizer/                 # 优化器模块
│   │   ├── __init__.py            # 模块初始化
│   │   └── optimizer.py           # 策略优化器
│   │
│   ├── plotting/                  # 绘图模块
│   │   ├── __init__.py            # 模块初始化
│   │   └── chart.py               # 图表绘制
│   │
│   ├── strategy/                  # 策略模块
│   │   ├── __init__.py            # 模块初始化
│   │   ├── template.py            # PA策略模板
│   │   └── examples/              # 策略示例
│   │       ├── pair_trading_strategy.py        # 配对交易策略
│   │       ├── pcp_arbitrage_strategy.py       # PCP套利策略
│   │       ├── portfolio_boll_channel_strategy.py # 投资组合布林通道策略
│   │       └── trend_following_strategy.py     # 趋势跟踪策略
│   │
│   ├── utils/                     # 工具函数
│   │   ├── __init__.py            # 模块初始化
│   │   ├── logger.py              # 日志工具
│   │   ├── order_manager.py       # 订单管理工具
│   │   ├── symbol.py              # 交易对工具
│   │   └── utility.py             # 通用工具函数
│   │
│   └── __init__.py                # 包初始化和API导出
│
├── data/                          # 数据目录
│
├── datafeed/                      # 外部数据提供模块
│   └── providers/                 # 数据提供者实现
│
├── examples/                      # 策略示例
│   ├── momentum_backtest.py       # 动量策略回测
│   ├── turtle_backtest.py         # 海龟策略回测
│   └── tick_test_strategy.py      # Tick数据测试策略
│
├── tests/                         # 测试文件
│   └── test_backtest.py           # 回测引擎测试
│
├── pyproject.toml                 # 项目配置
├── setup.py                       # 安装脚本
└── README.md                      # 项目说明

### 设计原则
- **Core 目录**：包含系统所有抽象接口和核心数据结构
  - 抽象基类 (BaseEngine, BaseGateway 等)
  - 数据模型 (OrderData, TickData 等)
  - 常量定义 (Direction, Interval 等)
  - 基础事件系统

- **功能目录**：包含各领域的具体实现
  - gateway/ - 各交易所接口实现
  - engine/ - 各种引擎的具体实现

## 开发计划

### 高优先级
- [ ] 发布到 PyPI

### 中优先级
- [ ] 添加更多单元测试
- [ ] 增加策略示例
- [ ] 测试风险管理模块
- [ ] 测试算法交易和 Websocket
- [ ] 测试优化器

### 长期规划
- [ ] 消息通知从邮件改为 Telegram
- [ ] 提供 Docker 部署方案
- [ ] 支持更多数据源

## 许可证

本项目采用 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件
