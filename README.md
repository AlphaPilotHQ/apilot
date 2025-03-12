# APilot - AI-Driven Quant, Open to All

<p align="center">
    <img src ="https://img.shields.io/badge/version-0.0.1-blueviolet.svg"/>
    <img src ="https://img.shields.io/badge/python-3.10|3.11.|3.12-blue.svg" />
</p>

APilot是一个量化交易系统开发框架。我们的网站是:www.alphapilot.tech.

## TODO
- algotrading should use websocket client


## 介绍

1. 基本功能
- 本地数据导入
- 策略回测
- 策略实盘

2. Plus功能
- 支持数据库读取策略
- 内置多种策略（cta，spread）
- 支持算法下单
- 支持风控管理

## 功能特点

1. 开箱即用。轻量级量化交易框架，专注于提供高效稳定的交易功能。

2. 支持以下交易接口（gateway）：

    * 加密货币

        * Binance（[binance_gateway](vnpy/gateway/binance_gateway.py)）：币安现货和合约
        * OKX（[okx_gateway](vnpy/gateway/okx_gateway.py)）：OKX现货和合约
        * Bybit（[bybit_gateway](vnpy/gateway/bybit_gateway.py)）：Bybit现货和合约
    
    * 证券/期货

        * Interactive Brokers（[ib_gateway](vnpy/gateway/ib_gateway.py)）：盈透证券全球证券、期货、期权

3. 覆盖下述各类量化策略（strategy）：

    * [cta_strategy]

    * [spread_trading](https://www.github.com/vnpy/vnpy_spreadtrading)：价差交易模块，支持自定义价差，实时计算价差行情和持仓，支持价差算法交易以及自动价差策略两种模式



    * [portfolio_strategy](https://www.github.com/vnpy/vnpy_portfoliostrategy)：组合策略模块，面向同时交易多合约的量化策略（Alpha、期权套利等），提供历史数据回测和实盘自动交易功能

    * [portfolio_manager](https://www.github.com/vnpy/vnpy_portfoliomanager)：交易组合管理模块，以独立的策略交易组合（子账户）为基础，提供委托成交记录管理、交易仓位自动跟踪以及每日盈亏实时统计功能


    * [risk_manager](https://www.github.com/vnpy/vnpy_riskmanager)：风险管理模块，提供包括交易流控、下单数量、活动委托、撤单总数等规则的统计和限制，有效实现前端风控功能


4. Python交易API接口封装（api），提供上述交易接口的底层对接实现。

    * REST Client（[rest](https://www.github.com/vnpy/vnpy_rest)）：基于协程异步IO的高性能REST API客户端，采用事件消息循环的编程模型，支持高并发实时交易请求发送

    * Websocket Client（[websocket](https://www.github.com/vnpy/vnpy_websocket)）：基于协程异步IO的高性能Websocket API客户端，支持和REST Client共用事件循环并发运行

5. 简洁易用的事件驱动引擎（event），作为事件驱动型交易程序的核心。

6. 对接各类数据库的适配器接口（database）：

    * CSV类

        * CSV（[csv](https://www.github.com/vnpy/vnpy_csv)）：CSV文件数据库，直接从CSV文件读取数据

    * SQL类

        * MySQL（[mysql](https://www.github.com/vnpy/vnpy_mysql)）：主流的开源关系型数据库，文档资料极为丰富，且可替换其他NewSQL兼容实现（如TiDB）

        * PostgreSQL（[postgresql](https://www.github.com/vnpy/vnpy_postgresql)）：特性更为丰富的开源关系型数据库，支持通过扩展插件来新增功能，只推荐熟手使用

    * NoSQL类

        * TimescaleDB（[timescaledb](https://www.github.com/vnpy/vnpy_timescaledb)）：基于PostgreSQL开发的一款时序数据库，以插件化扩展的形式安装，支持自动按空间和时间对数据进行分区

        * MongoDB（[mongodb](https://www.github.com/vnpy/vnpy_mongodb)）：基于分布式文件储存（bson格式）的文档式数据库，内置的热数据内存缓存提供更快读写速度



