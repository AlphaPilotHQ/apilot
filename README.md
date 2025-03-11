# APilot - AI-Driven Quant, Open to All

<p align="center">
    <img src ="https://img.shields.io/badge/version-0.0.1-blueviolet.svg"/>
    <img src ="https://img.shields.io/badge/python-3.10|3.11.|3.12-blue.svg" />
</p>

APilot是一个量化交易系统开发框架。我们的网站是:www.alphapilot.tech.


## 功能特点

1. 轻量级量化交易框架，专注于提供高效稳定的交易功能。

2. 支持以下交易接口（gateway）：

    * 加密货币

        * Binance（[binance_gateway](vnpy/gateway/binance_gateway.py)）：币安现货和合约
        * OKX（[okx_gateway](vnpy/gateway/okx_gateway.py)）：OKX现货和合约
        * Bybit（[bybit_gateway](vnpy/gateway/bybit_gateway.py)）：Bybit现货和合约
    
    * 证券/期货

        * Interactive Brokers（[ib_gateway](vnpy/gateway/ib_gateway.py)）：盈透证券全球证券、期货、期权

3. 覆盖下述各类量化策略的交易应用（app）：

    * [cta_strategy]


    * [portfolio_strategy](https://www.github.com/vnpy/vnpy_portfoliostrategy)：组合策略模块，面向同时交易多合约的量化策略（Alpha、期权套利等），提供历史数据回测和实盘自动交易功能

    * [algo_trading](https://www.github.com/vnpy/vnpy_algotrading)：算法交易模块，提供多种常用的智能交易算法：TWAP、Sniper、Iceberg、BestLimit等

    * [portfolio_manager](https://www.github.com/vnpy/vnpy_portfoliomanager)：交易组合管理模块，以独立的策略交易组合（子账户）为基础，提供委托成交记录管理、交易仓位自动跟踪以及每日盈亏实时统计功能

    * [data_manager](https://www.github.com/vnpy/vnpy_datamanager)：历史数据管理模块，通过树形目录查看数据库中已有的数据概况，选择任意时间段数据查看字段细节，支持CSV文件的数据导入和导出

    * [data_recorder](https://www.github.com/vnpy/vnpy_datarecorder)：行情记录模块，基于图形界面进行配置，根据需求实时录制Tick或者K线行情到数据库中，用于策略回测或者实盘初始化

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


7. 对接下述各类数据服务的适配器接口（datafeed）：

    * 迅投研（[xt](https://www.github.com/vnpy/vnpy_xt)）：股票、期货、期权、基金、债券

    * 米筐RQData（[rqdata](https://www.github.com/vnpy/vnpy_rqdata)）：股票、期货、期权、基金、债券、黄金TD

    * 咏春大师（[voltrader](https://www.github.com/vnpy/vnpy_voltrader)）：期货、期权

    * 恒生UData（[udata](https://www.github.com/vnpy/vnpy_udata)）：股票、期货、期权

    * TuShare（[tushare](https://www.github.com/vnpy/vnpy_tushare)）：股票、期货、期权、基金

    * 万得Wind（[wind](https://www.github.com/vnpy/vnpy_wind)）：股票、期货、基金、债券

    * 天软Tinysoft（[tinysoft](https://www.github.com/vnpy/vnpy_tinysoft)）：股票、期货、基金、债券

    * 同花顺iFinD（[ifind](https://www.github.com/vnpy/vnpy_ifind)）：股票、期货、基金、债券

    * 天勤TQSDK（[tqsdk](https://www.github.com/vnpy/vnpy_tqsdk)）：期货


## 代码目录
vnpy/
├── .env
├── .env.template
├── .git/
├── .github/
├── .gitignore
├── .vscode/
│   └── settings.json
├── CHANGELOG.md
├── README.md
├── SOL-USDT.csv
├── examples/
│   ├── cta_backtesting/
│   │   ├── backtesting_demo.ipynb
│   │   └── portfolio_backtesting.ipynb
│   ├── portfolio_backtesting/
│   │   └── backtesting_demo.ipynb
│   ├── spread_backtesting/
│   │   └── backtesting.ipynb
│   └── veighna_trader/
│       ├── demo_script.py
│       └── run.py
├── requirements.txt
├── setup.cfg
├── setup.py
└── vnpy/
    ├── .DS_Store
    ├── __init__.py
    ├── database/
    │   ├── __init__.py
    │   ├── mongodb/
    │   │   ├── __init__.py
    │   │   └── mongodb_database.py
    │   ├── mysql/
    │   │   └── mysql_database.py
    │   └── postgresql/
    │       └── postgresql_database.py
    ├── event/
    │   ├── __init__.py
    │   └── engine.py
    ├── gateway/
    │   ├── binance_gateway.py
    │   ├── bybit_gateway.py
    │   ├── ib_gateway.py
    │   └── okx_gateway.py
    └── trader/
        ├── __init__.py
        ├── app.py
        ├── constant.py
        ├── converter.py
        ├── database.py
        ├── datafeed.py
        ├── engine.py
        ├── event.py
        ├── gateway.py
        ├── object.py
        ├── optimize.py
        ├── setting.py
        └── utility.py