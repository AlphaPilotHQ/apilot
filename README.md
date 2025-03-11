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

    * [cta_strategy](https://www.github.com/vnpy/vnpy_ctastrategy)：CTA策略引擎模块，在保持易用性的同时，允许用户针对CTA类策略运行过程中委托的报撤行为进行细粒度控制（降低交易滑点、实现高频策略）

    * [cta_backtester](https://www.github.com/vnpy/vnpy_ctabacktester)：CTA策略回测模块，无需使用Jupyter Notebook，直接使用图形界面进行策略回测分析、参数优化等相关工作

    * [spread_trading](https://www.github.com/vnpy/vnpy_spreadtrading)：价差交易模块，支持自定义价差，实时计算价差行情和持仓，支持价差算法交易以及自动价差策略两种模式

    * [option_master](https://www.github.com/vnpy/vnpy_optionmaster)：期权交易模块，针对国内期权市场设计，支持多种期权定价模型、隐含波动率曲面计算、希腊值风险跟踪等功能

    * [portfolio_strategy](https://www.github.com/vnpy/vnpy_portfoliostrategy)：组合策略模块，面向同时交易多合约的量化策略（Alpha、期权套利等），提供历史数据回测和实盘自动交易功能

    * [algo_trading](https://www.github.com/vnpy/vnpy_algotrading)：算法交易模块，提供多种常用的智能交易算法：TWAP、Sniper、Iceberg、BestLimit等

    * [script_trader](https://www.github.com/vnpy/vnpy_scripttrader)：脚本策略模块，面向多标的类量化策略和计算任务设计，同时也可以在命令行中实现REPL指令形式的交易，不支持回测功能

    * [paper_account](https://www.github.com/vnpy/vnpy_paperaccount)：本地仿真模块，纯本地化实现的仿真模拟交易功能，基于交易接口获取的实时行情进行委托撮合，提供委托成交推送以及持仓记录

    * [chart_wizard](https://www.github.com/vnpy/vnpy_chartwizard)：K线图表模块，基于RQData数据服务（期货）或者交易接口获取历史数据，并结合Tick推送显示实时行情变化

    * [portfolio_manager](https://www.github.com/vnpy/vnpy_portfoliomanager)：交易组合管理模块，以独立的策略交易组合（子账户）为基础，提供委托成交记录管理、交易仓位自动跟踪以及每日盈亏实时统计功能

    * [rpc_service](https://www.github.com/vnpy/vnpy_rpcservice)：RPC服务模块，允许将某一进程启动为服务端，作为统一的行情和交易路由通道，允许多客户端同时连接，实现多进程分布式系统

    * [data_manager](https://www.github.com/vnpy/vnpy_datamanager)：历史数据管理模块，通过树形目录查看数据库中已有的数据概况，选择任意时间段数据查看字段细节，支持CSV文件的数据导入和导出

    * [data_recorder](https://www.github.com/vnpy/vnpy_datarecorder)：行情记录模块，基于图形界面进行配置，根据需求实时录制Tick或者K线行情到数据库中，用于策略回测或者实盘初始化

    * [excel_rtd](https://www.github.com/vnpy/vnpy_excelrtd)：Excel RTD（Real Time Data）实时数据服务，基于pyxll模块实现在Excel中获取各类数据（行情、合约、持仓等）的实时推送更新

    * [risk_manager](https://www.github.com/vnpy/vnpy_riskmanager)：风险管理模块，提供包括交易流控、下单数量、活动委托、撤单总数等规则的统计和限制，有效实现前端风控功能

    * [web_trader](https://www.github.com/vnpy/vnpy_webtrader)：Web服务模块，针对B-S架构需求设计，实现了提供主动函数调用（REST）和被动数据推送（Websocket）的Web服务器

4. Python交易API接口封装（api），提供上述交易接口的底层对接实现。

    * REST Client（[rest](https://www.github.com/vnpy/vnpy_rest)）：基于协程异步IO的高性能REST API客户端，采用事件消息循环的编程模型，支持高并发实时交易请求发送

    * Websocket Client（[websocket](https://www.github.com/vnpy/vnpy_websocket)）：基于协程异步IO的高性能Websocket API客户端，支持和REST Client共用事件循环并发运行

5. 简洁易用的事件驱动引擎（event），作为事件驱动型交易程序的核心。

6. 对接各类数据库的适配器接口（database）：

    * SQL类

        * SQLite（[sqlite](https://www.github.com/vnpy/vnpy_sqlite)）：轻量级单文件数据库，无需安装和配置数据服务程序，VeighNa的默认选项，适合入门新手用户

        * MySQL（[mysql](https://www.github.com/vnpy/vnpy_mysql)）：主流的开源关系型数据库，文档资料极为丰富，且可替换其他NewSQL兼容实现（如TiDB）

        * PostgreSQL（[postgresql](https://www.github.com/vnpy/vnpy_postgresql)）：特性更为丰富的开源关系型数据库，支持通过扩展插件来新增功能，只推荐熟手使用

    * NoSQL类

        * DolphinDB（[dolphindb](https://www.github.com/vnpy/vnpy_dolphindb)）：一款高性能分布式时序数据库，适用于对速度要求极高的低延时或实时性任务

        * Arctic（[arctic](https://www.github.com/vnpy/vnpy_arctic)）：高性能金融时序数据库，采用了分块化储存、LZ4压缩等性能优化方案，以实现时序数据的高效读写

        * TDengine（[taos](https://www.github.com/vnpy/vnpy_taos)）：分布式、高性能、支持SQL的时序数据库，带有内建的缓存、流式计算、数据订阅等系统功能，能大幅减少研发和运维的复杂度

        * TimescaleDB（[timescaledb](https://www.github.com/vnpy/vnpy_timescaledb)）：基于PostgreSQL开发的一款时序数据库，以插件化扩展的形式安装，支持自动按空间和时间对数据进行分区

        * MongoDB（[mongodb](https://www.github.com/vnpy/vnpy_mongodb)）：基于分布式文件储存（bson格式）的文档式数据库，内置的热数据内存缓存提供更快读写速度

        * InfluxDB（[influxdb](https://www.github.com/vnpy/vnpy_influxdb)）：针对TimeSeries Data专门设计的时序数据库，列式数据储存提供极高的读写效率和外围分析应用

        * LevelDB（[leveldb](https://www.github.com/vnpy/vnpy_leveldb)）：由Google推出的高性能Key/Value数据库，基于LSM算法实现进程内存储引擎，支持数十亿级别的海量数据

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
