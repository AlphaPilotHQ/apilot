# APilot 开发计划

本文档记录APilot项目的未来开发计划和优化点。

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



## 未来TODO
- 修改email成telegram
- 图表系统拆出来重新做
- 提供Docker部署方案
- 支持更多数据源


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
│   │   │   └── backtesting.py    # CTA策略回测（如有需要）
│   │   └── spread                # 价差策略
│   │       ├── base.py
│   │       ├── algo.py
│   │       ├── engine.py
│   │       └── backtesting.py    # 价差策略回测（如有需要）
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
│   ├── utils                     # 通用工具/常量/对象
│   │   ├── constant.py
│   │   ├── object.py
│   │   ├── setting.py
│   │   └── utility.py
│   └── __main__.py               # (可选) 如果需要入口脚本
│
├── docs                          # 文档
├── examples                      # 示例
│   ├── cta_backtesting
│   └── spread_backtesting
├── tests                         # 测试
├── scripts                       # (可选) 运维、部署脚本等
├── requirements.txt              # 依赖
├── README.md
└── setup.py (或 pyproject.toml) # 若要打包/发布到PyPI

