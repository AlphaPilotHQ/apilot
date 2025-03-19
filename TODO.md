# APilot 开发计划

本文档记录APilot项目的未来开发计划和优化点。

## 重要TODO：
- [ ] 完全移除CTA引擎中已弃用的write_log方法
- [ ] 更新所有策略代码，使用主引擎的log_xxx方法替代write_log
- [ ] 为回测引擎添加与主引擎一致的log_debug/log_info等方法，统一API
- upload to pypi


## 可选TODO
- [ ] 添加更多单元测试
- [ ] 添加更多策略示例
- test riskmanager
- test algotrading & websocket
- test optimizer


## 未来TODO
- [ ] 提供Docker部署方案
- [ ] 支持更多数据源
