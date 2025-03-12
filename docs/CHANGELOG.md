# 0.1.1版本 (2025-03-11)

## 新增

1. 测试框架：添加全面的单元测试框架文件，包括：
   - 核心引擎测试：EventEngine、MainEngine、LogEngine测试
   - 数据组件测试：Datafeed、DatabaseManager测试
   - 交易组件测试：OrderManager、PositionManager、RiskManager测试
   - 网关接口测试：BaseGateway测试
   - 工具配置测试：Setting、Utility测试
   - 策略组件测试：CtaEngine、BacktestingEngine测试
   - 集成测试：数据到策略流程、多网关并行测试

# 0.1.0版本 (2025-03-10)

## 项目重塑

1. 项目立项：APilot量化交易框架，专注于加密货币交易
2. 精简架构：移除了所有UI和图形界面相关组件，改为纯命令行/API操作
3. 专注交易：框架核心功能专注于加密货币交易场景优化

## 新增

1. 加密货币支持：增强对各大加密货币交易所API的支持
2. 命令行工具：新增纯命令行交互界面，适合服务器部署环境
3. 通知系统：增加Telegram和Email通知配置，实时监控交易状态

## 调整

1. 项目配置：重构setup.cfg，更新项目元数据和依赖
2. 精简依赖：移除不必要的UI相关依赖（如PySide6、pyqtgraph等）
3. 优化设置：修改setting.py，提供更完善的加密货币交易相关配置
4. 更新常量：调整constant.py中的枚举定义，聚焦加密货币交易场景
5. 安装脚本：优化install_local.sh，提供更清晰的安装流程和反馈
6. 移除图片：删除不必要的图片文件，减小项目体积

## 修复

1. 日志级别：调整默认日志级别为INFO，提供更丰富的运行时信息
2. 配置文件：修复配置文件中的路径和参数问题

