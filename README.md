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

## 策略类型

- **PA 策略**：支持趋势跟踪、均值回归等经典价格行为策略
- **因子策略**：基于多因子模型的量化策略 (开发中)


## 技术架构

### 设计原则
- **Core 目录**：包含系统所有抽象接口和核心数据结构
  - 抽象基类 (BaseEngine, BaseGateway 等)
  - 数据模型 (OrderData, TickData 等)
  - 常量定义 (Direction, Interval 等)
  - 基础事件系统

- **功能目录**：包含各领域的具体实现
  - gateway/ - 各交易所接口实现
  - engine/ - 各种引擎的具体实现

## Gateway


## 许可证

本项目采用 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件
