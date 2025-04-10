"""
网格搜索优化模块

提供网格搜索参数优化功能，用于策略参数优化。
"""

import logging
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from .settings import OptimizationSetting

# 类型定义
KEY_FUNC = Callable[[dict[str, Any]], float]

# 获取日志记录器
logger = logging.getLogger("Optimizer")


def run_grid_search(
    strategy_class: type[Any],
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int | None = None,
) -> list[dict]:
    """
    使用网格搜索进行参数优化

    Args:
        strategy_class: 策略类
        optimization_setting: 需要优化的参数设置
        key_func: 适应度评估函数，接收参数设置并返回适应度值（越大越好）
        max_workers: 并行计算的最大进程数（目前未使用，保留接口兼容性）

    Returns:
        按适应度排序的参数组合列表
    """
    # 验证优化参数
    valid, msg = optimization_setting.check_setting()
    if not valid:
        logger.error(f"优化参数无效: {msg}")
        return []

    # 生成所有参数组合
    settings = optimization_setting.generate_setting()
    total_combinations = len(settings)

    # 日志输出
    logger.info(f"开始网格搜索优化 (参数空间大小: {total_combinations})")
    logger.info(f"优化目标: {optimization_setting.target_name or '未指定'}")

    # 计算每个参数组合的适应度
    results = []
    progress_bar = tqdm(settings, desc="参数优化进度")

    for i, setting in enumerate(progress_bar):
        try:
            # 评估当前参数组合
            fitness = key_func(setting)

            # 跳过无效结果
            if (
                fitness is None
                or not isinstance(fitness, int | float)
                or abs(fitness) > 1e6
            ):
                continue

            # 保存有效结果
            result = setting.copy()
            result["fitness"] = fitness
            results.append(result)

            # 记录进度
            if (i + 1) % 5 == 0 or i == 0 or i == len(settings) - 1:
                logger.debug(f"已评估 {i + 1}/{total_combinations} 组参数")

        except Exception as e:
            logger.warning(f"评估参数 {setting} 失败: {e!s}")

    # 按适应度排序（降序）
    if results:
        results.sort(key=lambda x: x["fitness"], reverse=True)

        # 显示最佳结果
        top_n = min(5, len(results))
        logger.info(f"网格搜索完成,共找到 {len(results)} 个有效结果")
        for i in range(top_n):
            result = results[i]
            fitness = result.pop("fitness")  # 临时移除fitness以便打印参数
            logger.info(f"排名 {i + 1}: 适应度={fitness:.4f}, 参数={result}")
            result["fitness"] = fitness  # 放回fitness
    else:
        logger.warning("网格搜索未找到有效结果")

    return results
