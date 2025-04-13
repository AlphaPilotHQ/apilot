"""
Grid Search Optimization Module

Provides grid search parameter optimization for strategy parameters.
"""

import logging
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from .settings import OptimizationSetting

# Type definitions
KEY_FUNC = Callable[[dict[str, Any]], float]

# Get logger
logger = logging.getLogger("Optimizer")


def run_grid_search(
    strategy_class: type[Any],
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int | None = None,
) -> list[dict]:
    """
    Perform parameter optimization using grid search

    Args:
        strategy_class: Strategy class
        optimization_setting: Parameter settings to optimize
        key_func: Fitness evaluation function that receives parameter settings and returns fitness value (higher is better)
        max_workers: Maximum number of parallel processes (currently unused, kept for interface compatibility)

    Returns:
        List of parameter combinations sorted by fitness
    """
    # Validate optimization parameters
    valid, msg = optimization_setting.check_setting()
    if not valid:
        logger.error(f"Invalid optimization parameters: {msg}")
        return []

    # Generate all parameter combinations
    settings = optimization_setting.generate_setting()
    total_combinations = len(settings)

    # Log output
    logger.info(
        f"Starting grid search optimization (parameter space size: {total_combinations})"
    )
    logger.info(
        f"Optimization target: {optimization_setting.target_name or 'Not specified'}"
    )

    # Calculate fitness for each parameter combination
    results = []
    progress_bar = tqdm(settings, desc="Parameter optimization progress")

    for i, setting in enumerate(progress_bar):
        try:
            # Evaluate current parameter combination
            fitness = key_func(setting)

            # Skip invalid results
            if (
                fitness is None
                or not isinstance(fitness, int | float)
                or abs(fitness) > 1e6
            ):
                continue

            # Save valid result
            result = setting.copy()
            result["fitness"] = fitness
            results.append(result)

            # Record progress
            if (i + 1) % 5 == 0 or i == 0 or i == len(settings) - 1:
                logger.debug(f"Evaluated {i + 1}/{total_combinations} parameter sets")

        except Exception as e:
            logger.warning(f"Failed to evaluate parameters {setting}: {e!s}")

    # Sort by fitness (descending)
    if results:
        results.sort(key=lambda x: x["fitness"], reverse=True)

        # Display best results
        top_n = min(5, len(results))
        logger.info(f"Grid search completed, found {len(results)} valid results")
        for i in range(top_n):
            result = results[i]
            fitness = result.pop(
                "fitness"
            )  # Temporarily remove fitness for parameter printing
            logger.info(f"Rank {i + 1}: Fitness={fitness:.4f}, Parameters={result}")
            result["fitness"] = fitness  # Put fitness back
    else:
        logger.warning("Grid search did not find any valid results")

    return results
