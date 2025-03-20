from typing import Dict, List, Callable, Tuple
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from random import random, choice
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from _collections_abc import dict_keys, dict_values, Iterable

from tqdm import tqdm
from deap import creator, base, tools, algorithms

# 类型定义
OUTPUT_FUNC = Callable[[str], None] # TODO：这里output可以统一改成logger
EVALUATE_FUNC = Callable[[dict], dict]
KEY_FUNC = Callable[[list], float]

# 适应度函数: 目标是最大化收益（weights=(1.0,) 表示单目标优化）
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 个体定义: 每个个体是一个参数列表，并包含适应度信息
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """ 这个类用于存储和生成优化参数空间 """

    def __init__(self) -> None:
        self.params: Dict[str, List] = {}
        self.target_name: str = ""

    def add_parameter(self, name: str, start: float, end: float = None, step: float = None) -> Tuple[bool, str]:
        try:
            if end is None or step is None:
                self.params[name] = [start]
                return True, "固定参数添加成功"
                
            # 简单地尝试生成参数列表
            self.params[name] = [start + i * step for i in range(int((end - start) / step) + 1)]
            return True, f"范围参数添加成功，数量 {len(self.params[name])}"
            
        except Exception as e:
            return False, f"参数错误: {e}"

    def set_target(self, target_name: str) -> None:
        self.target_name = target_name

    def generate_settings(self) -> List[dict]:
        """ 生成所有可能的参数组合 """
        return [dict(zip(self.params.keys(), values)) for values in product(*self.params.values())]


def check_optimization_setting(optimization_setting: OptimizationSetting) -> Tuple[bool, str]:
    """
    检查优化参数设置是否有效
    - 检查是否设置了目标函数
    - 检查是否有参数需要优化
    """
    if not optimization_setting.target_name:
        return False, "优化目标未设置"
    
    if not optimization_setting.params:
        return False, "没有添加任何优化参数"
    
    # 检查是否至少有一个参数是范围参数
    params_range = False
    for value in optimization_setting.params.values():
        if len(value) > 1:
            params_range = True
            break
    
    if not params_range:
        return False, "所有参数都是固定值，无法进行优化"
    
    return True, ""


def run_ga_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int = None,
    population_size: int = None,
    ngen_size: int = 30,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    output: OUTPUT_FUNC = print
) -> List[Tuple]:
    """ 使用遗传算法进行参数优化
    - evaluate_func: 适应度计算函数
    - optimization_setting: 需要优化的参数设置
    - key_func: 评估适应度的方式
    - max_workers: 并行计算的最大进程数
    - population_size: 每代的个体数
    - ngen_size: 进化的代数（迭代次数）
    - cxpb: 交叉概率
    - mutpb: 变异率

    遗传算法（GA）是一种基于自然选择和遗传进化的优化方法，它通过模拟生物进化的过程（选择、交叉、变异）来寻找最优解。

    遗传算法的核心概念
    种群（Population）：多个可能解（个体）的集合，每个个体代表一组参数。
    个体（Individual）：种群中的一个元素，它是一个可能的解。
    适应度（Fitness）：衡量一个个体的质量，适应度高的个体更可能被选中。
    选择（Selection）：根据适应度选择优秀个体。
    交叉（Crossover）：两个个体的部分基因交换，生成新的个体。
    变异（Mutation）：个体基因发生随机变化，以增加多样性，防止陷入局部最优。

    """

    settings = optimization_setting.generate_settings()
    total_size = len(settings)

    # 动态调整种群大小，确保不会超过搜索空间大小
    if population_size is None:
        population_size = min(500, total_size)  # 设定最大上限，避免过大影响计算

    def generate_parameter() -> list:
        """ 生成一个随机个体（即随机选取一组参数） """
        setting = choice(settings)
        # 从字典提取值并转换为列表
        values = list(setting.values())
        return values

    def mutate_individual(individual: list, indpb: float) -> tuple:
        """ 变异操作：随机选择部分参数进行调整 """
        for i in range(len(individual)):
            if random() < indpb:
                # 获取一个随机设置并提取对应位置的值
                random_values = list(choice(settings).values())
                if i < len(random_values):
                    individual[i] = random_values[i]
        return individual,

    # 设置多进程环境，使用 multiprocessing 并行计算
    ctx = get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(max_workers) as pool:
        cache = manager.dict()

        # 遗传算法工具箱
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)  # 双点交叉
        toolbox.register("mutate", mutate_individual, indpb=mutpb)  # 变异
        toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择
        toolbox.register("map", pool.map)  # 并行计算
        toolbox.register("evaluate", ga_evaluate, cache, evaluate_func, key_func, optimization_setting)  # 适应度计算

        # 进化参数
        mu = int(population_size * 0.8)  # 每代保留的优秀个体数
        lambda_ = population_size        # 每代生成的新个体数

        # 生成初始种群
        pop = toolbox.population(n=population_size)

        output(f"开始执行遗传算法优化 (参数空间大小: {total_size})")
        output(f"种群大小: {population_size}, 迭代次数: {ngen_size}")
        output(f"交叉率: {cxpb}, 变异率: {mutpb}")

        # 执行遗传算法（Mu+Lambda 选择策略）
        algorithms.eaMuPlusLambda(pop, toolbox, mu, lambda_, cxpb, mutpb, ngen_size, verbose=True)

        # 按照适应度排序，并返回最优结果
        results = list(cache.values())
        results.sort(reverse=True, key=key_func)
        return results


def ga_evaluate(
    cache: dict,
    evaluate_func: EVALUATE_FUNC,
    key_func: KEY_FUNC,
    optimization_setting: OptimizationSetting,
    parameters: list
) -> float:
    """ 计算适应度（fitness）
    - cache: 共享缓存，避免重复计算
    - evaluate_func: 计算适应度的函数
    - key_func: 适应度评估方式
    - optimization_setting: 优化参数设置
    - parameters: 个体参数
    """
    param_tuple = tuple(parameters)  # 将参数转换为 tuple 作为 key
    if param_tuple in cache:
        result = cache[param_tuple]  # 如果已计算过，则直接从缓存中获取
    else:
        # 使用生成的参数创建设置字典
        param_names = list(optimization_setting.params.keys())
        setting = {}
        
        # 确保参数数量一致
        if len(param_names) == len(parameters):
            for i, name in enumerate(param_names):
                setting[name] = parameters[i]
        else:
            # 参数数量不匹配时的处理
            print(f"警告: 参数数量不匹配! 预期 {len(param_names)}, 实际 {len(parameters)}")
            print(f"参数名: {param_names}")
            print(f"参数值: {parameters}")
            # 使用默认参数字典
            setting = {k: 20 if "period" in k else 0.01 for k in param_names}
        
        result = evaluate_func(setting)  # 计算适应度
        cache[param_tuple] = result  # 存入缓存

    return key_func(result),
