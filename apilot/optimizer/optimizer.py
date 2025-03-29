from collections.abc import Callable
from itertools import product
from multiprocessing import get_context
from random import choice, random
from typing import Any

try:
    from deap import algorithms, base, creator, tools
except ImportError as err:
    # 安装提示
    raise ImportError(
        "需要安装deap库才能使用遗传算法优化.请运行: pip install deap"
    ) from err

# 类型定义
KEY_FUNC = Callable[[dict[str, Any]], float]


# 适应度函数: 目标是最大化收益(weights=(1.0,) 表示单目标优化)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 个体定义: 每个个体是一个参数列表,并包含适应度信息
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """存储和生成优化参数空间的类"""

    def __init__(self) -> None:
        self.params: dict[str, list] = {}
        self.target_name: str = ""

    def add_parameter(
        self,
        name: str,
        start: float,
        end: float | None = None,
        step: float | None = None,
    ) -> tuple[bool, str]:
        """添加优化参数

        可以添加两种类型的参数:
        1. 离散型: 直接添加具体数值列表 [1, 2, 3, 4, 5]
        2. 连续型: 给定 start/end/step, 生成均匀分布的数值
        """
        try:
            if end is None or step is None:
                if isinstance(start, list):
                    self.params[name] = start
                else:
                    self.params[name] = [start]
            else:
                value = start
                value_list = []

                while value <= end:
                    value_list.append(value)
                    value += step

                self.params[name] = value_list

            return True, ""
        except Exception as e:
            return False, str(e)

    def set_target(self, target_name: str) -> None:
        """设置优化的目标名称"""
        self.target_name = target_name

    def generate_setting(self) -> list[dict]:
        """生成所有参数组合"""
        keys = self.params.keys()
        values = self.params.values()
        products = list(product(*values))

        settings = []
        for p in products:
            setting = dict(zip(keys, p, strict=False))
            settings.append(setting)

        return settings

    def check_setting(self) -> tuple[bool, str]:
        """检查优化参数是否有效"""
        if not self.params:
            return False, "优化参数为空"

        params_range = False
        for value in self.params.values():
            if len(value) > 1:
                params_range = True
                break

        if not params_range:
            return False, "所有参数都是固定值,无法进行优化"

        return True, ""


def run_ga_optimization(
    strategy_class: type[Any],
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int | None = None,
    population_size: int | None = None,
    ngen_size: int = 30,
    cxpb: float = 0.7,
    mutpb: float = 0.3,
) -> list[dict]:
    """使用遗传算法进行参数优化

    Args:
        strategy_class: 策略类
        optimization_setting: 需要优化的参数设置
        key_func: 评估适应度的方式
        max_workers: 并行计算的最大进程数
        population_size: 每代的个体数
        ngen_size: 进化的代数(迭代次数)
        cxpb: 交叉概率
        mutpb: 变异率

    Returns:
        最优的参数组合列表
    """
    # 验证优化参数
    valid, msg = optimization_setting.check_setting()
    if not valid:
        print(f"优化参数无效: {msg}")
        return []

    # 获取参数空间
    params = optimization_setting.params
    param_names = list(params.keys())
    param_values = list(params.values())
    total_size = 1
    for v in param_values:
        total_size *= len(v)

    # 确定种群大小
    if population_size is None:
        population_size = min(total_size, 100)

    # 进程池 - 用于并行计算
    if max_workers is None:
        # 默认使用CPU核心数
        max_workers = 8

    with get_context("spawn").Pool(max_workers) as pool:
        # 创建工具箱
        toolbox = base.Toolbox()
        toolbox.register("map", pool.map)

        # 初始化参数缓存
        cache: dict[tuple, dict] = {}

        # 注册个体生成器
        toolbox.register(
            "individual",
            generate_parameter,
            creator.Individual,
            param_values,
        )
        # 注册种群生成器
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
        )
        # 注册适应度计算函数
        toolbox.register(
            "evaluate",
            ga_evaluate,
            cache,
            strategy_class,
            key_func,
            optimization_setting,
        )
        # 选择操作 - 锦标赛选择法
        toolbox.register("select", tools.selTournament, tournsize=3)
        # 交叉操作 - 两点交叉
        toolbox.register("mate", tools.cxTwoPoint)
        # 变异操作 - 均匀变异
        toolbox.register("mutate", mutate_individual, param_values=param_values)

        # 生成初始种群
        pop = toolbox.population(n=population_size)

        print(f"开始执行遗传算法优化 (参数空间大小: {total_size})")
        print(f"种群大小: {population_size}, 迭代次数: {ngen_size}")
        print(f"交叉率: {cxpb}, 变异率: {mutpb}")

        # 执行遗传算法(Mu+Lambda 选择策略)
        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=population_size,
            lambda_=population_size,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen_size,
            stats=None,
            halloffame=None,
            verbose=True,
        )

        # 根据适应度排序
        pop.sort(key=lambda x: x.fitness.values[0], reverse=True)

        # 返回参数和适应度
        result = []
        for ind in pop[: min(10, len(pop))]:
            setting = dict(zip(param_names, ind, strict=False))
            fitness = ind.fitness.values[0]
            setting["fitness"] = fitness
            result.append(setting)

        return result


def ga_evaluate(
    cache: dict,
    strategy_class: type[Any],
    key_func: KEY_FUNC,
    optimization_setting: OptimizationSetting,
    parameters: list,
) -> tuple:
    """计算适应度(fitness)

    Args:
        cache: 共享缓存,避免重复计算
        strategy_class: 策略类
        key_func: 适应度评估方式
        optimization_setting: 优化参数设置
        parameters: 个体参数

    Returns:
        适应度评分
    """
    # 转为元组当作缓存key
    param_tuple = tuple(parameters)
    param_names = list(optimization_setting.params.keys())

    # 如果已在缓存中,直接返回
    if param_tuple in cache:
        return (cache[param_tuple],)

    # 构建策略参数
    if param_names:
        setting = dict(zip(param_names, param_tuple, strict=False))
    else:
        # 使用默认参数字典
        setting = {k: 20 if "period" in k else 0.01 for k in param_names}

    try:
        # 计算策略适应度
        result = key_func(strategy_class(setting))
        cache[param_tuple] = result
    except Exception:
        # 出错时返回极低的适应度,保证不会被选中
        return (-999999.0,)

    return (result,)


def generate_parameter(
    container,
    param_values: list,
) -> list:
    """生成一个个体(参数组合)"""
    ind = container()
    for values in param_values:
        value = choice(values)
        ind.append(value)
    return ind


def mutate_individual(
    individual: list, param_values: list, indpb: float = 0.1
) -> tuple:
    """变异操作 - 按照概率随机替换参数值"""
    size = len(individual)
    for i in range(size):
        # 按照indpb概率决定是否变异
        if random() < indpb:
            individual[i] = choice(param_values[i])

    return (individual,)
