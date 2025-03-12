"""
指数退避重试工具
这个模块提供了指数退避重试策略的实现，可以用于网络请求、数据库访问等可能因临时问题而失败的操作。
"""

import logging
import random
from functools import wraps
from time import sleep
from typing import Callable, TypeVar, Optional, Any


# 定义泛型返回类型
T = TypeVar('T')


def exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions_to_retry: tuple = (Exception,),
    should_retry_func: Optional[Callable[[Exception], bool]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    指数退避重试装饰器。
    
    用法:
        @exponential_backoff(max_retries=3)
        def function_that_might_fail():
            # 可能会失败的代码
    
    参数:
        max_retries: int
            最大重试次数
        initial_delay: float
            初始延迟时间(秒)
        max_delay: float
            最大延迟时间(秒)
        backoff_factor: float
            退避因子，每次重试后延迟时间将乘以此因子
        jitter: bool
            是否添加随机抖动以避免多个客户端同时重试
        exceptions_to_retry: tuple
            需要重试的异常类型
        should_retry_func: Optional[Callable[[Exception], bool]]
            自定义函数，判断某个异常是否应该重试
        
    返回:
        装饰器函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retry_count = 0
            delay = initial_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions_to_retry as e:
                    # 判断是否应该重试
                    if should_retry_func and not should_retry_func(e):
                        logging.error(f"不满足重试条件，直接抛出异常: [{type(e).__name__}]: {e}")
                        raise
                    
                    retry_count += 1
                    if retry_count > max_retries:
                        logging.error(f"已达到最大重试次数({max_retries})，放弃重试: [{type(e).__name__}]: {e}")
                        raise
                    
                    # 计算下一次延迟时间
                    if jitter:
                        # 添加0-30%的随机抖动
                        jitter_factor = 1 + random.uniform(0, 0.3)
                        current_delay = min(delay * jitter_factor, max_delay)
                    else:
                        current_delay = min(delay, max_delay)
                    
                    logging.warning(f"操作失败({retry_count}/{max_retries})，将在{current_delay:.2f}秒后重试: [{type(e).__name__}]: {e}")
                    sleep(current_delay)
                    
                    # 计算下一次延迟
                    delay = min(delay * backoff_factor, max_delay)
                    
        return wrapper
    return decorator