# wpf_stat/core.py

import math
from sympy.ntheory import factorint
from typing import List
from multiprocessing import Pool, cpu_count

def calculate_wpf(n: int) -> float:
    """
    Calculates the Weighted Prime Factorization Entropy (WPF) H(n).
    (代码与之前版本相同，此处为简洁省略)
    """
    if n <= 1:
        return 0.0
    factors = factorint(n)
    if len(factors) == 1:
        return 0.0
    primes = list(factors.keys())
    exponents = list(factors.values())
    m = len(primes)
    A = sum(exponents)
    h_struct = -sum((k / A) * math.log2(k / A) for k in exponents)
    phi_order = 1 + math.log2(A)
    w_weight = sum(math.log2(p) for p in primes) / m
    return h_struct * phi_order * w_weight

def batch_calculate_wpf(numbers: List[int], parallel: bool = False) -> List[float]:
    """
    Calculates WPF entropy for a list of numbers, with optional parallel processing.
    
    Args:
        numbers: A list of integers.
        parallel: If True, uses all available CPU cores to speed up calculation.
        
    Returns:
        A list of corresponding WPF entropy values.
    """
    if not parallel:
        return [calculate_wpf(n) for n in numbers]
    else:
        # 使用所有可用的CPU核心进行并行计算
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(calculate_wpf, numbers)
        return results
