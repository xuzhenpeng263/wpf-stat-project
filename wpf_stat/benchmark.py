# wpf_stat/benchmark.py

import os
import numpy as np
from tqdm import tqdm
from .core import batch_calculate_wpf # 注意导入的是 batch_calculate_wpf

BENCHMARK_FILE = 'wpf_benchmark_1_to_100000.npz'

def get_benchmark_dist(max_n: int = 1000000, force_recompute: bool = False):
    """
    Loads or computes the WPF entropy distribution for natural numbers using parallel processing.
    """
    if not os.path.exists(BENCHMARK_FILE) or force_recompute:
        print(f"Generating benchmark file for numbers up to {max_n} (using parallel processing)...")
        print("This is a one-time process and is now much faster.")
        
        number_range = range(2, max_n + 1)
        # 调用并行计算函数
        wpf_values = batch_calculate_wpf(list(number_range), parallel=True)
        
        np.savez_compressed(BENCHMARK_FILE, wpf_values=np.array(wpf_values))
        print(f"Benchmark file '{BENCHMARK_FILE}' created successfully.")

    data = np.load(BENCHMARK_FILE)
    return data['wpf_values']
