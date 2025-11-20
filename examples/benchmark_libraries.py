# examples/benchmark_libraries.py

import sys
import os
import random
import secrets
import numpy as np

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wpf_stat.analyzer import WPFAnalyzer

# --- 选手4: 自定义一个简单的LCG ---
class SimpleLCG:
    """一个简单的线性同余生成器 (参数来自 C++11 minstd_rand)"""
    def __init__(self, seed=1):
        self.state = seed
        self.a = 48271
        self.m = 2147483647  # 2^31 - 1

    def next(self):
        self.state = (self.a * self.state) % self.m
        return self.state

    def randint(self, low, high):
        # 将生成的数映射到[low, high]范围
        return low + self.next() % (high - low + 1)

def run_benchmark():
    """执行完整的随机数库基准测试"""
    NUM_SAMPLES = 5000
    MAX_VAL = 1_000_000
    
    print(f"--- Starting Benchmark ---")
    print(f"Samples per generator: {NUM_SAMPLES}")
    print(f"Integer range: [2, {MAX_VAL}]")
    
    # --- 生成序列 ---
    print("\nGenerating sequences from all contestants...")
    
    # 选手1: random
    seq_random = [random.randint(2, MAX_VAL) for _ in range(NUM_SAMPLES)]
    
    # 选手2: secrets
    seq_secrets = [secrets.randbelow(MAX_VAL - 1) + 2 for _ in range(NUM_SAMPLES)]
    
    # 选手3: numpy.random
    # 使用新的 Generator API
    rng = np.random.default_rng()
    seq_numpy = rng.integers(low=2, high=MAX_VAL, size=NUM_SAMPLES, endpoint=True).tolist()
    
    # 选手4: LCG
    lcg = SimpleLCG()
    seq_lcg = [lcg.randint(2, MAX_VAL) for _ in range(NUM_SAMPLES)]

    sequences = {
        "1_Standard_Random": seq_random,
        "2_Crypto_Secrets": seq_secrets,
        "3_Scientific_NumPy": seq_numpy,
        "4_Simple_LCG": seq_lcg
    }
    
    # --- 运行分析 ---
    for name, seq in sequences.items():
        print(f"\n{'='*20} TESTING: {name} {'='*20}")
        analyzer = WPFAnalyzer(seq)
        analyzer.run_all_tests()
        plot_filename = os.path.join('plots', f'benchmark_{name}.png')
        analyzer.plot_distributions(filename=plot_filename)

if __name__ == '__main__':
    if not os.path.exists('wpf_stat'):
        print("Error: Please run this script from the 'wpf-stat-project' root directory.")
        sys.exit(1)
    
    run_benchmark()
