# examples/run_tests_v2.py

import sys
import os
import random
import numpy as np
from sympy import sieve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wpf_stat.analyzer import WPFAnalyzer

def generate_sequences(num_samples=2000, max_val=100000):
    """(代码与之前版本相同，此处为简洁省略)"""
    print(f"\nGenerating {num_samples} test numbers up to {max_val}...")
    try:
        from secrets import randbelow
        good_sequence = [randbelow(max_val-2) + 2 for _ in range(num_samples)]
    except ImportError:
        good_sequence = np.random.randint(2, max_val, num_samples).tolist()
    
    standard_sequence = [random.randint(2, max_val) for _ in range(num_samples)]
    
    primes = list(sieve.primerange(2, max_val))
    bad_sequence_biased = random.choices(primes, k=num_samples * 3 // 4) + \
                          [random.randint(2, max_val) for _ in range(num_samples // 4)]
    random.shuffle(bad_sequence_biased)
    
    bad_sequence_correlated = []
    current_is_prime = True
    for _ in range(num_samples):
        # 90% chance to stay in the same state (prime or composite)
        if random.random() < 0.9:
            pass
        else:
            current_is_prime = not current_is_prime
        
        if current_is_prime:
            bad_sequence_correlated.append(random.choice(primes))
        else:
            num = random.randint(2, max_val)
            while num in primes:
                num = random.randint(2, max_val)
            bad_sequence_correlated.append(num)

    return {
        "1_Crypto_Secure": good_sequence,
        "2_Standard_PRNG": standard_sequence,
        "3_Bad_PRNG_Biased": bad_sequence_biased,
        "4_Bad_PRNG_Correlated": bad_sequence_correlated
    }

if __name__ == '__main__':
    # Make sure we run this from the project root for path consistency
    if not os.path.exists('wpf_stat'):
        print("Error: Please run this script from the 'wpf-stat-project' root directory.")
        sys.exit(1)

    sequences_to_test = generate_sequences()
    
    for name, seq in sequences_to_test.items():
        # 1. Create an analyzer instance
        analyzer = WPFAnalyzer(seq)
        
        # 2. Run all tests
        analyzer.run_all_tests()
        
        # 3. Generate and save the visualization
        plot_filename = os.path.join('plots', f'distribution_{name}.png')
        analyzer.plot_distributions(filename=plot_filename)
