# wpf_stat/suite.py

import numpy as np
from scipy.stats import kstest, chisquare
from .core import batch_calculate_wpf
from .benchmark import get_benchmark_dist, get_benchmark_spectrum
from typing import List, Tuple

class TestResult:
    """A simple class to hold test results."""
    def __init__(self, test_name: str, statistic: float, p_value: float, passed: bool, details: str):
        self.test_name = test_name
        self.statistic = statistic
        self.p_value = p_value
        self.passed = passed
        self.details = details

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        return (f"[{status}] {self.test_name}\n"
                f"    P-value: {self.p_value:.4f}, Statistic: {self.statistic:.4f}\n"
                f"    Details: {self.details}")

def run_test_suite(sequence: List[int], alpha: float = 0.01):
    """
    Runs all WPF-Stat tests on a given sequence of integers.
    
    Args:
        sequence: A list of integers to be tested for randomness.
        alpha: The significance level for the statistical tests.
        
    Returns:
        A list of TestResult objects.
    """
    print("--- Running WPF-Stat Randomness Test Suite ---")
    if not isinstance(sequence, list) or not all(isinstance(x, int) for x in sequence):
        raise TypeError("Input must be a list of integers.")
    
    # Load the "golden standard" distribution
    benchmark_dist = get_benchmark_dist()
    
    # Calculate WPF for the input sequence
    print(f"Calculating WPF entropy for {len(sequence)} numbers...")
    sequence_wpf = np.array(batch_calculate_wpf(sequence))

    results = [
        test_entropy_distribution(sequence_wpf, benchmark_dist, alpha),
        test_micro_structure(sequence_wpf, benchmark_dist, alpha),
        test_autocorrelation(sequence_wpf, alpha)
    ]
    print("--- Test Suite Finished ---")
    return results

def test_entropy_distribution(sequence_wpf: np.ndarray, benchmark_dist: np.ndarray, alpha: float) -> TestResult:
    """
    Test 1: Compares the overall WPF distribution against the natural numbers' distribution.
    Uses a two-sample Kolmogorov-Smirnov (K-S) test.
    
    Null Hypothesis: The sample distribution is the same as the benchmark distribution.
    """
    test_name = "Entropy Distribution Test (K-S)"
    
    # The K-S test is sensitive to sample size, let's take a random sample
    # from the benchmark distribution of the same size as our sequence for a fair comparison.
    if len(sequence_wpf) > len(benchmark_dist):
        raise ValueError("Sequence size cannot be larger than benchmark size.")
        
    benchmark_sample = np.random.choice(benchmark_dist, size=len(sequence_wpf), replace=False)
    
    statistic, p_value = kstest(sequence_wpf, benchmark_sample)
    
    passed = p_value >= alpha
    details = f"Compares the ECDF of sequence's entropy against natural numbers. Low p-value suggests a different structural distribution."
    
    return TestResult(test_name, statistic, p_value, passed, details)

def test_micro_structure(sequence_wpf: np.ndarray, benchmark_dist: np.ndarray, alpha: float) -> TestResult:
    """
    Test 2: Checks if the proportion of 'simple' (zero-entropy) vs 'complex' (non-zero) numbers is natural.
    Uses a Chi-Squared test on frequency counts.
    """
    test_name = "Micro-Structure Test (Chi-Squared)"
    
    # 1. Get theoretical probabilities from the benchmark
    zero_prob_expected, non_zero_probs_expected, bin_edges = get_benchmark_spectrum(benchmark_dist)
    
    # 2. Calculate observed frequencies from the sequence
    total_obs_count = len(sequence_wpf)
    zero_obs_count = np.sum(sequence_wpf == 0)
    
    non_zero_obs = sequence_wpf[sequence_wpf > 0]
    non_zero_obs_counts, _ = np.histogram(non_zero_obs, bins=bin_edges)
    
    # Form the observed and expected frequency arrays for the test
    observed_freqs = np.append([zero_obs_count], non_zero_obs_counts)
    expected_freqs = total_obs_count * np.append([zero_prob_expected], non_zero_probs_expected)

    # Avoid bins with zero expected frequency, which can cause issues
    non_zero_mask = expected_freqs > 0
    observed_freqs = observed_freqs[non_zero_mask]
    expected_freqs = expected_freqs[non_zero_mask]

    statistic, p_value = chisquare(f_obs=observed_freqs, f_exp=expected_freqs)
    
    passed = p_value >= alpha
    details = "Checks if the proportion of primes vs. composites (in entropy bins) is natural. Low p-value indicates structural bias."
    
    return TestResult(test_name, statistic, p_value, passed, details)

def test_autocorrelation(sequence_wpf: np.ndarray, alpha: float) -> TestResult:
    """
    Test 3: Checks if the entropy of a number is correlated with the next one.
    Calculates the lag-1 autocorrelation coefficient.
    """
    test_name = "Entropy Autocorrelation Test"
    
    if len(sequence_wpf) < 2:
        return TestResult(test_name, 0.0, 1.0, True, "Sequence too short for autocorrelation test.")
        
    # Pearson correlation coefficient between the sequence and its shifted version
    corr_matrix = np.corrcoef(sequence_wpf[:-1], sequence_wpf[1:])
    statistic = corr_matrix[0, 1]
    
    # A simple significance test for the correlation coefficient
    n = len(sequence_wpf)
    # Using a t-test approximation for the p-value
    t_stat = statistic * np.sqrt((n - 2) / (1 - statistic**2))
    p_value = 2 * (1 - abs(np.random.standard_t(n-2, 1)[0] > abs(t_stat))) # Approximation
    # This p-value is a rough estimate. The statistic itself is more informative.
    
    # For this test, we check if the correlation is significantly different from zero.
    passed = abs(statistic) < 2 / np.sqrt(n) # A common rule of thumb
    
    details = f"Checks if a number's structural complexity predicts the next. Correlation should be near zero. Coeff: {statistic:.4f}"
    
    return TestResult(test_name, abs(statistic), p_value, passed, details)
