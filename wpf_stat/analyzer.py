# wpf_stat/analyzer.py

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import List

from .core import batch_calculate_wpf
from .benchmark import get_benchmark_dist

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
        return f"[{status}] {self.test_name:<35} P-value: {self.p_value:<8.4f} Statistic: {self.statistic:<8.4f}\n    Details: {self.details}"

class WPFAnalyzer:
    """
    Analyzes a sequence of integers for number-theoretic randomness using WPF entropy.
    """
    def __init__(self, sequence: List[int]):
        if not isinstance(sequence, list) or not all(isinstance(x, int) for x in sequence):
            raise TypeError("Input must be a list of integers.")
        
        self.sequence = sequence
        print(f"\n--- Analyzing sequence of {len(self.sequence)} numbers ---")
        self.benchmark_dist = get_benchmark_dist()
        
        print("Calculating WPF entropy for the sequence (using parallel processing)...")
        self.sequence_wpf = np.array(batch_calculate_wpf(self.sequence, parallel=True))
        self.results = []

    def test_entropy_distribution(self, alpha: float):
        """Test 1: K-S test for overall distribution similarity."""
        benchmark_sample = np.random.choice(self.benchmark_dist, size=len(self.sequence_wpf), replace=False)
        statistic, p_value = stats.kstest(self.sequence_wpf, benchmark_sample)
        details = "Compares the sample's entropy distribution against the natural numbers' distribution."
        self.results.append(TestResult("Distribution Similarity (K-S)", statistic, p_value, p_value >= alpha, details))

    def test_micro_structure(self, alpha: float):
        """Test 2: Chi-Squared test for structural proportions (primes vs composites)."""
        num_bins = 20
        total_obs_count = len(self.sequence_wpf)

        # --- FINAL FIX ---

        # 1. Define bins based on the benchmark distribution range
        non_zero_benchmark = self.benchmark_dist[self.benchmark_dist > 0]
        # Use quantiles for binning to ensure each bin has a reasonable number of expected samples
        bin_edges = np.quantile(non_zero_benchmark, np.linspace(0, 1, num_bins + 1))
        # Ensure bin edges are unique
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2: # handle case where all data is the same
            self.results.append(TestResult("Micro-Structure (Chi-Squared)", 0.0, 1.0, True, "Not enough data variability for Chi-Squared test."))
            return
        
        # 2. Calculate OBSERVED frequencies for the test sequence
        obs_zeros_count = np.sum(self.sequence_wpf == 0)
        obs_non_zeros_hist, _ = np.histogram(self.sequence_wpf[self.sequence_wpf > 0], bins=bin_edges)
        observed_freqs = np.append([obs_zeros_count], obs_non_zeros_hist)

        # 3. Calculate EXPECTED frequencies for a sequence of the same size
        bench_zeros_count = np.sum(self.benchmark_dist == 0)
        bench_non_zeros_hist, _ = np.histogram(non_zero_benchmark, bins=bin_edges)
        
        # Calculate theoretical probabilities from the full benchmark data
        benchmark_total_count = len(self.benchmark_dist)
        expected_probs = np.append([bench_zeros_count], bench_non_zeros_hist) / benchmark_total_count
        
        # Now, calculate expected frequencies for our sample size
        expected_freqs = expected_probs * total_obs_count

        # 4. Handle cases where expected frequency is too low (standard practice for Chi-Squared)
        # We merge bins with low expected counts to ensure test validity.
        merged_obs = []
        merged_exp = []
        temp_obs = 0
        temp_exp = 0
        for i in range(len(expected_freqs)):
            temp_obs += observed_freqs[i]
            temp_exp += expected_freqs[i]
            if temp_exp >= 5: # Merge until the expected count is at least 5
                merged_obs.append(temp_obs)
                merged_exp.append(temp_exp)
                temp_obs = 0
                temp_exp = 0
        # Add any remaining merged group
        if temp_exp > 0:
            merged_obs[-1] += temp_obs
            merged_exp[-1] += temp_exp
        
        final_obs = np.array(merged_obs)
        final_exp = np.array(merged_exp)

        # Final check to ensure sums match after merging and potential floating point issues
        final_exp[-1] = np.sum(final_obs) - np.sum(final_exp[:-1])

        if len(final_obs) < 2:
            self.results.append(TestResult("Micro-Structure (Chi-Squared)", 0.0, 1.0, True, "Not enough categories after merging for Chi-Squared test."))
            return

        statistic, p_value = stats.chisquare(f_obs=final_obs, f_exp=final_exp)
        
        details = "Checks if the ratio of simple (primes) vs. complex (composites) numbers is natural."
        self.results.append(TestResult("Micro-Structure (Chi-Squared)", statistic, p_value, p_value >= alpha, details))

    def test_median_runs(self, alpha: float):
        """[NEW] Test 3: Wald-Wolfowitz Runs Test for entropy clustering."""
        median = np.median(self.sequence_wpf)
        runs_sequence = np.where(self.sequence_wpf > median, 1, np.where(self.sequence_wpf < median, -1, 0))
        runs_sequence = runs_sequence[runs_sequence != 0]

        if len(runs_sequence) < 20: # Test requires a reasonable number of samples
            self.results.append(TestResult("Entropy Clustering (Runs Test)", 0.0, 1.0, True, "Sequence too short for a reliable runs test."))
            return

        n1 = np.sum(runs_sequence == 1)
        n2 = np.sum(runs_sequence == -1)
        
        if n1 == 0 or n2 == 0:
            self.results.append(TestResult("Entropy Clustering (Runs Test)", 0.0, 1.0, True, "Sequence lacks variability for this test."))
            return

        runs = np.sum(np.abs(np.diff(runs_sequence))) // 2 + 1
        
        n = n1 + n2
        mean_runs = 2 * n1 * n2 / n + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        
        if var_runs <= 0:
            self.results.append(TestResult("Entropy Clustering (Runs Test)", 0.0, 1.0, True, "Cannot compute variance for runs test."))
            return

        z_score = (runs - mean_runs) / np.sqrt(var_runs)
        p_value = 2 * stats.norm.sf(abs(z_score))

        details = "Detects clustering of high/low complexity numbers. Too few runs suggest correlation."
        self.results.append(TestResult("Entropy Clustering (Runs Test)", z_score, p_value, p_value >= alpha, details))
        
    def run_all_tests(self, alpha: float = 0.01):
        """Runs the complete test suite."""
        self.results = []
        self.test_entropy_distribution(alpha)
        self.test_micro_structure(alpha)
        self.test_median_runs(alpha)
        print("--- Test Suite Finished ---")
        for res in self.results:
            print(res)

    def plot_distributions(self, filename: str, num_bins: int = 50):
        """Generates and saves a plot comparing the sequence and benchmark distributions."""
        plt.figure(figsize=(12, 7))
        
        non_zero_benchmark = self.benchmark_dist[self.benchmark_dist > 0]
        non_zero_sequence = self.sequence_wpf[self.sequence_wpf > 0]
        
        max_entropy = max(np.max(non_zero_benchmark), np.max(non_zero_sequence)) if len(non_zero_sequence) > 0 else np.max(non_zero_benchmark)
        bins = np.linspace(0, max_entropy, num_bins)

        plt.hist(non_zero_benchmark, bins=bins, density=True, alpha=0.6, label='Benchmark (Natural Numbers)', color='gray')
        if len(non_zero_sequence) > 0:
            plt.hist(non_zero_sequence, bins=bins, density=True, alpha=0.8, label='Test Sequence', color='royalblue', edgecolor='white')
        
        plt.title('Comparison of WPF Entropy Distributions (Non-Zero Values)', fontsize=16)
        plt.xlabel('WPF Entropy', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plots_dir = os.path.dirname(filename)
        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            
        plt.savefig(filename)
        print(f"\n[INFO] Distribution plot saved to '{filename}'")
        plt.close()
