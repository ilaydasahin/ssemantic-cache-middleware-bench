"""
Statistical Power Analysis for Semantic Cache Benchmark.

Computes required sample size (N seeds) to detect meaningful effect sizes
with 80% power at α=0.05 significance level.

Usage: python3 power_analysis.py --results-dir results/

Dependencies: statsmodels, numpy, scipy
"""

import argparse
import numpy as np
from statsmodels.stats.power import TTestIndPower
from scipy import stats


def compute_required_sample_size(effect_size: float, alpha: float = 0.05, power: float = 0.80):
    """
    Calculate required N per group for independent t-test.
    
    Args:
        effect_size: Cohen's d (small=0.2, medium=0.5, large=0.8)
        alpha: Type I error rate (typically 0.05)
        power: Statistical power (typically 0.80)
    
    Returns:
        Required sample size per group
    """
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
    return int(np.ceil(n))


def analyze_pilot_data(pilot_results: list):
    """
    Estimate effect size from pilot data to inform full experiment design.
    
    Args:
        pilot_results: List of dicts with 'strategy' and 'hitRate' keys
    
    Returns:
        Estimated Cohen's d between strategies
    """
    strategies = {}
    for result in pilot_results:
        strat = result['strategy']
        if strat not in strategies:
            strategies[strat] = []
        strategies[strat].append(result['hitRate'])
    
    if len(strategies) < 2:
        return None
    
    # Compare first two strategies
    strat_names = list(strategies.keys())
    group1 = np.array(strategies[strat_names[0]])
    group2 = np.array(strategies[strat_names[1]])
    
    pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return abs(cohens_d)


def main():
    parser = argparse.ArgumentParser(description="Power analysis for benchmark")
    parser.add_argument("--effect-size", type=float, default=0.5, 
                       help="Expected Cohen's d (0.2=small, 0.5=medium, 0.8=large)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--power", type=float, default=0.80, help="Desired statistical power")
    args = parser.parse_args()
    
    print("=== Statistical Power Analysis ===\n")
    
    # Calculate for different effect sizes
    effect_sizes = {
        "Small (d=0.2)": 0.2,
        "Medium (d=0.5)": 0.5,
        "Large (d=0.8)": 0.8,
        "User-specified": args.effect_size
    }
    
    print(f"Target: α={args.alpha}, Power={args.power}\n")
    print(f"{'Effect Size':<20} | {'Required N (per group)':<25} | {'Total Seeds Needed'}")
    print("-" * 75)
    
    for label, d in effect_sizes.items():
        n = compute_required_sample_size(d, args.alpha, args.power)
        total = n * 2  # Two groups comparison
        print(f"{label:<20} | {n:<25} | {total}")
    
    print("\n--- Recommendations ---")
    print("• For medium effects (d=0.5): Use at least 64 seeds per configuration")
    print("• For large effects (d=0.8): Use at least 26 seeds per configuration")
    print("• Current setup (3-5 seeds): Only detects very large effects (d>1.5)")
    print("\n⚠️  Q1 journals require justification if N < power analysis recommendation")


if __name__ == "__main__":
    main()
