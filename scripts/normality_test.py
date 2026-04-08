#!/usr/bin/env python3
"""
Normality Testing for Statistical Assumptions

Tests whether data meets normality assumptions required for parametric tests.
Provides recommendations for appropriate statistical tests.

Tests performed:
- Shapiro-Wilk test (n < 50)
- Kolmogorov-Smirnov test (n ≥ 50)
- Anderson-Darling test
- Q-Q plot visualization

Usage: python3 normality_test.py --results-dir ../results/ --metric hitRate
"""

import argparse
import json
import glob
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def load_metric_data(results_dir, metric):
    """Load specific metric from all result files."""
    json_files = list(Path(results_dir).glob('*.json'))
    
    if not json_files:
        print(f"Error: No JSON files found in {results_dir}")
        return None
    
    values = []
    for filepath in json_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            value = data.get(metric)
            if value is not None:
                values.append(float(value))
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue
    
    if not values:
        print(f"Error: No values found for metric '{metric}'")
        return None
    
    return np.array(values)


def shapiro_wilk_test(data):
    """Shapiro-Wilk test for normality (best for n < 50)."""
    print("--- Shapiro-Wilk Test ---")
    print()
    
    if len(data) < 3:
        print("⚠️  Insufficient data (n < 3)")
        return None
    
    if len(data) > 5000:
        print("⚠️  Sample too large (n > 5000), using K-S test instead")
        return None
    
    stat, p_value = stats.shapiro(data)
    
    print(f"Statistic: {stat:.6f}")
    print(f"p-value: {p_value:.6f}")
    print()
    
    if p_value > 0.05:
        print("✅ Data appears normally distributed (p > 0.05)")
        print("   Parametric tests are appropriate")
    else:
        print("⚠️  Data may not be normally distributed (p ≤ 0.05)")
        print("   Consider non-parametric tests")
    
    print()
    return p_value


def kolmogorov_smirnov_test(data):
    """Kolmogorov-Smirnov test for normality (good for n ≥ 50)."""
    print("--- Kolmogorov-Smirnov Test ---")
    print()
    
    if len(data) < 50:
        print("⚠️  Sample too small (n < 50), use Shapiro-Wilk instead")
        return None
    
    # Standardize data
    standardized = (data - np.mean(data)) / np.std(data)
    
    # Test against standard normal
    stat, p_value = stats.kstest(standardized, 'norm')
    
    print(f"Statistic: {stat:.6f}")
    print(f"p-value: {p_value:.6f}")
    print()
    
    if p_value > 0.05:
        print("✅ Data appears normally distributed (p > 0.05)")
    else:
        print("⚠️  Data may not be normally distributed (p ≤ 0.05)")
    
    print()
    return p_value


def anderson_darling_test(data):
    """Anderson-Darling test for normality."""
    print("--- Anderson-Darling Test ---")
    print()
    
    result = stats.anderson(data, dist='norm')
    
    print(f"Statistic: {result.statistic:.6f}")
    print()
    print("Critical values:")
    for i, (sig_level, crit_val) in enumerate(zip(result.significance_level, result.critical_values)):
        if result.statistic < crit_val:
            status = "✅"
        else:
            status = "⚠️"
        print(f"  {status} {sig_level}%: {crit_val:.6f}")
    
    print()
    
    # Check at 5% significance level
    if result.statistic < result.critical_values[2]:  # 5% is typically index 2
        print("✅ Data appears normally distributed (5% level)")
    else:
        print("⚠️  Data may not be normally distributed (5% level)")
    
    print()


def qq_plot(data, output_file=None):
    """Generate Q-Q plot for visual normality assessment."""
    print("--- Q-Q Plot ---")
    print()
    
    plt.figure(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Q-Q Plot (Normal Distribution)")
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Q-Q plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    print()


def descriptive_statistics(data):
    """Calculate descriptive statistics."""
    print("--- Descriptive Statistics ---")
    print()
    
    print(f"Sample size: {len(data)}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Std Dev: {np.std(data, ddof=1):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")
    print(f"Skewness: {stats.skew(data):.4f}")
    print(f"Kurtosis: {stats.kurtosis(data):.4f}")
    print()
    
    # Interpretation
    skewness = stats.skew(data)
    if abs(skewness) < 0.5:
        print("✅ Skewness: Approximately symmetric")
    elif abs(skewness) < 1.0:
        print("⚠️  Skewness: Moderately skewed")
    else:
        print("⚠️  Skewness: Highly skewed")
    
    kurtosis = stats.kurtosis(data)
    if abs(kurtosis) < 0.5:
        print("✅ Kurtosis: Approximately normal")
    elif abs(kurtosis) < 1.0:
        print("⚠️  Kurtosis: Moderately non-normal")
    else:
        print("⚠️  Kurtosis: Highly non-normal")
    
    print()


def recommend_tests(p_values):
    """Recommend appropriate statistical tests based on normality."""
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    # Check if any test rejected normality
    valid_p_values = [p for p in p_values if p is not None]
    
    if not valid_p_values:
        print("⚠️  Could not determine normality")
        return
    
    min_p = min(valid_p_values)
    
    if min_p > 0.05:
        print("✅ Data appears normally distributed")
        print()
        print("Recommended parametric tests:")
        print("  • Independent samples: Two-sample t-test")
        print("  • Paired samples: Paired t-test")
        print("  • Multiple groups: One-way ANOVA")
        print("  • Correlation: Pearson correlation")
        print()
    else:
        print("⚠️  Data may not be normally distributed")
        print()
        print("Recommended non-parametric tests:")
        print("  • Independent samples: Mann-Whitney U test")
        print("  • Paired samples: Wilcoxon signed-rank test")
        print("  • Multiple groups: Kruskal-Wallis test")
        print("  • Correlation: Spearman correlation")
        print()
        print("Alternative approaches:")
        print("  • Transform data (log, sqrt, Box-Cox)")
        print("  • Use robust statistics (median, MAD)")
        print("  • Bootstrap confidence intervals")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Test normality assumptions for statistical analysis'
    )
    parser.add_argument('--results-dir', required=True,
                       help='Directory containing result JSON files')
    parser.add_argument('--metric', default='hitRate',
                       help='Metric to test (default: hitRate)')
    parser.add_argument('--output', type=str,
                       help='Output file for Q-Q plot (PNG)')
    
    args = parser.parse_args()
    
    print()
    print("=" * 70)
    print("NORMALITY TESTING")
    print("=" * 70)
    print()
    
    # Load data
    print(f"Loading metric: {args.metric}")
    print(f"From directory: {args.results_dir}")
    print()
    
    data = load_metric_data(args.results_dir, args.metric)
    
    if data is None:
        return 1
    
    print(f"Loaded {len(data)} values")
    print()
    
    # Descriptive statistics
    descriptive_statistics(data)
    
    # Normality tests
    p_values = []
    
    # Shapiro-Wilk (best for small samples)
    if len(data) < 5000:
        p = shapiro_wilk_test(data)
        if p is not None:
            p_values.append(p)
    
    # Kolmogorov-Smirnov (good for large samples)
    if len(data) >= 50:
        p = kolmogorov_smirnov_test(data)
        if p is not None:
            p_values.append(p)
    
    # Anderson-Darling (always applicable)
    anderson_darling_test(data)
    
    # Q-Q plot
    qq_plot(data, args.output)
    
    # Recommendations
    recommend_tests(p_values)
    
    return 0


if __name__ == '__main__':
    exit(main())
