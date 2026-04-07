#!/usr/bin/env python3
"""
Effect Size Calculator for Q1 Publications

Calculates multiple effect size measures:
- Cohen's d (standardized mean difference)
- Hedges' g (bias-corrected Cohen's d)
- Glass's delta (using control group SD)
- Probability of superiority
- Common language effect size

Required for senior-level Q1 publications.
"""

import argparse
import numpy as np
from scipy import stats
import json


def cohens_d(group1, group2):
    """
    Cohen's d: Standardized mean difference using pooled SD.
    
    Interpretation:
    - 0.2: Small effect
    - 0.5: Medium effect
    - 0.8: Large effect
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def hedges_g(group1, group2):
    """
    Hedges' g: Bias-corrected Cohen's d for small samples.
    
    Recommended for N < 20 per group.
    """
    n1, n2 = len(group1), len(group2)
    d = cohens_d(group1, group2)
    
    # Correction factor
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    
    return d * correction


def glass_delta(group1, group2):
    """
    Glass's delta: Uses control group SD only.
    
    Useful when groups have different variances.
    """
    std2 = np.std(group2, ddof=1)
    
    if std2 == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / std2


def probability_superiority(group1, group2):
    """
    Probability that a random value from group1 > group2.
    
    Also known as the Area Under the ROC Curve (AUC).
    """
    n1, n2 = len(group1), len(group2)
    
    # Count how many times group1 > group2
    count = 0
    for val1 in group1:
        for val2 in group2:
            if val1 > val2:
                count += 1
            elif val1 == val2:
                count += 0.5
    
    return count / (n1 * n2)


def common_language_effect_size(group1, group2):
    """
    Probability that a random person from group1 has a higher score than
    a random person from group2.
    
    More interpretable than Cohen's d for non-statisticians.
    """
    return probability_superiority(group1, group2)


def confidence_interval_d(group1, group2, confidence=0.95):
    """
    Calculate confidence interval for Cohen's d.
    """
    n1, n2 = len(group1), len(group2)
    d = cohens_d(group1, group2)
    
    # Standard error of d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    
    # Critical value
    alpha = 1 - confidence
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    ci_lower = d - z_crit * se_d
    ci_upper = d + z_crit * se_d
    
    return ci_lower, ci_upper


def interpret_effect_size(d):
    """
    Interpret Cohen's d according to standard thresholds.
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    parser = argparse.ArgumentParser(
        description='Calculate effect sizes for Q1 publications'
    )
    parser.add_argument('--group1', nargs='+', type=float, required=True,
                       help='Values for experimental group')
    parser.add_argument('--group2', nargs='+', type=float, required=True,
                       help='Values for control group')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level (default: 0.95)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    group1 = np.array(args.group1)
    group2 = np.array(args.group2)
    
    # Calculate all effect sizes
    d = cohens_d(group1, group2)
    g = hedges_g(group1, group2)
    delta = glass_delta(group1, group2)
    prob_sup = probability_superiority(group1, group2)
    cles = common_language_effect_size(group1, group2)
    ci_lower, ci_upper = confidence_interval_d(group1, group2, args.confidence)
    
    # Interpretation
    interpretation = interpret_effect_size(d)
    
    # Results
    results = {
        'sample_sizes': {
            'group1': len(group1),
            'group2': len(group2)
        },
        'descriptive_statistics': {
            'group1': {
                'mean': float(np.mean(group1)),
                'sd': float(np.std(group1, ddof=1)),
                'median': float(np.median(group1))
            },
            'group2': {
                'mean': float(np.mean(group2)),
                'sd': float(np.std(group2, ddof=1)),
                'median': float(np.median(group2))
            }
        },
        'effect_sizes': {
            'cohens_d': {
                'value': float(d),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'interpretation': interpretation
            },
            'hedges_g': {
                'value': float(g),
                'note': 'Bias-corrected for small samples'
            },
            'glass_delta': {
                'value': float(delta),
                'note': 'Uses control group SD only'
            },
            'probability_superiority': {
                'value': float(prob_sup),
                'interpretation': f'{prob_sup*100:.1f}% chance group1 > group2'
            },
            'common_language_effect_size': {
                'value': float(cles),
                'interpretation': f'{cles*100:.1f}% of group1 exceeds average of group2'
            }
        }
    }
    
    # Print results
    print("=" * 70)
    print("EFFECT SIZE ANALYSIS")
    print("=" * 70)
    print()
    print(f"Sample Sizes: n1={len(group1)}, n2={len(group2)}")
    print()
    print("Descriptive Statistics:")
    print(f"  Group 1: M={np.mean(group1):.3f}, SD={np.std(group1, ddof=1):.3f}")
    print(f"  Group 2: M={np.mean(group2):.3f}, SD={np.std(group2, ddof=1):.3f}")
    print()
    print("Effect Sizes:")
    print(f"  Cohen's d:     {d:.3f} ({interpretation})")
    print(f"    95% CI:      [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Hedges' g:     {g:.3f} (bias-corrected)")
    print(f"  Glass's Δ:     {delta:.3f}")
    print(f"  Prob. Sup.:    {prob_sup:.3f} ({prob_sup*100:.1f}%)")
    print(f"  CLES:          {cles:.3f} ({cles*100:.1f}%)")
    print()
    print("Interpretation:")
    print(f"  • The effect size is {interpretation} (|d| = {abs(d):.3f})")
    print(f"  • {cles*100:.1f}% of experimental group exceeds control group average")
    print(f"  • {prob_sup*100:.1f}% probability that random experimental > control")
    print()
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
