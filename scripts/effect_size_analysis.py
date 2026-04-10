#!/usr/bin/env python3
"""
Effect Size Analysis for Q1 Publication

Calculates Cohen's d, confidence intervals, and effect size interpretations
for semantic cache benchmark results. Follows APA/IEEE reporting standards.

Metrics:
- Cohen's d (standardized mean difference)
- Hedges' g (bias-corrected Cohen's d for small samples)
- 95% confidence intervals
- Effect size interpretation (negligible/small/medium/large)
- Practical significance assessment

Usage:
    python3 effect_size_analysis.py --results-dir results/q1_comprehensive
    python3 effect_size_analysis.py --results-dir results/q1_comprehensive --baseline EXACT_MATCH

Output:
- effect_sizes.json: Numerical results
- effect_sizes_table.tex: LaTeX table for paper
- effect_sizes_report.txt: Human-readable report
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats


def load_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """Load all benchmark results grouped by strategy."""
    results = defaultdict(list)
    
    for filepath in results_dir.glob("*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
                strategy = data.get("strategy", "UNKNOWN")
                results[strategy].append(data)
        except Exception as e:
            print(f"⚠️  Failed to load {filepath}: {e}")
    
    return dict(results)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Cohen's d = (M1 - M2) / pooled_std
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Hedges' g (bias-corrected Cohen's d).
    
    Hedges' g applies a correction factor for small sample sizes.
    Use when N < 20 per group.
    """
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    
    # Correction factor
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    
    g = d * correction
    return g


def confidence_interval_d(group1: np.ndarray, group2: np.ndarray, 
                         confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for Cohen's d.
    
    Returns:
        (d, lower_bound, upper_bound)
    """
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    
    # Standard error of d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    
    # Critical value for confidence interval
    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - alpha / 2)
    
    lower = d - z_critical * se_d
    upper = d + z_critical * se_d
    
    return d, lower, upper


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d according to standard guidelines."""
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def practical_significance(d: float, metric: str) -> str:
    """Assess practical significance beyond statistical significance."""
    abs_d = abs(d)
    
    if metric == "hit_rate":
        if abs_d >= 0.8:
            return "Highly practically significant - substantial improvement in cache effectiveness"
        elif abs_d >= 0.5:
            return "Moderately practically significant - noticeable improvement"
        elif abs_d >= 0.2:
            return "Marginally practically significant - small but measurable improvement"
        else:
            return "Not practically significant - negligible difference"
    elif metric == "latency":
        if abs_d >= 0.8:
            return "Highly practically significant - major performance impact"
        elif abs_d >= 0.5:
            return "Moderately practically significant - noticeable performance difference"
        elif abs_d >= 0.2:
            return "Marginally practically significant - minor performance difference"
        else:
            return "Not practically significant - negligible performance difference"
    else:
        return interpret_effect_size(d)


def analyze_effect_sizes(results: Dict[str, List[Dict]], 
                        baseline: str = "EXACT_MATCH",
                        metric: str = "hit_rate") -> Dict:
    """
    Analyze effect sizes comparing all strategies to baseline.
    
    Args:
        results: Results grouped by strategy
        baseline: Baseline strategy name
        metric: Metric to compare (hit_rate, latency, etc.)
    
    Returns:
        Dictionary with effect size analysis
    """
    if baseline not in results:
        raise ValueError(f"Baseline strategy '{baseline}' not found in results")
    
    baseline_values = np.array([r.get(metric, 0) for r in results[baseline]])
    
    analysis = {
        "baseline": baseline,
        "metric": metric,
        "baseline_n": len(baseline_values),
        "baseline_mean": float(np.mean(baseline_values)),
        "baseline_std": float(np.std(baseline_values, ddof=1)),
        "comparisons": {}
    }
    
    for strategy, strategy_results in results.items():
        if strategy == baseline:
            continue
        
        strategy_values = np.array([r.get(metric, 0) for r in strategy_results])
        
        # Cohen's d with CI
        d, ci_lower, ci_upper = confidence_interval_d(strategy_values, baseline_values)
        
        # Hedges' g (for small samples)
        g = hedges_g(strategy_values, baseline_values)
        
        # T-test for statistical significance
        t_stat, p_value = stats.ttest_ind(strategy_values, baseline_values)
        
        # Interpretation
        interpretation = interpret_effect_size(d)
        practical = practical_significance(d, metric)
        
        analysis["comparisons"][strategy] = {
            "n": len(strategy_values),
            "mean": float(np.mean(strategy_values)),
            "std": float(np.std(strategy_values, ddof=1)),
            "cohens_d": float(d),
            "hedges_g": float(g),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "interpretation": interpretation,
            "practical_significance": practical,
            "statistically_significant": p_value < 0.05,
            "mean_difference": float(np.mean(strategy_values) - np.mean(baseline_values)),
            "percent_improvement": float((np.mean(strategy_values) - np.mean(baseline_values)) / np.mean(baseline_values) * 100) if np.mean(baseline_values) != 0 else 0
        }
    
    return analysis


def generate_report(analysis: Dict, output_dir: Path):
    """Generate human-readable report."""
    report_path = output_dir / "effect_sizes_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EFFECT SIZE ANALYSIS - Q1 PUBLICATION STANDARDS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Baseline Strategy: {analysis['baseline']}\n")
        f.write(f"Metric: {analysis['metric']}\n")
        f.write(f"Baseline N: {analysis['baseline_n']}\n")
        f.write(f"Baseline Mean: {analysis['baseline_mean']:.4f} ± {analysis['baseline_std']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PAIRWISE COMPARISONS\n")
        f.write("=" * 80 + "\n\n")
        
        for strategy, comp in sorted(analysis['comparisons'].items()):
            f.write(f"Strategy: {strategy}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Sample size:           N = {comp['n']}\n")
            f.write(f"  Mean:                  {comp['mean']:.4f} ± {comp['std']:.4f}\n")
            f.write(f"  Mean difference:       {comp['mean_difference']:.4f} ({comp['percent_improvement']:+.2f}%)\n")
            f.write(f"\n")
            f.write(f"  Cohen's d:             {comp['cohens_d']:.3f}\n")
            f.write(f"  95% CI:                [{comp['ci_95_lower']:.3f}, {comp['ci_95_upper']:.3f}]\n")
            f.write(f"  Hedges' g:             {comp['hedges_g']:.3f}\n")
            f.write(f"  Interpretation:        {comp['interpretation'].upper()}\n")
            f.write(f"\n")
            f.write(f"  t-statistic:           {comp['t_statistic']:.3f}\n")
            f.write(f"  p-value:               {comp['p_value']:.6f} {'***' if comp['p_value'] < 0.001 else '**' if comp['p_value'] < 0.01 else '*' if comp['p_value'] < 0.05 else 'ns'}\n")
            f.write(f"  Statistically sig.:    {'YES' if comp['statistically_significant'] else 'NO'}\n")
            f.write(f"\n")
            f.write(f"  Practical significance:\n")
            f.write(f"    {comp['practical_significance']}\n")
            f.write(f"\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 80 + "\n")
        f.write("""
Cohen's d interpretation (Cohen, 1988):
  • |d| < 0.2:  Negligible effect
  • 0.2 ≤ |d| < 0.5:  Small effect
  • 0.5 ≤ |d| < 0.8:  Medium effect
  • |d| ≥ 0.8:  Large effect

Statistical significance:
  • p < 0.001:  *** (highly significant)
  • p < 0.01:   ** (very significant)
  • p < 0.05:   * (significant)
  • p ≥ 0.05:   ns (not significant)

Confidence intervals:
  • If CI excludes 0, effect is statistically significant
  • Narrow CI indicates precise estimate
  • Wide CI indicates high variability

Reporting in paper:
  "Strategy X showed a [interpretation] improvement over baseline
   (d = [cohens_d], 95% CI [ci_lower, ci_upper], p [< or =] [p_value])."

Example:
  "SEMANTIC strategy showed a large improvement over EXACT_MATCH
   (d = 1.23, 95% CI [0.98, 1.48], p < 0.001)."
""")
    
    print(f"✅ Report saved to: {report_path}")


def generate_latex_table(analysis: Dict, output_dir: Path):
    """Generate LaTeX table for paper."""
    latex_path = output_dir / "effect_sizes_table.tex"
    
    latex = r"""\begin{table}[h]
\centering
\caption{Effect Sizes Comparing Strategies to """ + analysis['baseline'] + r""" Baseline}
\label{tab:effect_sizes}
\begin{tabular}{lrrrrrl}
\toprule
Strategy & Mean & $\Delta$ & Cohen's $d$ & 95\% CI & $p$ & Interpretation \\
\midrule
"""
    
    # Baseline row
    latex += f"{analysis['baseline']} (baseline) & {analysis['baseline_mean']:.3f} & --- & --- & --- & --- & --- \\\\\n"
    
    # Comparison rows
    for strategy, comp in sorted(analysis['comparisons'].items(), 
                                 key=lambda x: abs(x[1]['cohens_d']), 
                                 reverse=True):
        mean = comp['mean']
        delta = comp['mean_difference']
        d = comp['cohens_d']
        ci_lower = comp['ci_95_lower']
        ci_upper = comp['ci_95_upper']
        p = comp['p_value']
        interp = comp['interpretation']
        
        # Format p-value
        if p < 0.001:
            p_str = "$<$0.001***"
        elif p < 0.01:
            p_str = f"{p:.3f}**"
        elif p < 0.05:
            p_str = f"{p:.3f}*"
        else:
            p_str = f"{p:.3f}"
        
        latex += f"{strategy} & {mean:.3f} & {delta:+.3f} & {d:.2f} & [{ci_lower:.2f}, {ci_upper:.2f}] & {p_str} & {interp} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $\Delta$ = mean difference from baseline. 
\item * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$
\item Effect size interpretation: negligible ($|d| < 0.2$), small ($0.2 \leq |d| < 0.5$), medium ($0.5 \leq |d| < 0.8$), large ($|d| \geq 0.8$)
\end{tablenotes}
\end{table}
"""
    
    with open(latex_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved to: {latex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Effect size analysis for Q1 publication"
    )
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing benchmark results")
    parser.add_argument("--baseline", default="EXACT_MATCH",
                       help="Baseline strategy for comparison")
    parser.add_argument("--metric", default="hit_rate",
                       help="Metric to analyze (hit_rate, latency, etc.)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (defaults to results-dir)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("EFFECT SIZE ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Baseline: {args.baseline}")
    print(f"Metric: {args.metric}")
    print()
    
    # Load results
    print("Loading results...")
    results = load_results(results_dir)
    
    if not results:
        print("❌ No results found")
        return 1
    
    print(f"Found {len(results)} strategies:")
    for strategy, strategy_results in results.items():
        print(f"  • {strategy}: {len(strategy_results)} runs")
    print()
    
    # Analyze effect sizes
    print("Calculating effect sizes...")
    analysis = analyze_effect_sizes(results, args.baseline, args.metric)
    
    # Save JSON
    json_path = output_dir / "effect_sizes.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✅ JSON saved to: {json_path}")
    
    # Generate report
    generate_report(analysis, output_dir)
    
    # Generate LaTeX table
    generate_latex_table(analysis, output_dir)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Find best strategy
    best_strategy = max(analysis['comparisons'].items(), 
                       key=lambda x: abs(x[1]['cohens_d']))
    
    print(f"\nBest performing strategy: {best_strategy[0]}")
    print(f"  Cohen's d: {best_strategy[1]['cohens_d']:.3f} ({best_strategy[1]['interpretation']})")
    print(f"  Improvement: {best_strategy[1]['percent_improvement']:+.2f}%")
    print(f"  p-value: {best_strategy[1]['p_value']:.6f}")
    print()
    
    # Count significant results
    sig_count = sum(1 for comp in analysis['comparisons'].values() 
                   if comp['statistically_significant'])
    print(f"Statistically significant comparisons: {sig_count}/{len(analysis['comparisons'])}")
    
    # Count large effects
    large_count = sum(1 for comp in analysis['comparisons'].values() 
                     if abs(comp['cohens_d']) >= 0.8)
    print(f"Large effect sizes (|d| ≥ 0.8): {large_count}/{len(analysis['comparisons'])}")
    
    print("\n✅ Effect size analysis complete!")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
