#!/usr/bin/env python3
"""
Cache Hit Rate Convergence Analysis

Demonstrates that 10K queries is sufficient for measuring cache effectiveness.

Key Question: At what dataset size does hit rate stabilize?

Method:
1. Sample results at increasing sizes (1K, 2K, 5K, 10K, 20K, 50K)
2. Calculate hit rate and variance at each size
3. Statistical test: Is 10K significantly different from 50K?
4. Plot convergence curve with confidence intervals

Usage: python3 analyze_convergence.py --results-dir ../results/q1_comprehensive_*
"""

import argparse
import json
import glob
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_query_logs(results_dir):
    """Load all query logs from .logs.jsonl files."""
    log_files = list(Path(results_dir).glob('*.logs.jsonl'))
    
    if not log_files:
        print(f"⚠️  No .logs.jsonl files found in {results_dir}")
        return []
    
    all_logs = []
    for log_file in log_files:
        try:
            with open(log_file) as f:
                logs = [json.loads(line) for line in f if line.strip()]
                all_logs.append(logs)
        except Exception as e:
            print(f"⚠️  Error reading {log_file}: {e}")
            continue
    
    return all_logs


def calculate_hit_rate_at_size(logs, size):
    """Calculate hit rate using first N queries."""
    if len(logs) < size:
        return None
    
    hits = sum(1 for log in logs[:size] if log.get('isHit', False))
    return (hits / size) * 100


def analyze_convergence(results_dir, output_dir):
    """
    Main convergence analysis.
    
    Returns:
        dict with convergence metrics
    """
    print("=" * 70)
    print("CACHE HIT RATE CONVERGENCE ANALYSIS")
    print("=" * 70)
    print()
    
    all_logs = load_query_logs(results_dir)
    
    if not all_logs:
        print("❌ No query logs found. Cannot perform convergence analysis.")
        print()
        print("To generate logs, run benchmark with detailed logging enabled:")
        print("  ./bin/run_q1_comprehensive_benchmark.sh")
        return None
    
    print(f"Loaded {len(all_logs)} experiment runs")
    print()
    
    # Sample sizes to test
    sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    # Calculate hit rates at each size
    results = {}
    for size in sizes:
        rates = []
        for logs in all_logs:
            rate = calculate_hit_rate_at_size(logs, size)
            if rate is not None:
                rates.append(rate)
        
        if rates:
            results[size] = {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'n': len(rates),
                'rates': rates
            }
    
    if not results:
        print("❌ Insufficient data for convergence analysis")
        return None
    
    # Print results
    print(f"{'Size':<10} | {'Hit Rate':<15} | {'Std Dev':<10} | {'CV':<10} | {'N':<5}")
    print("-" * 70)
    
    for size in sorted(results.keys()):
        r = results[size]
        cv = (r['std'] / r['mean']) * 100 if r['mean'] > 0 else 0
        print(f"{size:<10} | {r['mean']:>6.2f}% ± {r['std']:>4.2f}% | "
              f"{r['std']:>9.2f}% | {cv:>9.2f}% | {r['n']:<5}")
    
    print()
    
    # Statistical test: 10K vs largest available
    if 10000 in results and len(results) > 1:
        largest_size = max(results.keys())
        
        if largest_size > 10000:
            rates_10k = results[10000]['rates']
            rates_large = results[largest_size]['rates']
            
            # Paired t-test (same experiments, different sample sizes)
            if len(rates_10k) == len(rates_large):
                t_stat, p_value = stats.ttest_rel(rates_10k, rates_large)
                test_type = "Paired t-test"
            else:
                t_stat, p_value = stats.ttest_ind(rates_10k, rates_large)
                test_type = "Independent t-test"
            
            print(f"{test_type}: 10K vs {largest_size}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print()
            
            if p_value > 0.05:
                print("✅ CONVERGENCE ACHIEVED")
                print(f"   No significant difference between 10K and {largest_size} (p={p_value:.4f})")
                print("   10K queries is sufficient for measuring cache effectiveness")
            else:
                print("⚠️  CONVERGENCE NOT ACHIEVED")
                print(f"   Significant difference detected (p={p_value:.4f})")
                print(f"   Consider using {largest_size}+ queries")
            
            print()
    
    # Plot convergence curve
    plot_convergence(results, output_dir)
    
    # Calculate convergence metrics
    convergence_metrics = calculate_convergence_metrics(results)
    
    return convergence_metrics


def calculate_convergence_metrics(results):
    """Calculate when convergence is achieved."""
    sizes = sorted(results.keys())
    
    if len(sizes) < 2:
        return None
    
    # Find size where CV drops below 5%
    convergence_size = None
    for size in sizes:
        r = results[size]
        cv = (r['std'] / r['mean']) * 100 if r['mean'] > 0 else 0
        if cv < 5.0:
            convergence_size = size
            break
    
    # Find size where change from previous is < 2%
    stable_size = None
    for i in range(1, len(sizes)):
        prev_mean = results[sizes[i-1]]['mean']
        curr_mean = results[sizes[i]]['mean']
        change = abs(curr_mean - prev_mean) / prev_mean * 100
        if change < 2.0:
            stable_size = sizes[i]
            break
    
    return {
        'convergence_size_cv': convergence_size,
        'stable_size_change': stable_size,
        'final_mean': results[sizes[-1]]['mean'],
        'final_std': results[sizes[-1]]['std']
    }


def plot_convergence(results, output_dir):
    """Generate publication-quality convergence plot."""
    sizes = sorted(results.keys())
    means = [results[s]['mean'] for s in sizes]
    stds = [results[s]['std'] for s in sizes]
    
    # Calculate 95% confidence intervals
    cis = []
    for s in sizes:
        r = results[s]
        n = r['n']
        if n > 1:
            ci = 1.96 * (r['std'] / np.sqrt(n))
            cis.append(ci)
        else:
            cis.append(0)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Main plot
    plt.errorbar(sizes, means, yerr=cis, marker='o', markersize=8,
                 linewidth=2, capsize=5, capthick=2,
                 label='Hit Rate (95% CI)')
    
    # Steady-state line (last value)
    steady_state = means[-1]
    plt.axhline(y=steady_state, color='red', linestyle='--', linewidth=2,
                label=f'Steady-state ({steady_state:.1f}%)')
    
    # 2% tolerance band
    plt.axhline(y=steady_state * 0.98, color='red', linestyle=':', 
                linewidth=1, alpha=0.5)
    plt.axhline(y=steady_state * 1.02, color='red', linestyle=':', 
                linewidth=1, alpha=0.5)
    plt.fill_between(sizes, steady_state * 0.98, steady_state * 1.02,
                     color='red', alpha=0.1, label='±2% tolerance')
    
    # Highlight 10K
    if 10000 in sizes:
        idx = sizes.index(10000)
        plt.axvline(x=10000, color='green', linestyle='--', linewidth=2,
                   label='Our choice (10K)')
        plt.plot(10000, means[idx], 'go', markersize=15, markeredgewidth=2,
                markerfacecolor='none')
    
    plt.xlabel('Dataset Size (queries)', fontsize=14, fontweight='bold')
    plt.ylabel('Cache Hit Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Cache Hit Rate Convergence Analysis', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'convergence_analysis.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Convergence plot saved: {output_path}")
    
    # Also save PNG for README
    output_path_png = Path(output_dir) / 'convergence_analysis.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"✅ Convergence plot saved: {output_path_png}")
    
    plt.close()


def generate_latex_table(results, output_dir):
    """Generate LaTeX table for paper."""
    sizes = sorted(results.keys())
    
    latex = r"""\begin{table}[h]
\centering
\caption{Cache Hit Rate Convergence Analysis}
\label{tab:convergence}
\begin{tabular}{rrrr}
\toprule
Dataset Size & Hit Rate (\%) & Std Dev (\%) & CV (\%) \\
\midrule
"""
    
    for size in sizes:
        r = results[size]
        cv = (r['std'] / r['mean']) * 100 if r['mean'] > 0 else 0
        
        # Highlight 10K
        if size == 10000:
            latex += r"\rowcolor{lightgray}" + "\n"
        
        latex += f"{size:,} & {r['mean']:.2f} & {r['std']:.2f} & {cv:.2f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    output_path = Path(output_dir) / 'convergence_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze cache hit rate convergence'
    )
    parser.add_argument('--results-dir', required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for plots and tables')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"❌ Error: {results_dir} does not exist")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CONVERGENCE ANALYSIS FOR Q1" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    metrics = analyze_convergence(results_dir, output_dir)
    
    if metrics:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print(f"Convergence achieved at: {metrics.get('convergence_size_cv', 'N/A')} queries (CV < 5%)")
        print(f"Stability achieved at: {metrics.get('stable_size_change', 'N/A')} queries (change < 2%)")
        print(f"Final hit rate: {metrics['final_mean']:.2f}% ± {metrics['final_std']:.2f}%")
        print()
        print("✅ Analysis complete. Use figures in your paper to justify 10K dataset size.")
        print()
    
    return 0


if __name__ == '__main__':
    exit(main())
