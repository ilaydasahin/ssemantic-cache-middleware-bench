#!/usr/bin/env python3
"""
Generate Publication-Ready Figures for Q1 Journal

Creates high-quality figures suitable for academic publication:
- Box plots for performance comparison
- Violin plots for distribution visualization
- Pareto fronts for multi-objective trade-offs
- Heatmaps for configuration comparison
- Bar charts for statistical significance

All figures are saved in publication-ready formats (PDF, PNG, SVG).
"""

import argparse
import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 13
rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
rcParams['ps.fonttype'] = 42

# Color palette
COLORS = {
    'SEMANTIC': '#2E86AB',
    'EXACT_MATCH': '#A23B72',
    'NONE': '#F18F01',
    'HYBRID': '#C73E1D'
}


def load_results(results_dir):
    """Load all JSON result files from directory."""
    results = []
    json_files = list(Path(results_dir).glob('*.json'))
    
    for filepath in json_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return results


def group_by_strategy(results):
    """Group results by cache strategy."""
    grouped = defaultdict(list)
    
    for result in results:
        strategy = result.get('strategy', 'UNKNOWN')
        grouped[strategy].append(result)
    
    return grouped


def figure1_hit_rate_comparison(results, output_dir):
    """Figure 1: Hit Rate Comparison (Box Plot)"""
    print("Generating Figure 1: Hit Rate Comparison...")
    
    grouped = group_by_strategy(results)
    
    # Prepare data
    strategies = []
    hit_rates = []
    
    for strategy, data in sorted(grouped.items()):
        for result in data:
            strategies.append(strategy)
            hit_rates.append(result.get('hitRate', 0) * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Box plot
    positions = []
    labels = []
    data_by_strategy = []
    
    for i, (strategy, data) in enumerate(sorted(grouped.items())):
        rates = [r.get('hitRate', 0) * 100 for r in data]
        data_by_strategy.append(rates)
        positions.append(i)
        labels.append(strategy)
    
    bp = ax.boxplot(data_by_strategy, positions=positions, labels=labels,
                     patch_artist=True, widths=0.6)
    
    # Color boxes
    for patch, strategy in zip(bp['boxes'], labels):
        patch.set_facecolor(COLORS.get(strategy, '#CCCCCC'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Hit Rate (%)')
    ax.set_xlabel('Cache Strategy')
    ax.set_title('Hit Rate Comparison Across Strategies')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(f"{output_dir}/figure1_hit_rate_comparison.{fmt}",
                   dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  Saved to {output_dir}/figure1_hit_rate_comparison.*")


def figure2_latency_distribution(results, output_dir):
    """Figure 2: Latency Distribution (Violin Plot)"""
    print("Generating Figure 2: Latency Distribution...")
    
    grouped = group_by_strategy(results)
    
    # Prepare data
    fig, ax = plt.subplots(figsize=(6, 4))
    
    data_by_strategy = []
    labels = []
    
    for strategy, data in sorted(grouped.items()):
        latencies = [r.get('p99Latency', 0) for r in data]
        data_by_strategy.append(latencies)
        labels.append(strategy)
    
    # Violin plot
    parts = ax.violinplot(data_by_strategy, positions=range(len(labels)),
                          showmeans=True, showmedians=True)
    
    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS.get(labels[i], '#CCCCCC'))
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_xlabel('Cache Strategy')
    ax.set_title('Latency Distribution Across Strategies')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(f"{output_dir}/figure2_latency_distribution.{fmt}",
                   dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  Saved to {output_dir}/figure2_latency_distribution.*")


def figure3_pareto_front(results, output_dir):
    """Figure 3: Pareto Front (Hit Rate vs Latency)"""
    print("Generating Figure 3: Pareto Front...")
    
    grouped = group_by_strategy(results)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for strategy, data in sorted(grouped.items()):
        hit_rates = [r.get('hitRate', 0) * 100 for r in data]
        latencies = [r.get('p99Latency', 0) for r in data]
        
        ax.scatter(hit_rates, latencies, 
                  label=strategy,
                  color=COLORS.get(strategy, '#CCCCCC'),
                  alpha=0.6, s=50)
    
    ax.set_xlabel('Hit Rate (%)')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('Pareto Front: Hit Rate vs Latency Trade-off')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(f"{output_dir}/figure3_pareto_front.{fmt}",
                   dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  Saved to {output_dir}/figure3_pareto_front.*")


def figure4_cost_savings(results, output_dir):
    """Figure 4: Cost Savings Comparison (Bar Chart)"""
    print("Generating Figure 4: Cost Savings...")
    
    grouped = group_by_strategy(results)
    
    # Calculate mean and std
    strategies = []
    means = []
    stds = []
    
    for strategy, data in sorted(grouped.items()):
        savings = [r.get('costSavings', 0) * 100 for r in data]
        strategies.append(strategy)
        means.append(np.mean(savings))
        stds.append(np.std(savings))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    x = np.arange(len(strategies))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS.get(s, '#CCCCCC') for s in strategies],
                  alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_ylabel('Cost Savings (%)')
    ax.set_xlabel('Cache Strategy')
    ax.set_title('Cost Savings Comparison (Mean ± SD)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(f"{output_dir}/figure4_cost_savings.{fmt}",
                   dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  Saved to {output_dir}/figure4_cost_savings.*")


def figure5_heatmap(results, output_dir):
    """Figure 5: Configuration Heatmap"""
    print("Generating Figure 5: Configuration Heatmap...")
    
    # Group by strategy and threshold
    matrix_data = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        strategy = result.get('strategy', 'UNKNOWN')
        threshold = result.get('threshold', 0.9)
        hit_rate = result.get('hitRate', 0) * 100
        matrix_data[strategy][threshold].append(hit_rate)
    
    # Create matrix
    strategies = sorted(matrix_data.keys())
    thresholds = sorted(set(t for s in matrix_data.values() for t in s.keys()))
    
    matrix = np.zeros((len(strategies), len(thresholds)))
    
    for i, strategy in enumerate(strategies):
        for j, threshold in enumerate(thresholds):
            if threshold in matrix_data[strategy]:
                matrix[i, j] = np.mean(matrix_data[strategy][threshold])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax.set_yticklabels(strategies)
    
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Cache Strategy')
    ax.set_title('Hit Rate Heatmap: Strategy × Threshold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Hit Rate (%)', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(thresholds)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(f"{output_dir}/figure5_heatmap.{fmt}",
                   dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  Saved to {output_dir}/figure5_heatmap.*")


def figure6_throughput(results, output_dir):
    """Figure 6: Throughput Comparison"""
    print("Generating Figure 6: Throughput Comparison...")
    
    grouped = group_by_strategy(results)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    data_by_strategy = []
    labels = []
    
    for strategy, data in sorted(grouped.items()):
        throughputs = [r.get('throughput', 0) / 1000 for r in data]  # Convert to K rps
        data_by_strategy.append(throughputs)
        labels.append(strategy)
    
    bp = ax.boxplot(data_by_strategy, labels=labels, patch_artist=True)
    
    for patch, strategy in zip(bp['boxes'], labels):
        patch.set_facecolor(COLORS.get(strategy, '#CCCCCC'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Throughput (K requests/sec)')
    ax.set_xlabel('Cache Strategy')
    ax.set_title('Throughput Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(f"{output_dir}/figure6_throughput.{fmt}",
                   dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  Saved to {output_dir}/figure6_throughput.*")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready figures for Q1 journal'
    )
    parser.add_argument('results_dir', help='Directory containing result JSON files')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for figures (default: results_dir/figures)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return 1
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    if not results:
        print("Error: No result files found")
        return 1
    
    print(f"Loaded {len(results)} result files")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate all figures
    figure1_hit_rate_comparison(results, output_dir)
    figure2_latency_distribution(results, output_dir)
    figure3_pareto_front(results, output_dir)
    figure4_cost_savings(results, output_dir)
    figure5_heatmap(results, output_dir)
    figure6_throughput(results, output_dir)
    
    print()
    print("✅ All figures generated successfully!")
    print(f"   Location: {output_dir}")
    print()
    print("Figures created:")
    print("  • figure1_hit_rate_comparison.* - Box plot comparison")
    print("  • figure2_latency_distribution.* - Violin plot")
    print("  • figure3_pareto_front.* - Hit rate vs latency trade-off")
    print("  • figure4_cost_savings.* - Bar chart with error bars")
    print("  • figure5_heatmap.* - Configuration heatmap")
    print("  • figure6_throughput.* - Throughput comparison")
    print()
    print("Formats: PDF (vector), PNG (raster), SVG (web)")
    
    return 0


if __name__ == '__main__':
    exit(main())
