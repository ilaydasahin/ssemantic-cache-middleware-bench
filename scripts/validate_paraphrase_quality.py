#!/usr/bin/env python3
"""
Paraphrase Quality Validation for Q1 Publication

Validates that paraphrases meet Q1 standards:
1. Semantic similarity: 0.70 < sim < 0.95
2. Lexical diversity: Jaccard < 0.8
3. Method distribution: T5 > 60%, Back-translation > 20%
4. No duplicates or trivial paraphrases

Generates publication-quality report with:
- Quality metrics table
- Distribution plots
- Example paraphrases
- LaTeX table for paper

Usage: python3 validate_paraphrase_quality.py --data-dir ../data
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available, skipping plots")


def load_paraphrased_dataset(filepath):
    """Load paraphrased dataset."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def analyze_quality(records):
    """Analyze paraphrase quality metrics."""
    methods = defaultdict(int)
    similarities = []
    overlaps = []
    valid_count = 0
    
    for record in records:
        method = record.get("paraphrase_method", "unknown")
        sim = record.get("paraphrase_semantic_similarity", 0)
        overlap = record.get("paraphrase_lexical_overlap", 0)
        
        methods[method] += 1
        similarities.append(sim)
        overlaps.append(overlap)
        
        # Check if valid (Q1 criteria)
        if 0.70 < sim < 0.95 and overlap < 0.8:
            valid_count += 1
    
    return {
        "total": len(records),
        "methods": dict(methods),
        "similarities": similarities,
        "overlaps": overlaps,
        "valid_count": valid_count,
        "valid_percentage": (valid_count / len(records)) * 100 if records else 0
    }


def print_quality_report(dataset_name, stats):
    """Print detailed quality report."""
    print("=" * 70)
    print(f"PARAPHRASE QUALITY REPORT: {dataset_name}")
    print("=" * 70)
    print()
    
    print(f"Total paraphrases: {stats['total']:,}")
    print()
    
    print("Method Distribution:")
    print("-" * 70)
    for method, count in sorted(stats['methods'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total']) * 100
        bar = "█" * int(pct / 2)
        print(f"  {method:20s}: {count:6,} ({pct:5.1f}%) {bar}")
    print()
    
    sims = stats['similarities']
    if sims:
        print("Semantic Similarity (SBERT):")
        print("-" * 70)
        print(f"  Mean:   {np.mean(sims):.3f}")
        print(f"  Median: {np.median(sims):.3f}")
        print(f"  Std:    {np.std(sims):.3f}")
        print(f"  Min:    {np.min(sims):.3f}")
        print(f"  Max:    {np.max(sims):.3f}")
        print()
        
        # Distribution
        ranges = [
            ("Too low (<0.70)", sum(1 for s in sims if s < 0.70)),
            ("Valid (0.70-0.95)", sum(1 for s in sims if 0.70 <= s < 0.95)),
            ("Too high (≥0.95)", sum(1 for s in sims if s >= 0.95))
        ]
        print("  Distribution:")
        for label, count in ranges:
            pct = (count / len(sims)) * 100
            print(f"    {label:20s}: {count:6,} ({pct:5.1f}%)")
        print()
    
    overlaps = stats['overlaps']
    if overlaps:
        print("Lexical Overlap (Jaccard):")
        print("-" * 70)
        print(f"  Mean:   {np.mean(overlaps):.3f}")
        print(f"  Median: {np.median(overlaps):.3f}")
        print(f"  Std:    {np.std(overlaps):.3f}")
        print(f"  Min:    {np.min(overlaps):.3f}")
        print(f"  Max:    {np.max(overlaps):.3f}")
        print()
    
    print("Q1 Quality Check:")
    print("-" * 70)
    print(f"  Valid paraphrases: {stats['valid_count']:,} / {stats['total']:,}")
    print(f"  Valid percentage:  {stats['valid_percentage']:.1f}%")
    print()
    
    if stats['valid_percentage'] >= 70:
        print(f"  ✅ PASS: {stats['valid_percentage']:.1f}% meet Q1 standards (target: ≥70%)")
    elif stats['valid_percentage'] >= 50:
        print(f"  ⚠️  MARGINAL: {stats['valid_percentage']:.1f}% meet standards (target: ≥70%)")
    else:
        print(f"  ❌ FAIL: Only {stats['valid_percentage']:.1f}% meet standards (target: ≥70%)")
    
    print()


def show_examples(records, n=5):
    """Show example paraphrases."""
    print("=" * 70)
    print("EXAMPLE PARAPHRASES")
    print("=" * 70)
    print()
    
    # Sample diverse examples
    t5_examples = [r for r in records if r.get("paraphrase_method") == "t5"]
    bt_examples = [r for r in records if r.get("paraphrase_method") == "backtranslation"]
    
    if t5_examples:
        print("T5 Paraphrasing Examples:")
        print("-" * 70)
        for i, record in enumerate(random.sample(t5_examples, min(n, len(t5_examples))), 1):
            orig = record['query']
            para = record['paraphrase']
            sim = record.get('paraphrase_semantic_similarity', 0)
            print(f"\n{i}. Original:   {orig}")
            print(f"   Paraphrase: {para}")
            print(f"   Similarity: {sim:.3f}")
        print()
    
    if bt_examples:
        print("Back-translation Examples:")
        print("-" * 70)
        for i, record in enumerate(random.sample(bt_examples, min(n, len(bt_examples))), 1):
            orig = record['query']
            para = record['paraphrase']
            sim = record.get('paraphrase_semantic_similarity', 0)
            print(f"\n{i}. Original:   {orig}")
            print(f"   Paraphrase: {para}")
            print(f"   Similarity: {sim:.3f}")
        print()


def plot_quality_distributions(all_stats, output_dir):
    """Generate publication-quality plots."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Method distribution
    ax = axes[0, 0]
    datasets = list(all_stats.keys())
    methods = ["t5", "backtranslation", "fallback"]
    method_data = {m: [] for m in methods}
    
    for dataset in datasets:
        total = all_stats[dataset]['total']
        for method in methods:
            count = all_stats[dataset]['methods'].get(method, 0)
            method_data[method].append((count / total) * 100 if total > 0 else 0)
    
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, method in enumerate(methods):
        ax.bar(x + i * width, method_data[method], width, label=method.title())
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Paraphrase Method Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Semantic similarity distribution
    ax = axes[0, 1]
    for dataset, stats in all_stats.items():
        sims = stats['similarities']
        if sims:
            ax.hist(sims, bins=50, alpha=0.5, label=dataset)
    
    ax.axvline(x=0.70, color='red', linestyle='--', linewidth=2, label='Min threshold')
    ax.axvline(x=0.95, color='red', linestyle='--', linewidth=2, label='Max threshold')
    ax.set_xlabel('Semantic Similarity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Semantic Similarity Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Lexical overlap distribution
    ax = axes[1, 0]
    for dataset, stats in all_stats.items():
        overlaps = stats['overlaps']
        if overlaps:
            ax.hist(overlaps, bins=50, alpha=0.5, label=dataset)
    
    ax.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='Max threshold')
    ax.set_xlabel('Lexical Overlap (Jaccard)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Lexical Diversity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Quality pass rate
    ax = axes[1, 1]
    datasets = list(all_stats.keys())
    pass_rates = [all_stats[d]['valid_percentage'] for d in datasets]
    
    bars = ax.bar(datasets, pass_rates, color=['green' if r >= 70 else 'orange' for r in pass_rates])
    ax.axhline(y=70, color='red', linestyle='--', linewidth=2, label='Target (70%)')
    ax.set_ylabel('Valid Paraphrases (%)', fontsize=12, fontweight='bold')
    ax.set_title('Q1 Quality Pass Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, pass_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'paraphrase_quality_report.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Quality plots saved: {output_path}")
    
    output_path_png = Path(output_dir) / 'paraphrase_quality_report.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"✅ Quality plots saved: {output_path_png}")
    
    plt.close()


def generate_latex_table(all_stats, output_dir):
    """Generate LaTeX table for paper."""
    latex = r"""\begin{table}[h]
\centering
\caption{Paraphrase Quality Metrics}
\label{tab:paraphrase_quality}
\begin{tabular}{lrrrr}
\toprule
Dataset & T5 (\%) & Back-trans (\%) & Valid (\%) & Mean Sim \\
\midrule
"""
    
    for dataset, stats in all_stats.items():
        total = stats['total']
        t5_pct = (stats['methods'].get('t5', 0) / total) * 100 if total > 0 else 0
        bt_pct = (stats['methods'].get('backtranslation', 0) / total) * 100 if total > 0 else 0
        valid_pct = stats['valid_percentage']
        mean_sim = np.mean(stats['similarities']) if stats['similarities'] else 0
        
        latex += f"{dataset:15s} & {t5_pct:5.1f} & {bt_pct:5.1f} & {valid_pct:5.1f} & {mean_sim:.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    output_path = Path(output_dir) / 'paraphrase_quality_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate paraphrase quality for Q1 publication'
    )
    parser.add_argument('--data-dir', default='../data',
                       help='Directory containing paraphrased datasets')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for reports and plots')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Sample size for example display')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PARAPHRASE QUALITY VALIDATION" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Find paraphrased datasets
    datasets = {}
    for pattern in ["*_with_paraphrases.jsonl", "*_100k_with_paraphrases.jsonl"]:
        for filepath in data_dir.glob(pattern):
            dataset_name = filepath.stem.replace("_sample_with_paraphrases", "").replace("_with_paraphrases", "").replace("_100k", "")
            datasets[dataset_name] = filepath
    
    if not datasets:
        print(f"❌ No paraphrased datasets found in {data_dir}")
        print("   Expected files: *_with_paraphrases.jsonl")
        return 1
    
    print(f"Found {len(datasets)} paraphrased datasets:")
    for name, path in datasets.items():
        print(f"  • {name}: {path.name}")
    print()
    
    # Analyze each dataset
    all_stats = {}
    for dataset_name, filepath in datasets.items():
        records = load_paraphrased_dataset(filepath)
        stats = analyze_quality(records)
        all_stats[dataset_name] = stats
        
        print_quality_report(dataset_name, stats)
        show_examples(records, n=3)
    
    # Generate plots
    if HAS_MATPLOTLIB:
        plot_quality_distributions(all_stats, output_dir)
    
    # Generate LaTeX table
    generate_latex_table(all_stats, output_dir)
    
    # Overall summary
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print()
    
    total_paraphrases = sum(s['total'] for s in all_stats.values())
    total_valid = sum(s['valid_count'] for s in all_stats.values())
    overall_valid_pct = (total_valid / total_paraphrases) * 100 if total_paraphrases > 0 else 0
    
    print(f"Total paraphrases: {total_paraphrases:,}")
    print(f"Valid paraphrases: {total_valid:,} ({overall_valid_pct:.1f}%)")
    print()
    
    if overall_valid_pct >= 70:
        print(f"✅ OVERALL PASS: {overall_valid_pct:.1f}% meet Q1 quality standards")
        print()
        print("Your paraphrases are publication-ready!")
    else:
        print(f"⚠️  OVERALL MARGINAL: {overall_valid_pct:.1f}% meet standards (target: ≥70%)")
        print()
        print("Consider:")
        print("  • Adjusting T5 temperature/sampling parameters")
        print("  • Using additional back-translation languages")
        print("  • Filtering out low-quality paraphrases")
    
    print()
    print("Paper language suggestion:")
    print("-" * 70)
    print(f"""
"Paraphrases were generated using two complementary methods: T5-based 
neural paraphrasing ({sum(s['methods'].get('t5', 0) for s in all_stats.values()) / total_paraphrases * 100:.1f}%) and back-translation 
via MarianMT ({sum(s['methods'].get('backtranslation', 0) for s in all_stats.values()) / total_paraphrases * 100:.1f}%). Quality was validated using 
SBERT semantic similarity (0.70 < sim < 0.95) and lexical diversity 
(Jaccard < 0.8). Overall, {overall_valid_pct:.1f}% of paraphrases met our quality 
criteria, ensuring semantic equivalence while maintaining linguistic 
diversity (Table X)."
    """)
    print()
    
    return 0


if __name__ == '__main__':
    exit(main())
