#!/usr/bin/env python3
"""
Ablation Study Analysis for Q1 Publication

Analyzes component-wise contributions:
1. HNSW vs Brute-Force
2. Embedding Model Comparison
3. Threshold Sensitivity
4. Strategy Comparison

Usage: python3 analyze_ablation_study.py ../results/ablation_study_*
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(results_dir: Path) -> dict:
    """Load all ablation study results."""
    results = {
        "hnsw": {"enabled": [], "disabled": []},
        "embedding": {"minilm": [], "mpnet": [], "tinybert": []},
        "threshold": {},
        "strategy": {}
    }
    
    for json_file in results_dir.rglob("*.json"):
        if "all_results" in json_file.name or "scalability" in json_file.name:
            continue
        
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # HNSW ablation
            if "hnsw_enabled" in str(json_file.parent):
                results["hnsw"]["enabled"].append(data)
            elif "hnsw_disabled" in str(json_file.parent):
                results["hnsw"]["disabled"].append(data)
            
            # Embedding ablation
            for emb in ["minilm", "mpnet", "tinybert"]:
                if f"embedding_{emb}" in str(json_file.parent):
                    results["embedding"][emb].append(data)
            
            # Threshold ablation
            for threshold in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
                threshold_str = f"{threshold:.2f}"
                if f"threshold_{threshold_str}" in str(json_file.parent):
                    if threshold_str not in results["threshold"]:
                        results["threshold"][threshold_str] = []
                    results["threshold"][threshold_str].append(data)
            
            # Strategy ablation
            for strategy in ["SEMANTIC", "HYBRID", "EXACT_MATCH", "GPTCACHE_BASELINE", "NONE"]:
                if f"strategy_{strategy}" in str(json_file.parent):
                    if strategy not in results["strategy"]:
                        results["strategy"][strategy] = []
                    results["strategy"][strategy].append(data)
        
        except Exception as e:
            print(f"⚠️  Failed to load {json_file}: {e}")
    
    return results


def analyze_hnsw_impact(results: dict, output_dir: Path):
    """Analyze HNSW vs Brute-Force impact."""
    print("\n" + "=" * 80)
    print("ABLATION 1: HNSW vs Brute-Force")
    print("=" * 80)
    
    if not results["hnsw"]["enabled"] or not results["hnsw"]["disabled"]:
        print("⚠️  Insufficient data for HNSW ablation")
        return
    
    # Extract metrics
    hnsw_enabled_latency = [r["p99LatencyMs"] for r in results["hnsw"]["enabled"]]
    hnsw_disabled_latency = [r["p99LatencyMs"] for r in results["hnsw"]["disabled"]]
    
    hnsw_enabled_hit_rate = [r["hitRate"] for r in results["hnsw"]["enabled"]]
    hnsw_disabled_hit_rate = [r["hitRate"] for r in results["hnsw"]["disabled"]]
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(hnsw_enabled_latency, hnsw_disabled_latency)
    cohens_d = (np.mean(hnsw_enabled_latency) - np.mean(hnsw_disabled_latency)) / \
               np.sqrt((np.std(hnsw_enabled_latency)**2 + np.std(hnsw_disabled_latency)**2) / 2)
    
    print(f"\nP99 Latency:")
    print(f"  HNSW Enabled:  {np.mean(hnsw_enabled_latency):.2f} ± {np.std(hnsw_enabled_latency):.2f} ms")
    print(f"  HNSW Disabled: {np.mean(hnsw_disabled_latency):.2f} ± {np.std(hnsw_disabled_latency):.2f} ms")
    print(f"  Improvement: {((np.mean(hnsw_disabled_latency) - np.mean(hnsw_enabled_latency)) / np.mean(hnsw_disabled_latency) * 100):.1f}%")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.2f}")
    
    print(f"\nHit Rate:")
    print(f"  HNSW Enabled:  {np.mean(hnsw_enabled_hit_rate):.1f}%")
    print(f"  HNSW Disabled: {np.mean(hnsw_disabled_hit_rate):.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Latency comparison
    axes[0].boxplot([hnsw_enabled_latency, hnsw_disabled_latency],
                    labels=["HNSW\nEnabled", "HNSW\nDisabled"])
    axes[0].set_ylabel("P99 Latency (ms)")
    axes[0].set_title("HNSW Impact on Latency")
    axes[0].grid(True, alpha=0.3)
    
    # Hit rate comparison
    axes[1].bar(["HNSW\nEnabled", "HNSW\nDisabled"],
                [np.mean(hnsw_enabled_hit_rate), np.mean(hnsw_disabled_hit_rate)],
                yerr=[np.std(hnsw_enabled_hit_rate), np.std(hnsw_disabled_hit_rate)],
                capsize=5)
    axes[1].set_ylabel("Hit Rate (%)")
    axes[1].set_title("HNSW Impact on Hit Rate")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_hnsw.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {output_dir / 'ablation_hnsw.png'}")


def analyze_embedding_impact(results: dict, output_dir: Path):
    """Analyze embedding model comparison."""
    print("\n" + "=" * 80)
    print("ABLATION 2: Embedding Model Comparison")
    print("=" * 80)
    
    models = ["minilm", "mpnet", "tinybert"]
    metrics = {"hit_rate": [], "latency": [], "model": []}
    
    for model in models:
        if not results["embedding"][model]:
            continue
        
        hit_rates = [r["hitRate"] for r in results["embedding"][model]]
        latencies = [r["p99LatencyMs"] for r in results["embedding"][model]]
        
        print(f"\n{model.upper()}:")
        print(f"  Hit Rate: {np.mean(hit_rates):.1f} ± {np.std(hit_rates):.1f}%")
        print(f"  P99 Latency: {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")
        
        metrics["hit_rate"].extend(hit_rates)
        metrics["latency"].extend(latencies)
        metrics["model"].extend([model.upper()] * len(hit_rates))
    
    # ANOVA test
    if len(models) == 3:
        f_stat, p_value = stats.f_oneway(
            [r["hitRate"] for r in results["embedding"]["minilm"]],
            [r["hitRate"] for r in results["embedding"]["mpnet"]],
            [r["hitRate"] for r in results["embedding"]["tinybert"]]
        )
        print(f"\nANOVA (Hit Rate): F={f_stat:.2f}, p={p_value:.4f}")
    
    # Visualization
    df = pd.DataFrame(metrics)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.boxplot(data=df, x="model", y="hit_rate", ax=axes[0])
    axes[0].set_ylabel("Hit Rate (%)")
    axes[0].set_xlabel("Embedding Model")
    axes[0].set_title("Embedding Model Impact on Hit Rate")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    sns.boxplot(data=df, x="model", y="latency", ax=axes[1])
    axes[1].set_ylabel("P99 Latency (ms)")
    axes[1].set_xlabel("Embedding Model")
    axes[1].set_title("Embedding Model Impact on Latency")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_embedding.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {output_dir / 'ablation_embedding.png'}")


def analyze_threshold_sensitivity(results: dict, output_dir: Path):
    """Analyze threshold sensitivity."""
    print("\n" + "=" * 80)
    print("ABLATION 3: Threshold Sensitivity")
    print("=" * 80)
    
    thresholds = sorted(results["threshold"].keys())
    hit_rates_mean = []
    hit_rates_std = []
    latencies_mean = []
    latencies_std = []
    
    for threshold in thresholds:
        if not results["threshold"][threshold]:
            continue
        
        hit_rates = [r["hitRate"] for r in results["threshold"][threshold]]
        latencies = [r["p99LatencyMs"] for r in results["threshold"][threshold]]
        
        hit_rates_mean.append(np.mean(hit_rates))
        hit_rates_std.append(np.std(hit_rates))
        latencies_mean.append(np.mean(latencies))
        latencies_std.append(np.std(latencies))
        
        print(f"\nθ = {threshold}:")
        print(f"  Hit Rate: {np.mean(hit_rates):.1f} ± {np.std(hit_rates):.1f}%")
        print(f"  P99 Latency: {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    thresholds_float = [float(t) for t in thresholds]
    
    axes[0].errorbar(thresholds_float, hit_rates_mean, yerr=hit_rates_std,
                     marker='o', capsize=5, linewidth=2)
    axes[0].set_xlabel("Similarity Threshold (θ)")
    axes[0].set_ylabel("Hit Rate (%)")
    axes[0].set_title("Threshold Sensitivity: Hit Rate")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].errorbar(thresholds_float, latencies_mean, yerr=latencies_std,
                     marker='o', capsize=5, linewidth=2, color='orange')
    axes[1].set_xlabel("Similarity Threshold (θ)")
    axes[1].set_ylabel("P99 Latency (ms)")
    axes[1].set_title("Threshold Sensitivity: Latency")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_threshold.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {output_dir / 'ablation_threshold.png'}")


def analyze_strategy_comparison(results: dict, output_dir: Path):
    """Analyze strategy comparison."""
    print("\n" + "=" * 80)
    print("ABLATION 4: Strategy Comparison")
    print("=" * 80)
    
    strategies = ["SEMANTIC", "HYBRID", "EXACT_MATCH", "GPTCACHE_BASELINE", "NONE"]
    metrics = {"hit_rate": [], "latency": [], "strategy": []}
    
    for strategy in strategies:
        if strategy not in results["strategy"] or not results["strategy"][strategy]:
            continue
        
        hit_rates = [r["hitRate"] for r in results["strategy"][strategy]]
        latencies = [r["p99LatencyMs"] for r in results["strategy"][strategy]]
        
        print(f"\n{strategy}:")
        print(f"  Hit Rate: {np.mean(hit_rates):.1f} ± {np.std(hit_rates):.1f}%")
        print(f"  P99 Latency: {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")
        
        metrics["hit_rate"].extend(hit_rates)
        metrics["latency"].extend(latencies)
        metrics["strategy"].extend([strategy] * len(hit_rates))
    
    # Visualization
    df = pd.DataFrame(metrics)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=df, x="strategy", y="hit_rate", ax=axes[0], ci="sd")
    axes[0].set_ylabel("Hit Rate (%)")
    axes[0].set_xlabel("Strategy")
    axes[0].set_title("Strategy Comparison: Hit Rate")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    sns.barplot(data=df, x="strategy", y="latency", ax=axes[1], ci="sd")
    axes[1].set_ylabel("P99 Latency (ms)")
    axes[1].set_xlabel("Strategy")
    axes[1].set_title("Strategy Comparison: Latency")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_strategy.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {output_dir / 'ablation_strategy.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("results_dir", help="Directory containing ablation study results")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        return 1
    
    print("=" * 80)
    print(" " * 25 + "ABLATION STUDY ANALYSIS")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    
    # Load results
    results = load_results(results_dir)
    
    # Create output directory for figures
    output_dir = results_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Run analyses
    analyze_hnsw_impact(results, output_dir)
    analyze_embedding_impact(results, output_dir)
    analyze_threshold_sensitivity(results, output_dir)
    analyze_strategy_comparison(results, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ABLATION STUDY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {output_dir}")
    print("\nUse these results in your paper:")
    print("  • Table 5: Ablation study results")
    print("  • Figure 7: Ablation study bar charts")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
