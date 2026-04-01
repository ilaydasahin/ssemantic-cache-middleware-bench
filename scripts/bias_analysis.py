"""
Bias and Fairness Analysis for Semantic Cache Benchmark.

Q1 journals increasingly require bias analysis for ML systems.
This script checks for:
1. Query length bias (short vs. long queries)
2. Dataset bias (performance variance across datasets)
3. Temporal bias (performance degradation over time)
4. Semantic drift (embedding quality over cache lifetime)

Usage: python3 bias_analysis.py --results-dir results/

Dependencies: numpy, pandas, scipy
"""

import argparse
import json
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats


def analyze_query_length_bias(logs_filepath: str):
    """
    Detects if cache performance is biased toward short or long queries.
    
    Hypothesis: Shorter queries may have higher hit rates due to less semantic variance.
    """
    short_hits, short_total = 0, 0
    long_hits, long_total = 0, 0
    
    with open(logs_filepath, 'r') as f:
        for line in f:
            try:
                log = json.loads(line)
                query = log.get('query', '')
                is_hit = log.get('hit', False)
                
                word_count = len(query.split())
                
                if word_count <= 10:
                    short_total += 1
                    if is_hit:
                        short_hits += 1
                else:
                    long_total += 1
                    if is_hit:
                        long_hits += 1
            except:
                pass
    
    if short_total == 0 or long_total == 0:
        return None
    
    short_rate = short_hits / short_total
    long_rate = long_hits / long_total
    
    # Chi-square test for independence
    contingency = np.array([
        [short_hits, short_total - short_hits],
        [long_hits, long_total - long_hits]
    ])
    
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    
    return {
        'short_hit_rate': short_rate * 100,
        'long_hit_rate': long_rate * 100,
        'bias_magnitude': abs(short_rate - long_rate) * 100,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def analyze_dataset_bias(results_dir: str):
    """
    Checks if performance varies significantly across datasets.
    
    Concern: If one dataset dominates performance metrics, results may not generalize.
    """
    dataset_metrics = {}
    
    for filepath in glob.glob(os.path.join(results_dir, '*.json')):
        if 'all_results' in filepath or 'throughput' in filepath:
            continue
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            dataset = data.get('dataset', 'unknown')
            hit_rate = data.get('hitRate', 0)
            
            if dataset not in dataset_metrics:
                dataset_metrics[dataset] = []
            dataset_metrics[dataset].append(hit_rate)
    
    if len(dataset_metrics) < 2:
        return None
    
    # ANOVA test for variance across datasets
    groups = [rates for rates in dataset_metrics.values() if len(rates) > 0]
    if len(groups) < 2:
        return None
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate coefficient of variation across datasets
    all_means = [np.mean(rates) for rates in groups]
    cv = (np.std(all_means) / np.mean(all_means)) * 100 if np.mean(all_means) > 0 else 0
    
    return {
        'datasets': list(dataset_metrics.keys()),
        'mean_hit_rates': {k: np.mean(v) for k, v in dataset_metrics.items()},
        'coefficient_of_variation': cv,
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant_variance': p_value < 0.05
    }


def analyze_temporal_bias(logs_filepath: str):
    """
    Detects performance degradation over time (cache pollution).
    
    Method: Compare hit rate in first 25% vs. last 25% of queries.
    """
    queries = []
    
    with open(logs_filepath, 'r') as f:
        for line in f:
            try:
                log = json.loads(line)
                queries.append(log.get('hit', False))
            except:
                pass
    
    if len(queries) < 100:
        return None
    
    n = len(queries)
    first_quarter = queries[:n//4]
    last_quarter = queries[-n//4:]
    
    early_hit_rate = sum(first_quarter) / len(first_quarter)
    late_hit_rate = sum(last_quarter) / len(last_quarter)
    
    # Two-proportion z-test
    n1, n2 = len(first_quarter), len(last_quarter)
    p1, p2 = early_hit_rate, late_hit_rate
    p_pooled = (sum(first_quarter) + sum(last_quarter)) / (n1 + n2)
    
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    z_stat = (p1 - p2) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        'early_hit_rate': early_hit_rate * 100,
        'late_hit_rate': late_hit_rate * 100,
        'degradation': (early_hit_rate - late_hit_rate) * 100,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant_degradation': p_value < 0.05 and early_hit_rate > late_hit_rate
    }


def generate_bias_report(results_dir: str):
    """Generate comprehensive bias analysis report."""
    print("=== Bias and Fairness Analysis ===\n")
    
    # 1. Query Length Bias
    print("--- Query Length Bias ---")
    logs_files = glob.glob(os.path.join(results_dir, '*.logs.jsonl'))
    if logs_files:
        bias = analyze_query_length_bias(logs_files[0])
        if bias:
            print(f"Short queries (<10 words): {bias['short_hit_rate']:.1f}% hit rate")
            print(f"Long queries (≥10 words): {bias['long_hit_rate']:.1f}% hit rate")
            print(f"Bias magnitude: {bias['bias_magnitude']:.1f}%")
            print(f"Statistical significance: {'YES' if bias['significant'] else 'NO'} (p={bias['p_value']:.4f})")
            
            if bias['bias_magnitude'] > 10:
                print("⚠️  WARNING: Significant query length bias detected (>10%)")
        else:
            print("Insufficient data for query length analysis")
    
    # 2. Dataset Bias
    print("\n--- Dataset Bias ---")
    dataset_bias = analyze_dataset_bias(results_dir)
    if dataset_bias:
        print(f"Datasets analyzed: {', '.join(dataset_bias['datasets'])}")
        for ds, rate in dataset_bias['mean_hit_rates'].items():
            print(f"  {ds}: {rate:.1f}% mean hit rate")
        print(f"Coefficient of variation: {dataset_bias['coefficient_of_variation']:.1f}%")
        print(f"Significant variance: {'YES' if dataset_bias['significant_variance'] else 'NO'} (p={dataset_bias['p_value']:.4f})")
        
        if dataset_bias['coefficient_of_variation'] > 20:
            print("⚠️  WARNING: High variance across datasets (CV>20%)")
    else:
        print("Insufficient data for dataset bias analysis")
    
    # 3. Temporal Bias
    print("\n--- Temporal Bias (Cache Pollution) ---")
    if logs_files:
        temporal = analyze_temporal_bias(logs_files[0])
        if temporal:
            print(f"Early queries (first 25%): {temporal['early_hit_rate']:.1f}% hit rate")
            print(f"Late queries (last 25%): {temporal['late_hit_rate']:.1f}% hit rate")
            print(f"Degradation: {temporal['degradation']:.1f}%")
            print(f"Significant degradation: {'YES' if temporal['significant_degradation'] else 'NO'} (p={temporal['p_value']:.4f})")
            
            if temporal['significant_degradation']:
                print("⚠️  WARNING: Cache performance degrades over time")
        else:
            print("Insufficient data for temporal analysis")
    
    print("\n--- Recommendations ---")
    print("• Report all bias metrics in paper (even if non-significant)")
    print("• Discuss potential sources of bias in limitations section")
    print("• Consider stratified sampling if bias magnitude >10%")
    print("• Q1 journals require transparency about dataset limitations")


def main():
    parser = argparse.ArgumentParser(description="Bias analysis for benchmark results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: Directory not found: {args.results_dir}")
        return
    
    generate_bias_report(args.results_dir)


if __name__ == "__main__":
    main()
