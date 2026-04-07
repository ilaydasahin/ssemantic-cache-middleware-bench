#!/usr/bin/env python3
"""
Result Comparison Tool for Reproducibility Verification

Compares your experimental results with published results to verify reproducibility.

Usage: python3 compare_results.py --your-results results/ --published-results published/ --tolerance 0.05
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON result files from a directory."""
    results = {}
    
    for filepath in glob.glob(f"{results_dir}/*.json"):
        if "all_results" in filepath or "scalability" in filepath:
            continue
        
        with open(filepath) as f:
            data = json.load(f)
        
        # Create key from config
        key = f"{data.get('dataset', 'unknown')}_{data.get('strategy', 'unknown')}_{data.get('embeddingModel', 'unknown')}_{data.get('threshold', 0.9)}"
        
        if key not in results:
            results[key] = []
        results[key].append(data)
    
    return results


def aggregate_results(results: List[Dict]) -> Dict[str, float]:
    """Aggregate results across seeds."""
    import numpy as np
    
    metrics = ['hitRate', 'avgLatencyMs', 'p99LatencyMs', 'costSavingsPercent']
    aggregated = {}
    
    for metric in metrics:
        values = [r.get(metric, 0) for r in results if metric in r]
        if values:
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_n"] = len(values)
    
    return aggregated


def compare_metric(your_value: float, published_value: float, tolerance: float) -> Tuple[bool, float]:
    """Compare a metric with tolerance."""
    if published_value == 0:
        return your_value == 0, 0.0
    
    relative_diff = abs(your_value - published_value) / published_value
    within_tolerance = relative_diff <= tolerance
    
    return within_tolerance, relative_diff


def main():
    parser = argparse.ArgumentParser(description="Compare experimental results")
    parser.add_argument("--your-results", required=True, help="Your results directory")
    parser.add_argument("--published-results", required=True, help="Published results directory")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Tolerance (default: 5%)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 70)
    print()
    
    # Load results
    your_results = load_results(args.your_results)
    published_results = load_results(args.published_results)
    
    print(f"Your results: {len(your_results)} configurations")
    print(f"Published results: {len(published_results)} configurations")
    print(f"Tolerance: ±{args.tolerance * 100}%")
    print()
    
    # Compare each configuration
    all_pass = True
    comparison_count = 0
    
    for config_key in published_results:
        if config_key not in your_results:
            print(f"⚠️  Missing configuration: {config_key}")
            all_pass = False
            continue
        
        your_agg = aggregate_results(your_results[config_key])
        pub_agg = aggregate_results(published_results[config_key])
        
        print(f"\n--- {config_key} ---")
        
        metrics_to_compare = ['hitRate_mean', 'avgLatencyMs_mean', 'p99LatencyMs_mean', 'costSavingsPercent_mean']
        
        for metric in metrics_to_compare:
            if metric not in pub_agg:
                continue
            
            your_val = your_agg.get(metric, 0)
            pub_val = pub_agg[metric]
            
            within_tol, rel_diff = compare_metric(your_val, pub_val, args.tolerance)
            
            status = "✅" if within_tol else "❌"
            
            print(f"  {status} {metric:25s}: {pub_val:8.3f} (published) vs {your_val:8.3f} (yours) | diff: {rel_diff*100:5.2f}%")
            
            if not within_tol:
                all_pass = False
            
            comparison_count += 1
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    if all_pass:
        print(f"✅ ALL {comparison_count} COMPARISONS PASSED")
        print()
        print("Your results are reproducible within the specified tolerance.")
        print("You may proceed with publication.")
        return 0
    else:
        print(f"❌ SOME COMPARISONS FAILED")
        print()
        print("Possible reasons:")
        print("  1. Different hardware (CPU, RAM speed)")
        print("  2. Different software versions (Java, ONNX Runtime)")
        print("  3. Non-deterministic behavior (check random seeds)")
        print("  4. Dataset differences (verify checksums)")
        print()
        print("Recommendations:")
        print("  • Run hardware profiler: python3 scripts/hardware_profiler.py")
        print("  • Verify dataset checksums: bash scripts/verify_checksums.sh")
        print("  • Check dependency versions: mvn dependency:tree")
        print("  • Increase tolerance if hardware differs significantly")
        return 1


if __name__ == "__main__":
    exit(main())
