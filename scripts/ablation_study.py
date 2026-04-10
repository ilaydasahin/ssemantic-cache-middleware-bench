#!/usr/bin/env python3
"""
Ablation Study for Semantic Cache Components

Tests contribution of each component to overall performance:
1. Embedding model comparison (MiniLM vs MPNet vs TinyBERT)
2. Similarity search (HNSW vs Brute-force)
3. Threshold sensitivity (0.70-0.95)
4. Cache size impact (100-10000 entries)
5. Eviction policy (LRU vs LFU vs FIFO)

Q1 Requirement: Demonstrate which components contribute most to performance

Usage:
    python3 ablation_study.py --output-dir results/ablation
    python3 ablation_study.py --component embedding_model
    python3 ablation_study.py --component similarity_search

Output:
- ablation_results.json: Numerical results
- ablation_report.txt: Human-readable report
- ablation_figures.pdf: Visualization
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List
import numpy as np


def run_benchmark_config(config: Dict, output_file: Path) -> Dict:
    """
    Run benchmark with specific configuration.
    
    Args:
        config: Configuration dict with strategy, model, threshold, etc.
        output_file: Path to save results
    
    Returns:
        Benchmark results dict
    """
    print(f"\n  Running: {config['name']}")
    print(f"    Config: {config}")
    
    # Build Maven command
    cmd = [
        "mvn", "spring-boot:run",
        "-Dspring-boot.run.profiles=benchmark,benchmark-mock",
        f"-Dspring-boot.run.arguments="
        f"--benchmark.current-dataset={config.get('dataset', 'msmarco')} "
        f"--benchmark.current-seed={config.get('seed', 42)} "
        f"--benchmark.strategy={config.get('strategy', 'SEMANTIC')} "
        f"--benchmark.output-file={output_file}"
    ]
    
    # Add component-specific parameters
    if 'embedding_model' in config:
        cmd[-1] += f" --cache.embedding-model={config['embedding_model']}"
    if 'similarity_threshold' in config:
        cmd[-1] += f" --cache.similarity-threshold={config['similarity_threshold']}"
    if 'max_size' in config:
        cmd[-1] += f" --cache.max-size={config['max_size']}"
    if 'eviction_policy' in config:
        cmd[-1] += f" --cache.eviction-policy={config['eviction_policy']}"
    if 'use_hnsw' in config:
        cmd[-1] += f" --cache.use-hnsw={str(config['use_hnsw']).lower()}"
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"    ❌ Failed: {result.stderr}")
            return None
        
        # Load results
        with open(output_file) as f:
            results = json.load(f)
        
        elapsed = time.time() - start_time
        print(f"    ✅ Completed in {elapsed:.1f}s")
        print(f"       Hit rate: {results.get('hit_rate', 0):.3f}")
        print(f"       Avg latency: {results.get('avg_latency_ms', 0):.2f}ms")
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"    ⏱️  Timeout after 600s")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def ablation_embedding_model(output_dir: Path, dataset: str = "msmarco", seed: int = 42) -> List[Dict]:
    """
    Ablation study: Embedding model comparison.
    
    Tests: MiniLM-L6, MPNet-base, TinyBERT-L6
    """
    print("\n" + "=" * 80)
    print("ABLATION: EMBEDDING MODEL")
    print("=" * 80)
    
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-TinyBERT-L6-v2"
    ]
    
    results = []
    
    for model in models:
        config = {
            "name": f"embedding_{model}",
            "component": "embedding_model",
            "embedding_model": model,
            "dataset": dataset,
            "seed": seed,
            "strategy": "SEMANTIC"
        }
        
        output_file = output_dir / f"ablation_embedding_{model}_{seed}.json"
        result = run_benchmark_config(config, output_file)
        
        if result:
            result["config"] = config
            results.append(result)
    
    return results


def ablation_similarity_search(output_dir: Path, dataset: str = "msmarco", seed: int = 42) -> List[Dict]:
    """
    Ablation study: Similarity search algorithm.
    
    Tests: HNSW vs Brute-force
    """
    print("\n" + "=" * 80)
    print("ABLATION: SIMILARITY SEARCH")
    print("=" * 80)
    
    configs = [
        {"name": "HNSW", "use_hnsw": True},
        {"name": "Brute-force", "use_hnsw": False}
    ]
    
    results = []
    
    for cfg in configs:
        config = {
            **cfg,
            "component": "similarity_search",
            "dataset": dataset,
            "seed": seed,
            "strategy": "SEMANTIC"
        }
        
        output_file = output_dir / f"ablation_search_{cfg['name']}_{seed}.json"
        result = run_benchmark_config(config, output_file)
        
        if result:
            result["config"] = config
            results.append(result)
    
    return results


def ablation_threshold_sensitivity(output_dir: Path, dataset: str = "msmarco", seed: int = 42) -> List[Dict]:
    """
    Ablation study: Similarity threshold sensitivity.
    
    Tests: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    """
    print("\n" + "=" * 80)
    print("ABLATION: THRESHOLD SENSITIVITY")
    print("=" * 80)
    
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    results = []
    
    for threshold in thresholds:
        config = {
            "name": f"threshold_{threshold}",
            "component": "threshold",
            "similarity_threshold": threshold,
            "dataset": dataset,
            "seed": seed,
            "strategy": "SEMANTIC"
        }
        
        output_file = output_dir / f"ablation_threshold_{threshold}_{seed}.json"
        result = run_benchmark_config(config, output_file)
        
        if result:
            result["config"] = config
            results.append(result)
    
    return results


def ablation_cache_size(output_dir: Path, dataset: str = "msmarco", seed: int = 42) -> List[Dict]:
    """
    Ablation study: Cache size impact.
    
    Tests: 100, 500, 1000, 5000, 10000
    """
    print("\n" + "=" * 80)
    print("ABLATION: CACHE SIZE")
    print("=" * 80)
    
    sizes = [100, 500, 1000, 5000, 10000]
    
    results = []
    
    for size in sizes:
        config = {
            "name": f"size_{size}",
            "component": "cache_size",
            "max_size": size,
            "dataset": dataset,
            "seed": seed,
            "strategy": "SEMANTIC"
        }
        
        output_file = output_dir / f"ablation_size_{size}_{seed}.json"
        result = run_benchmark_config(config, output_file)
        
        if result:
            result["config"] = config
            results.append(result)
    
    return results


def ablation_eviction_policy(output_dir: Path, dataset: str = "msmarco", seed: int = 42) -> List[Dict]:
    """
    Ablation study: Eviction policy comparison.
    
    Tests: LRU, LFU, FIFO
    """
    print("\n" + "=" * 80)
    print("ABLATION: EVICTION POLICY")
    print("=" * 80)
    
    policies = ["LRU", "LFU", "FIFO"]
    
    results = []
    
    for policy in policies:
        config = {
            "name": f"eviction_{policy}",
            "component": "eviction_policy",
            "eviction_policy": policy,
            "dataset": dataset,
            "seed": seed,
            "strategy": "SEMANTIC"
        }
        
        output_file = output_dir / f"ablation_eviction_{policy}_{seed}.json"
        result = run_benchmark_config(config, output_file)
        
        if result:
            result["config"] = config
            results.append(result)
    
    return results


def analyze_ablation_results(results: List[Dict], component: str) -> Dict:
    """Analyze ablation study results."""
    if not results:
        return {}
    
    # Extract metrics
    configs = [r["config"]["name"] for r in results]
    hit_rates = [r.get("hit_rate", 0) for r in results]
    latencies = [r.get("avg_latency_ms", 0) for r in results]
    
    # Find best configuration
    best_idx = np.argmax(hit_rates)
    worst_idx = np.argmin(hit_rates)
    
    analysis = {
        "component": component,
        "num_configs": len(results),
        "best_config": configs[best_idx],
        "best_hit_rate": hit_rates[best_idx],
        "worst_config": configs[worst_idx],
        "worst_hit_rate": hit_rates[worst_idx],
        "improvement": hit_rates[best_idx] - hit_rates[worst_idx],
        "improvement_pct": ((hit_rates[best_idx] - hit_rates[worst_idx]) / hit_rates[worst_idx] * 100) if hit_rates[worst_idx] > 0 else 0,
        "mean_hit_rate": np.mean(hit_rates),
        "std_hit_rate": np.std(hit_rates),
        "mean_latency": np.mean(latencies),
        "std_latency": np.std(latencies),
        "configs": configs,
        "hit_rates": hit_rates,
        "latencies": latencies
    }
    
    return analysis


def generate_ablation_report(all_results: Dict[str, List[Dict]], output_dir: Path):
    """Generate comprehensive ablation study report."""
    report_path = output_dir / "ablation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY REPORT - Q1 PUBLICATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Purpose: Identify contribution of each component to overall performance\n\n")
        
        for component, results in all_results.items():
            if not results:
                continue
            
            analysis = analyze_ablation_results(results, component)
            
            f.write("=" * 80 + "\n")
            f.write(f"COMPONENT: {component.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Configurations tested: {analysis['num_configs']}\n\n")
            
            f.write("Results:\n")
            f.write("-" * 80 + "\n")
            for i, config in enumerate(analysis['configs']):
                hit_rate = analysis['hit_rates'][i]
                latency = analysis['latencies'][i]
                marker = " ⭐ BEST" if config == analysis['best_config'] else ""
                f.write(f"  {config:30s}: Hit rate = {hit_rate:.3f}, Latency = {latency:.2f}ms{marker}\n")
            f.write("\n")
            
            f.write("Analysis:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Best configuration:    {analysis['best_config']}\n")
            f.write(f"  Best hit rate:         {analysis['best_hit_rate']:.3f}\n")
            f.write(f"  Worst configuration:   {analysis['worst_config']}\n")
            f.write(f"  Worst hit rate:        {analysis['worst_hit_rate']:.3f}\n")
            f.write(f"  Improvement:           {analysis['improvement']:.3f} ({analysis['improvement_pct']:+.1f}%)\n")
            f.write(f"  Mean ± Std:            {analysis['mean_hit_rate']:.3f} ± {analysis['std_hit_rate']:.3f}\n")
            f.write("\n")
            
            # Interpretation
            if analysis['improvement_pct'] > 20:
                impact = "HIGH IMPACT - Critical component"
            elif analysis['improvement_pct'] > 10:
                impact = "MEDIUM IMPACT - Important component"
            elif analysis['improvement_pct'] > 5:
                impact = "LOW IMPACT - Minor component"
            else:
                impact = "NEGLIGIBLE IMPACT - Not critical"
            
            f.write(f"  Impact assessment:     {impact}\n")
            f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS FOR PAPER\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
1. Report ablation results in a table showing each component's contribution
2. Highlight which components are critical vs optional
3. Discuss trade-offs (e.g., accuracy vs latency)
4. Justify your default configuration choices based on ablation results

Example paper language:
"We conducted an ablation study to assess the contribution of each component.
 Results show that [component X] has the highest impact (Δ = +X.XX, +XX%),
 while [component Y] has minimal impact (Δ = +0.0X, +X%). This justifies
 our choice of [configuration] as the default setting."
""")
    
    print(f"\n✅ Ablation report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for semantic cache components"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/ablation"),
                       help="Output directory for results")
    parser.add_argument("--component", choices=["all", "embedding_model", "similarity_search", 
                                                "threshold", "cache_size", "eviction_policy"],
                       default="all",
                       help="Component to test (default: all)")
    parser.add_argument("--dataset", default="msmarco",
                       help="Dataset to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY - COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Component: {args.component}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print()
    
    all_results = {}
    
    # Run ablation studies
    if args.component in ["all", "embedding_model"]:
        results = ablation_embedding_model(args.output_dir, args.dataset, args.seed)
        all_results["embedding_model"] = results
    
    if args.component in ["all", "similarity_search"]:
        results = ablation_similarity_search(args.output_dir, args.dataset, args.seed)
        all_results["similarity_search"] = results
    
    if args.component in ["all", "threshold"]:
        results = ablation_threshold_sensitivity(args.output_dir, args.dataset, args.seed)
        all_results["threshold"] = results
    
    if args.component in ["all", "cache_size"]:
        results = ablation_cache_size(args.output_dir, args.dataset, args.seed)
        all_results["cache_size"] = results
    
    if args.component in ["all", "eviction_policy"]:
        results = ablation_eviction_policy(args.output_dir, args.dataset, args.seed)
        all_results["eviction_policy"] = results
    
    # Save all results
    results_path = args.output_dir / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All results saved to: {results_path}")
    
    # Generate report
    generate_ablation_report(all_results, args.output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ABLATION STUDY COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review ablation_report.txt for component contributions")
    print("  2. Generate figures: python3 scripts/generate_ablation_figures.py")
    print("  3. Include ablation table in paper methods section")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
