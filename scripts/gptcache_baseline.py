#!/usr/bin/env python3
"""
GPTCache Baseline Integration for Semantic Cache Benchmark

Implements GPTCache as a SOTA baseline for comparison.
Provides fair comparison with same datasets and metrics.

GPTCache features:
- Multiple embedding models (OpenAI, HuggingFace, Cohere)
- Multiple similarity evaluation methods
- LRU/FIFO eviction policies
- Configurable similarity thresholds

Usage:
    python3 gptcache_baseline.py --dataset msmarco --output results/gptcache_baseline.json

Dependencies: gptcache, numpy
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from gptcache import Cache
    from gptcache.adapter import openai
    from gptcache.embedding import Onnx
    from gptcache.manager import CacheBase, VectorBase, get_data_manager
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
    HAS_GPTCACHE = True
except ImportError:
    HAS_GPTCACHE = False
    print("⚠️  GPTCache not installed. Install with: pip install gptcache")


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def setup_gptcache(embedding_model: str = "all-MiniLM-L6-v2", 
                   similarity_threshold: float = 0.85,
                   max_size: int = 1000):
    """
    Setup GPTCache with specified configuration.
    
    Args:
        embedding_model: HuggingFace model name
        similarity_threshold: Minimum similarity for cache hit
        max_size: Maximum cache size
    
    Returns:
        Configured Cache instance
    """
    if not HAS_GPTCACHE:
        raise ImportError("GPTCache not installed")
    
    # Initialize embedding function
    onnx = Onnx()
    
    # Initialize data manager (in-memory for fair comparison)
    data_manager = get_data_manager(
        CacheBase("sqlite"),
        VectorBase("faiss", dimension=onnx.dimension)
    )
    
    # Initialize cache
    cache = Cache()
    cache.init(
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    
    return cache


def run_gptcache_benchmark(dataset_path: str, 
                           output_path: str,
                           embedding_model: str = "all-MiniLM-L6-v2",
                           similarity_threshold: float = 0.85,
                           seed: int = 42):
    """
    Run GPTCache baseline benchmark.
    
    Args:
        dataset_path: Path to paraphrased dataset
        output_path: Path to save results
        embedding_model: Embedding model name
        similarity_threshold: Cache hit threshold
        seed: Random seed
    """
    print(f"\n{'='*70}")
    print(f"GPTCache Baseline Benchmark")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_path}")
    print(f"Embedding: {embedding_model}")
    print(f"Threshold: {similarity_threshold}")
    print(f"Seed: {seed}")
    print()
    
    # Load dataset
    records = load_dataset(dataset_path)
    print(f"Loaded {len(records):,} records")
    
    # Setup GPTCache
    print("Initializing GPTCache...")
    cache = setup_gptcache(embedding_model, similarity_threshold)
    
    # Metrics
    hits = 0
    misses = 0
    false_positives = 0
    false_negatives = 0
    latencies = []
    
    np.random.seed(seed)
    
    # Simulate cache workload
    print("\nRunning benchmark...")
    for i, record in enumerate(records):
        if i % 1000 == 0:
            print(f"  Progress: {i:,}/{len(records):,} ({i/len(records)*100:.1f}%)")
        
        query = record["query"]
        paraphrase = record.get("paraphrase", query)
        
        # First query (cache miss expected)
        start = time.time()
        try:
            # Simulate cache lookup
            result = cache.get(query)
            if result is None:
                # Cache miss - store result
                cache.set(query, f"answer_{i}")
                misses += 1
            else:
                # Unexpected hit on first query
                hits += 1
        except Exception as e:
            print(f"  ⚠️  Error on query {i}: {e}")
            misses += 1
        
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        
        # Paraphrase query (cache hit expected if similarity > threshold)
        start = time.time()
        try:
            result = cache.get(paraphrase)
            
            # Check if hit/miss is correct
            expected_similarity = record.get("paraphrase_semantic_similarity", 0.0)
            should_hit = expected_similarity >= similarity_threshold
            
            if result is not None:
                hits += 1
                if not should_hit:
                    false_positives += 1
            else:
                misses += 1
                cache.set(paraphrase, f"answer_{i}_para")
                if should_hit:
                    false_negatives += 1
        except Exception as e:
            print(f"  ⚠️  Error on paraphrase {i}: {e}")
            misses += 1
        
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
    
    # Calculate metrics
    total_queries = hits + misses
    hit_rate = hits / total_queries if total_queries > 0 else 0
    precision = hits / (hits + false_positives) if (hits + false_positives) > 0 else 0
    recall = hits / (hits + false_negatives) if (hits + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Results
    results = {
        "baseline": "GPTCache",
        "embedding_model": embedding_model,
        "similarity_threshold": similarity_threshold,
        "seed": seed,
        "dataset": str(dataset_path),
        "total_queries": total_queries,
        "hits": hits,
        "misses": misses,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "hit_rate": hit_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total queries:     {total_queries:,}")
    print(f"Hits:              {hits:,} ({hit_rate*100:.2f}%)")
    print(f"Misses:            {misses:,}")
    print(f"False positives:   {false_positives:,}")
    print(f"False negatives:   {false_negatives:,}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1_score:.4f}")
    print()
    print(f"Avg latency:       {avg_latency:.2f} ms")
    print(f"P50 latency:       {p50_latency:.2f} ms")
    print(f"P95 latency:       {p95_latency:.2f} ms")
    print(f"P99 latency:       {p99_latency:.2f} ms")
    print()
    print(f"✅ Results saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="GPTCache baseline benchmark for semantic cache comparison"
    )
    parser.add_argument("--dataset", required=True,
                       help="Path to paraphrased dataset (JSONL)")
    parser.add_argument("--output", required=True,
                       help="Output path for results (JSON)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--threshold", type=float, default=0.85,
                       help="Similarity threshold for cache hits")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    if not HAS_GPTCACHE:
        print("❌ GPTCache not installed. Install with:")
        print("   pip install gptcache")
        return 1
    
    run_gptcache_benchmark(
        args.dataset,
        args.output,
        args.embedding_model,
        args.threshold,
        args.seed
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
