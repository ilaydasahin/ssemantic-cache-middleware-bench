# Semantic Cache Benchmark

A production-grade semantic caching middleware for LLM API calls in microservice architectures, designed for reproducible academic research.

## Overview

This project implements a semantic cache that uses embedding similarity to serve cached LLM responses for semantically equivalent queries, reducing API costs and latency. The system is built for rigorous benchmarking and evaluation in academic publications.

## Key Features

- **Multiple Lookup Strategies**: Semantic (HNSW/brute-force), Exact-match, Hybrid cascade, Middleware baseline
- **ONNX-based Embeddings**: Local CPU inference with MiniLM, MPNet, and TinyBERT models
- **Thread-safe Session Pooling**: Concurrent ONNX inference without contention
- **Redis 8 Integration**: Native vectorset support (VADD/VSIM) for HNSW-based ANN search
- **Background LFU Eviction**: Fine-grained locking to minimize p99 latency jitter
- **Comprehensive Metrics**: Hit rate, latency percentiles, cost savings, memory usage
- **Reproducible Experiments**: Deterministic seeding, SHA-256 dataset fingerprinting

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  SemanticCacheService                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  CacheLookupStrategy (Strategy Pattern)          │  │
│  │  ├─ SemanticStrategy (L1 exact → HNSW → brute)  │  │
│  │  ├─ HybridCascadeStrategy (MiniLM → MPNet)      │  │
│  │  ├─ ExactMatchStrategy (O(1) hash lookup)       │  │
│  │  └─ MiddlewareBaselineStrategy (15ms overhead)  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Single Data Store: cacheStore + queryIndex            │
│  Background Eviction: LFU with chunked locking         │
└─────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
  OnnxEmbeddingService          RedisSearchService
  (Session Pool)                (Redis 8 Vectorset)
```

## Requirements

- **Java**: 21 (LTS)
- **Maven**: 3.8+
- **Redis**: 8.x (optional, for HNSW mode)
- **Python**: 3.8+ (for dataset preparation and analysis)

## Quick Start

### 0. Setup API Keys (20 Keys for Free Tier)

For free tier experiments with 20 Gemini keys (~29,000 calls/day):

```bash
# Set environment variable with your 20 keys
export GEMINI_API_KEYS="key1,key2,key3,...,key20"

# Or create application-local.yml (gitignored)
cat > src/main/resources/application-local.yml << EOF
llm:
  api-keys: "key1,key2,key3,...,key20"
EOF
```

**See [MULTI_KEY_SETUP.md](MULTI_KEY_SETUP.md) for detailed guide on:**
- Getting 20 free API keys
- Quota management (1,500 RPD per key = 29,000 total)
- Automatic key rotation and failover
- Cost estimation and best practices

### 1. Fetch Embedding Models

```bash
bash scripts/fetch_embedding_assets.sh
```

This downloads ONNX models for MiniLM, MPNet, and TinyBERT.

### 2. Prepare Datasets

```bash
cd scripts
pip install -r requirements.txt
python prepare_datasets.py
```

Generates paraphrased versions of MS MARCO, Natural Questions, and Quora Question Pairs.

### 3. Run Benchmark

```bash
# Single experiment
mvn spring-boot:run -Dspring-boot.run.profiles=benchmark,benchmark-mock \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42 \
  -Dbenchmark.output-file=results/test.json

# Full benchmark suite
bash run_full_benchmark_suite.sh
```

### 4. Analyze Results

```bash
cd scripts
python analyze_results.py ../results/
python visualize_results.py ../results/
```

## Configuration

Key parameters in `src/main/resources/application.yml`:

```yaml
cache:
  similarity-threshold: 0.90    # Cosine similarity threshold (θ)
  hnsw-enabled: true            # Use Redis HNSW vs brute-force
  strategy: SEMANTIC            # SEMANTIC | HYBRID | EXACT_MATCH | MIDDLEWARE_BASELINE
  max-entries: 50000            # Cache capacity
  ttl-seconds: 86400            # Entry TTL (24 hours)

embedding:
  model-name: minilm           # minilm | mpnet | tinybert

benchmark:
  warmup-ratio: 0.30           # Fraction of dataset for cache pre-population
  zipfian-skew: 0.0            # 0.0 = uniform, 1.0 = realistic skew
  noise-probability: 0.0       # Adversarial noise injection [0,1]
  ttl-seconds: 86400           # TTL for benchmark runs
```

## Experiment Scripts

- `run_baseline_comparison.sh` - Compare semantic vs exact-match vs no-cache
- `run_convergence_study.sh` - Threshold sweep (0.80, 0.85, 0.90, 0.95)
- `run_hybrid_comparison.sh` - Hybrid cascade vs single-model strategies
- `run_cache_size_study.sh` - Capacity scaling (1K, 5K, 10K, 50K entries)
- `run_throughput_test.sh` - Concurrent user load (50, 100, 500, 1000)
- `run_eviction_stress_test.sh` - Heavy churn p99 latency validation

## Performance Optimizations

### Thread Safety
- **ONNX Session Pool**: `ArrayBlockingQueue<OrtSession>` prevents inference contention
- **ConcurrentHashMap**: Lock-free reads for `cacheStore` and `queryIndex`
- **Fine-grained Eviction**: Chunked write locks (50 entries/batch) reduce p99 jitter

### Memory Efficiency
- **Streaming SHA-256**: 8KB buffer prevents OOM on large datasets
- **Primitive Arrays**: Latency collection avoids boxing overhead
- **Single Data Store**: Eliminated `LocalVectorIndex` duplication

### Error Handling
- **Redis Metrics**: `redis.store.errors` and `redis.search.errors` counters
- **Full Stack Traces**: Eviction failures logged with context
- **Graceful Degradation**: Redis unavailable → automatic brute-force fallback

## Testing

```bash
# Unit tests
mvn test

# Compile only (no tests)
mvn compile

# Package
mvn package -DskipTests
```

## Project Structure

```
src/main/java/com/semcache/
├── benchmark/          # Experiment orchestration
│   ├── BenchmarkRunner.java
│   ├── MetricsCollector.java
│   ├── DatasetLoader.java
│   └── ExperimentResultExporter.java
├── service/            # Core cache logic
│   ├── SemanticCacheService.java
│   ├── CacheLookupStrategy.java
│   ├── strategy/       # Strategy implementations
│   ├── OnnxEmbeddingService.java
│   └── RedisSearchService.java
├── config/             # Spring configuration
└── model/              # Data transfer objects

scripts/
├── prepare_datasets.py         # Dataset generation
├── analyze_results.py          # Statistical analysis
└── visualize_results.py        # Plotting (Pareto, latency)
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{semantic-cache-2025,
  title={Semantic Caching for LLM APIs: A Production-Grade Middleware Approach},
  author={[Your Name]},
  journal={[Target Journal]},
  year={2025}
}
```

## License

[Specify your license here]

## Reproducibility

All experiments are deterministic given the same:
- Random seed (`benchmark.seeds` in `application.yml`)
- Dataset (verified via SHA-256 fingerprint in logs)
- Configuration parameters (embedded in result JSON files)

Results include full `ExperimentConfig` metadata for independent replication.

## Troubleshooting

### Redis Connection Failed
```
Redis vectorset not available. HNSW phase will be skipped (local-only mode).
```
**Solution**: Start Redis 8.x or set `cache.hnsw-enabled: false` for brute-force mode.

### ONNX Model Not Found
```
Failed to load model minilm: models/all-MiniLM-L6-v2/model.onnx (No such file)
```
**Solution**: Run `bash scripts/fetch_embedding_assets.sh`

### Out of Memory
```
java.lang.OutOfMemoryError: Java heap space
```
**Solution**: Increase heap size: `export MAVEN_OPTS="-Xmx4g"`

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
