# Semantic Cache Benchmark - Ollama Edition

A production-grade semantic caching middleware for LLM API calls, now with **FREE local Ollama support**!

## Overview

This project implements a semantic cache that uses embedding similarity to serve cached LLM responses for semantically equivalent queries. Now runs completely FREE with local Ollama models - no API keys, no rate limits, no costs!

## Key Features

- **FREE Local LLM**: Ollama integration - no API keys, unlimited queries
- **Multiple Lookup Strategies**: Semantic (HNSW/brute-force), Exact-match, Hybrid cascade
- **ONNX-based Embeddings**: Local CPU inference with MiniLM, MPNet, and TinyBERT models
- **Thread-safe Session Pooling**: Concurrent ONNX inference without contention
- **Redis 8 Integration**: Native vectorset support (VADD/VSIM) for HNSW-based ANN search
- **Comprehensive Metrics**: Hit rate, latency percentiles, cost savings, memory usage
- **16 GB RAM Optimized**: Efficient memory usage for consumer hardware

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

### 1. Install Ollama (5 minutes)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve &

# Download model (Llama 3.2 recommended for 16 GB RAM)
ollama pull llama3.2
```

### 2. Fetch Embedding Models

```bash
bash scripts/fetch_embedding_assets.sh
```

This downloads ONNX models for MiniLM, MPNet, and TinyBERT.

### 3. Prepare Datasets

```bash
cd scripts
pip install -r requirements.txt
python prepare_datasets.py
```

Generates paraphrased versions of MS MARCO, Natural Questions, and Quora Question Pairs.

### 4. Run Quick Test (5-10 minutes)

```bash
./run_ollama_test.sh
```

### 5. Run Full Benchmark (2-4 hours)

```bash
./run_ollama_full_benchmark.sh
```

## Model Options (16 GB RAM)

| Model | Size | RAM | Speed | Quality | Command |
|-------|------|-----|-------|---------|---------|
| Gemma2:2b | 1.6 GB | 2 GB | ⚡⚡⚡ | ⭐⭐ | `ollama pull gemma2:2b` |
| Phi-3 | 2.3 GB | 3 GB | ⚡⚡⚡ | ⭐⭐⭐ | `ollama pull phi3` |
| Llama 3.2 | 2 GB | 4 GB | ⚡⚡ | ⭐⭐⭐⭐ | `ollama pull llama3.2` ✅ |
| Mistral | 4.1 GB | 5 GB | ⚡ | ⭐⭐⭐⭐ | `ollama pull mistral` |

**Recommended:** Llama 3.2 (best balance for 16 GB RAM)

## Configuration

Key parameters in `src/main/resources/application.yml`:

```yaml
llm:
  ollama:
    url: http://localhost:11434
    model: llama3.2  # or phi3, mistral, gemma2:2b
    temperature: 0.0
    max-tokens: 1024

cache:
  similarity-threshold: 0.90
  hnsw-enabled: true
  strategy: SEMANTIC
  max-entries: 10000  # Reduced for 16 GB RAM

embedding:
  model-name: minilm  # or mpnet, tinybert

benchmark:
  warmup-ratio: 0.30
  parallel-threads: 4  # Safe for 16 GB RAM
```

## Available Scripts

```bash
# Quick test (5-10 minutes)
./run_ollama_test.sh

# Full benchmark (2-4 hours)
./run_ollama_full_benchmark.sh

# Clean all old results
./clean_all.sh

# Test with different model
OLLAMA_MODEL=phi3 ./run_ollama_test.sh
```

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

# Compile
mvn compile

# Quick test with Ollama
./run_ollama_test.sh

# Mock test (no LLM needed)
mvn spring-boot:run -Dspring-boot.run.profiles=benchmark,benchmark-mock \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42
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
  year={2025},
  note={Reproducibility artifacts available at [DOI]}
}
```

## Q1 Publication Readiness

This benchmark follows ACM/IEEE reproducibility standards:

### ✅ Completed
- [x] Statistical power analysis (see `scripts/power_analysis.py`)
- [x] Baseline comparisons (no-cache, exact-match, state-of-the-art)
- [x] Multiple testing correction (Benjamini-Hochberg FDR)
- [x] Effect size reporting (Cohen's d)
- [x] Bias analysis (query length, dataset, temporal)
- [x] Reproducibility checklist (see `REPRODUCIBILITY.md`)
- [x] System information logging (see `scripts/collect_system_info.sh`)

### 📋 Pre-submission Checklist
- [ ] Run full benchmark with 5+ seeds per configuration
- [ ] Verify reproducibility score >90/100
- [ ] Generate all figures and tables
- [ ] Complete ethics statement (if using human data)
- [ ] Obtain independent verification (if possible)
- [ ] Archive code and data on Zenodo (DOI)

### 🔬 Validation Commands
```bash
# 1. Validate experimental setup
python3 scripts/validate_experiment.py

# 2. Run power analysis
python3 scripts/power_analysis.py --effect-size 0.5

# 3. Collect system information
bash scripts/collect_system_info.sh

# 4. Run full benchmark
./run_ollama_full_benchmark.sh

# 5. Analyze results with statistical tests
cd scripts
python3 analyze_results.py ../results/

# 6. Check for bias
python3 bias_analysis.py --results-dir ../results/
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

### Ollama Connection Failed
```
❌ Ollama bağlantısı başarısız!
```
**Solution**: Start Ollama with `ollama serve &`

### Model Not Found
```
⚠️ Model bulunamadı: llama3.2
```
**Solution**: Download model with `ollama pull llama3.2`

### Out of Memory
```
java.lang.OutOfMemoryError: Java heap space
```
**Solution**: 
- Use smaller model: `ollama pull gemma2:2b`
- Reduce cache size: `-Dcache.max-entries=5000`
- Increase heap: `export MAVEN_OPTS="-Xmx8g"`

### Redis Connection Failed
```
Redis vectorset not available. HNSW phase will be skipped.
```
**Solution**: Start Redis 8.x or set `cache.hnsw-enabled: false`

## Benefits

- ✅ **FREE**: No API costs, no rate limits
- ✅ **Private**: All data stays local
- ✅ **Unlimited**: No query limits
- ✅ **Offline**: Works without internet (after model download)
- ✅ **16 GB RAM**: Optimized for consumer hardware

## Contact

For questions or issues, please open a GitHub issue.
