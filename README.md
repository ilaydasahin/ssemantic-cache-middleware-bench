# Semantic Cache Benchmark

High-performance semantic caching middleware for LLM API calls in microservice architectures.

## Quick Start

```bash
# 1. Prepare datasets (100K samples for Q1 publication)
python3 scripts/prepare_datasets.py --sample-size 100000

# 2. Run comprehensive benchmark (26 seeds, 80% power)
./scripts/run_q1_comprehensive_benchmark.sh

# 3. Analyze results
python3 scripts/analyze_results.py results/q1_comprehensive

# 4. Run load test (1000+ RPS validation)
./scripts/run_load_test.sh
```

## Project Structure

```
├── src/                    # Java source code
│   ├── main/java/         # Application code
│   └── test/java/         # Unit tests (80%+ coverage)
├── scripts/               # Python analysis & benchmark scripts
├── data/                  # Datasets (JSONL format)
├── results/               # Experiment results
├── docs/                  # Documentation
└── models/                # ONNX embedding models
```

## Key Features

- **Neural Paraphrasing**: T5 + back-translation with SBERT validation
- **Statistical Rigor**: 26-64 seeds, FDR correction, Cohen's d reporting
- **SOTA Baselines**: GPTCache, exact-match, middleware, no-cache
- **Production Ready**: 1000+ RPS, load tested, Docker support

## Documentation

- [Publication Guide](docs/PUBLICATION_GUIDE.md) - Q1 journal submission checklist
- [Docker Guide](docs/DOCKER_GUIDE.md) - Container deployment
- [Reproducibility](docs/REPRODUCIBILITY.md) - Experiment replication
- [Production Deployment](docs/PRODUCTION_DEPLOYMENT.md) - Scaling guide

## Requirements

- Java 21+
- Python 3.9+
- Maven 3.8+
- Redis 7.0+ (with RedisSearch)

## License

MIT License - see [LICENSE](LICENSE) for details.
