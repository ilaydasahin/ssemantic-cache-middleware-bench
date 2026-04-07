# Project Structure

This document describes the organization of the Semantic Cache Benchmark project.

## Directory Layout

```
semantic-cache-benchmark/
├── bin/                          # Executable scripts
│   ├── run_ollama_test.sh       # Quick test (5-10 min)
│   ├── run_ollama_full_benchmark.sh
│   ├── run_q1_quick_test.sh     # Q1 validation (30-45 min)
│   ├── run_q1_comprehensive_benchmark.sh  # Full Q1 (12-16 hours)
│   ├── run_q1plus_mega_benchmark.sh       # Nature/Science (4-5 days)
│   └── clean_all.sh             # Clean generated files
│
├── data/                         # Benchmark datasets
│   ├── msmarco_sample.jsonl
│   ├── msmarco_sample_with_paraphrases.jsonl
│   ├── nq_sample.jsonl
│   ├── nq_sample_with_paraphrases.jsonl
│   ├── qqp_sample.jsonl
│   └── qqp_sample_with_paraphrases.jsonl
│
├── docs/                         # Documentation
│   ├── CHANGELOG.md             # Version history
│   ├── CITATION.cff             # Citation metadata
│   ├── DOCKER_GUIDE.md          # Docker deployment
│   ├── PROJECT_STRUCTURE.md     # This file
│   ├── PUBLICATION_GUIDE.md     # Q1 publication checklist
│   └── REPRODUCIBILITY.md       # ACM/IEEE reproducibility
│
├── models/                       # ONNX embedding models
│   ├── all-MiniLM-L6-v2/        # 384-dim, fast
│   ├── all-mpnet-base-v2/       # 768-dim, accurate
│   └── paraphrase-TinyBERT-L6-v2/  # 312-dim, compact
│
├── scripts/                      # Analysis and utilities
│   ├── analyze_results.py       # Statistical analysis
│   ├── bias_analysis.py         # Fairness testing
│   ├── collect_system_info.sh   # Hardware specs
│   ├── compare_results.py       # Baseline comparison
│   ├── cross_validation_analysis.py
│   ├── effect_size_calculator.py
│   ├── fetch_embedding_assets.sh
│   ├── generate_figures.py
│   ├── generate_publication_figures.py
│   ├── hardware_profiler.py
│   ├── normality_test.py
│   ├── ollama_watchdog.sh
│   ├── power_analysis.py        # Sample size calculation
│   ├── prepare_datasets.py      # Dataset generation
│   ├── requirements.txt         # Python dependencies
│   ├── run_experiments.sh
│   ├── statistical_validation.py
│   ├── validate_experiment.py
│   ├── verify_checksums.sh
│   ├── visualize_decision_boundary.py
│   └── visualize_results.py
│
├── src/                          # Java source code
│   ├── main/
│   │   ├── java/com/semcache/
│   │   │   ├── benchmark/       # Experiment orchestration
│   │   │   ├── config/          # Spring configuration
│   │   │   ├── controller/      # REST endpoints
│   │   │   ├── model/           # Data models
│   │   │   ├── notification/    # Notification service
│   │   │   └── service/         # Core cache logic
│   │   └── resources/
│   │       └── application.yml  # Configuration
│   └── test/
│       └── java/com/semcache/   # Unit & integration tests
│
├── checkpoints/                  # Experiment checkpoints
├── logs/                         # Application logs
├── monitoring/                   # Prometheus config
├── results/                      # Experiment results
├── secrets/                      # API keys (gitignored)
│
├── .dockerignore
├── .gitignore
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Multi-container setup
├── LICENSE                       # MIT License
├── pom.xml                       # Maven configuration
└── README.md                     # Main documentation
```

## Key Components

### Benchmark Orchestration (`src/main/java/com/semcache/benchmark/`)
- **BenchmarkRunner**: Main experiment coordinator
- **MetricsCollector**: Performance metrics aggregation
- **DatasetLoader**: JSONL dataset parsing with SHA-256 verification
- **ExperimentResultExporter**: JSON result serialization
- **ExperimentValidator**: Pre-flight configuration checks
- **CheckpointManager**: Experiment state persistence

### Cache Service (`src/main/java/com/semcache/service/`)
- **SemanticCacheService**: Main cache coordinator
- **CacheLookupStrategy**: Strategy pattern interface
  - **SemanticStrategy**: L1 exact → HNSW → brute-force
  - **HybridCascadeStrategy**: MiniLM → MPNet fallback
  - **ExactMatchStrategy**: Hash-based O(1) lookup
  - **MiddlewareBaselineStrategy**: 15ms overhead baseline
- **OnnxEmbeddingService**: CPU-based embedding inference
- **RedisSearchService**: Redis 8 vectorset integration
- **CircuitBreaker**: Fault tolerance
- **KeyHealthMonitor**: API key rotation

### LLM Integration (`src/main/java/com/semcache/service/`)
- **LLMService**: Abstract LLM interface
- **OllamaService**: Local Ollama integration (FREE)
- **GeminiService**: Google Gemini API
- **MockGeminiService**: Zero-cost testing

## Configuration Files

### `pom.xml`
Maven project configuration with locked dependency versions:
- Spring Boot 3.5.13
- ONNX Runtime 1.24.3
- Jedis 5.2.0
- JUnit 5.11.4

### `src/main/resources/application.yml`
Runtime configuration:
- Cache parameters (threshold, max-entries, TTL)
- Embedding model selection (minilm, mpnet, tinybert)
- LLM provider (ollama, gemini, mock)
- Benchmark settings (seeds, datasets, parallelism)

### `scripts/requirements.txt`
Python dependencies for analysis:
- numpy, scipy, statsmodels (statistics)
- matplotlib, seaborn, plotly (visualization)
- pandas (data manipulation)
- sentence-transformers (embeddings)

## Data Flow

### Experiment Execution
```
BenchmarkRunner
  ↓
DatasetLoader → load & verify dataset (SHA-256)
  ↓
SemanticCacheService → initialize cache
  ↓
Warmup Phase → pre-populate cache with originals
  ↓
Test Phase → query with paraphrases
  ↓
MetricsCollector → aggregate performance metrics
  ↓
ExperimentResultExporter → save JSON results
```

### Cache Lookup
```
Query → SemanticCacheService
  ↓
CacheLookupStrategy.lookup()
  ↓
L1: Exact match (O(1) hash)
  ↓ (miss)
L2: HNSW search (O(log N))
  ↓ (miss)
L3: Brute-force (O(N))
  ↓ (miss)
LLM fallback → cache result
```

## Build & Test

### Compile
```bash
mvn clean compile
```

### Run Tests
```bash
mvn test
```

### Package
```bash
mvn package
```

### Run Application
```bash
mvn spring-boot:run
```

## Deployment

### Docker
```bash
docker build -t semantic-cache-benchmark .
docker run -p 8080:8080 semantic-cache-benchmark
```

### Docker Compose
```bash
docker-compose up
```

## Results Organization

### Experiment Results (`results/`)
```
results/
├── q1_quick_YYYYMMDD_HHMMSS/
│   ├── *.json                    # Experiment results
│   ├── *.logs.jsonl              # Query logs
│   ├── system_info.txt           # Hardware specs
│   └── power_analysis.txt        # Statistical power
│
├── q1_comprehensive_YYYYMMDD_HHMMSS/
│   ├── *.json                    # 26 seeds × configs
│   ├── *.logs.jsonl
│   ├── figures/                  # Publication figures
│   │   ├── figure1_hit_rate_comparison.*
│   │   ├── figure2_latency_distribution.*
│   │   ├── figure3_pareto_front.*
│   │   ├── figure4_cost_savings.*
│   │   ├── figure5_heatmap.*
│   │   └── figure6_throughput.*
│   └── analysis/                 # Statistical analysis
│       ├── summary_statistics.csv
│       ├── pairwise_tests.csv
│       └── bias_report.txt
│
└── q1plus_mega_YYYYMMDD_HHMMSS/
    └── ...                       # 64 seeds × configs
```

## Naming Conventions

### Java Classes
- **PascalCase**: `SemanticCacheService`, `BenchmarkRunner`
- **Interfaces**: End with `Strategy`, `Service`, or descriptive noun
- **Tests**: End with `Test` (e.g., `MetricsCollectorTest`)

### Python Scripts
- **snake_case**: `analyze_results.py`, `power_analysis.py`
- **Descriptive**: Action + noun (e.g., `generate_figures.py`)

### Shell Scripts
- **snake_case**: `run_q1_quick_test.sh`
- **Prefix**: `run_` for executables, `fetch_` for downloaders

### Configuration Files
- **UPPERCASE**: `README.md`, `LICENSE`, `CHANGELOG.md`
- **lowercase**: `pom.xml`, `docker-compose.yml`

## Version Control

### Branches
- `main`: Stable, production-ready
- `develop`: Integration branch
- `feature/*`: New features
- `fix/*`: Bug fixes

### Tags
- `v1.0.0`: Semantic versioning (MAJOR.MINOR.PATCH)
- `q1-submission`: Publication milestones

## License

MIT License - See [LICENSE](../LICENSE) for details.

## Contact

- **GitHub**: https://github.com/ilaydasahin/semantic-cache-middleware-bench
- **Issues**: https://github.com/ilaydasahin/semantic-cache-middleware-bench/issues
