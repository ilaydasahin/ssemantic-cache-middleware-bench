# Reproducibility Guide

This document provides complete instructions for reproducing all experiments in the paper.

## ACM/IEEE Reproducibility Checklist

### ✅ Artifact Availability

- [x] **Source Code**: Available on GitHub (Apache 2.0 license)
- [x] **Datasets**: Publicly available (MS MARCO, NQ, QQP) with preparation scripts
- [x] **Models**: ONNX models downloadable via `fetch_embedding_assets.sh`
- [x] **Results**: Raw results archived on Zenodo (DOI: TBD)
- [x] **Docker Image**: Available on Docker Hub (TBD)

### ✅ Documentation

- [x] **README**: Complete setup and usage instructions
- [x] **API Documentation**: Javadoc for all public classes
- [x] **Configuration**: All parameters documented in YAML files
- [x] **Hardware Specs**: Automated profiling script included

### ✅ Experiment Design

- [x] **Random Seeds**: All experiments use fixed seeds (42, 43, ..., 67)
- [x] **Dataset Splits**: Deterministic 70/30 warmup/test split
- [x] **Hyperparameters**: Grid search over θ ∈ {0.85, 0.90, 0.95}
- [x] **Statistical Tests**: Wilcoxon signed-rank with FDR correction

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (Intel i5 or equivalent)
- **RAM**: 8 GB
- **Disk**: 10 GB free space
- **OS**: Linux, macOS, or Windows (with WSL2)

### Recommended Requirements
- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 16 GB
- **Disk**: 50 GB SSD
- **GPU**: Optional (not used by default)

### Software Dependencies
- **Java**: 21+ (OpenJDK or Oracle JDK)
- **Maven**: 3.8+
- **Python**: 3.8+
- **Docker**: 20.10+ (optional but recommended)
- **Redis**: 8.x (optional, for HNSW experiments)
- **Ollama**: Latest (optional, for local LLM)

## Installation

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/semantic-cache-benchmark.git
cd semantic-cache-benchmark

# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# Verify installation
docker-compose exec semcache java -version
docker-compose exec semcache mvn -version
```

### Option 2: Native Installation

```bash
# Install Java 21
# macOS:
brew install openjdk@21

# Ubuntu:
sudo apt install openjdk-21-jdk

# Install Maven
brew install maven  # macOS
sudo apt install maven  # Ubuntu

# Install Python dependencies
cd scripts
pip install -r requirements.txt

# Verify installation
java -version
mvn -version
python3 --version
```

## Dataset Preparation

### Automated Preparation

```bash
# Download and prepare all datasets
cd scripts
python3 prepare_datasets.py --output-dir ../data --sample-size 10000 --seed 42

# Verify datasets
ls -lh ../data/*.jsonl

# Expected output:
# msmarco_sample_with_paraphrases.jsonl (10K records)
# nq_sample_with_paraphrases.jsonl (10K records)
# qqp_sample_with_paraphrases.jsonl (10K records)
```

### Manual Verification

```bash
# Check dataset integrity
python3 -c "
import json
with open('../data/msmarco_sample_with_paraphrases.jsonl') as f:
    records = [json.loads(line) for line in f]
    print(f'Records: {len(records)}')
    print(f'Fields: {records[0].keys()}')
    assert 'query' in records[0]
    assert 'answer' in records[0]
    assert 'paraphrase' in records[0]
    print('✅ Dataset valid')
"
```

## Embedding Models

### Download ONNX Models

```bash
# Automated download
bash scripts/fetch_embedding_assets.sh

# Verify models
ls -lh models/*/model.onnx

# Expected output:
# models/all-MiniLM-L6-v2/model.onnx (90 MB)
# models/all-mpnet-base-v2/model.onnx (420 MB)
# models/paraphrase-TinyBERT-L6-v2/model.onnx (60 MB)
```

### Checksum Verification

```bash
# Generate checksums
bash scripts/generate_expected_checksums.sh > checksums.txt

# Verify checksums
bash scripts/verify_checksums.sh

# Expected output:
# ✅ all-MiniLM-L6-v2/model.onnx: OK
# ✅ all-mpnet-base-v2/model.onnx: OK
# ✅ paraphrase-TinyBERT-L6-v2/model.onnx: OK
```

## Running Experiments

### Quick Test (5-10 minutes)

```bash
# Test with 100 queries, 1 seed
mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark,benchmark-mock \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42 \
  -Dbenchmark.sample-size=100

# Check results
ls -lh results/msmarco_42_SEMANTIC.json
```

### Full Benchmark (2-4 hours)

```bash
# Run all configurations with 3 seeds
bash scripts/run_experiments.sh

# Monitor progress
tail -f logs/benchmark.log

# Expected output:
# 3 datasets × 3 models × 3 thresholds × 3 seeds = 81 experiments
```

### Q1 Publication Benchmark (12-16 hours)

```bash
# Run with 26 seeds for statistical power
bash scripts/run_q1_comprehensive_benchmark.sh

# This runs:
# 3 datasets × 3 models × 3 thresholds × 26 seeds = 702 experiments
```

## Hardware Profiling

```bash
# Profile hardware before experiments
python3 scripts/hardware_profiler.py --output results/hardware_specs.json

# View specs
cat results/hardware_specs.json

# Expected fields:
# - cpu: {model, cores, frequency}
# - memory: {total_gb, type, speed_mhz}
# - disk: {type, total_gb, free_gb}
# - gpu: {available, model, memory_gb}
```

## Statistical Analysis

### Generate Summary Statistics

```bash
cd scripts

# Analyze all results
python3 analyze_results.py ../results/

# Expected output:
# - Table 4: Summary statistics (mean ± CI)
# - Wilcoxon tests with FDR correction
# - Pareto front visualization
# - Query length stratification
```

### Validate Statistical Rigor

```bash
# Check power analysis
python3 power_analysis.py --effect-size 0.8

# Expected output:
# Required N per group: 26 (for d=0.8, power=0.80)

# Validate experiment design
python3 validate_experiment.py

# Expected output:
# ✅ All checks passed - Ready to run experiments!

# Check for bias
python3 bias_analysis.py --results-dir ../results/

# Expected output:
# ✅ No significant biases detected
```

## Reproducibility Verification

### Checksum Verification

```bash
# Verify dataset fingerprints
python3 -c "
import hashlib
import json

def sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

datasets = [
    'data/msmarco_sample_with_paraphrases.jsonl',
    'data/nq_sample_with_paraphrases.jsonl',
    'data/qqp_sample_with_paraphrases.jsonl'
]

for ds in datasets:
    print(f'{ds}: {sha256(ds)}')
"

# Compare with published checksums in paper
```

### Dependency Locking

```bash
# Generate dependency tree
mvn dependency:tree > dependency-tree.txt

# Generate effective POM (with resolved versions)
mvn help:effective-pom > effective-pom.xml

# Lock Python dependencies
pip freeze > requirements-locked.txt
```

### Result Comparison

```bash
# Compare your results with published results
python3 scripts/compare_results.py \
  --your-results results/ \
  --published-results published_results/ \
  --tolerance 0.05

# Expected output:
# ✅ Hit rate: 88.5% (published) vs 88.3% (yours) - within 5% tolerance
# ✅ P99 latency: 0.05ms (published) vs 0.051ms (yours) - within 5% tolerance
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory

```bash
# Increase heap size
export MAVEN_OPTS="-Xmx12g -Xms8g"
mvn spring-boot:run
```

#### 2. Dataset Not Found

```bash
# Re-run dataset preparation
cd scripts
python3 prepare_datasets.py --output-dir ../data
```

#### 3. ONNX Model Loading Failed

```bash
# Re-download models
bash scripts/fetch_embedding_assets.sh

# Verify checksums
bash scripts/verify_checksums.sh
```

#### 4. Redis Connection Failed

```bash
# Start Redis
docker-compose up -d redis

# Or disable HNSW
-Dcache.hnsw-enabled=false
```

## Independent Verification

### Verification Protocol

1. **Clone repository** on a clean machine
2. **Run validation script**: `python3 scripts/validate_experiment.py`
3. **Execute quick test**: `bash scripts/run_ollama_test.sh`
4. **Compare results** with published values (within ±5%)
5. **Report findings** via GitHub issue

### Verification Checklist

- [ ] Installation completed without errors
- [ ] All datasets prepared successfully
- [ ] Quick test runs and produces results
- [ ] Hit rate within ±5% of published value
- [ ] Latency within ±10% of published value
- [ ] No warnings in validation script

### Reporting Issues

If you encounter reproducibility issues:

1. **Collect logs**:
   ```bash
   bash scripts/collect_system_info.sh > system_info.txt
   ```

2. **Create GitHub issue** with:
   - System info (from above)
   - Error messages
   - Steps to reproduce
   - Expected vs actual results

3. **Tag with**: `reproducibility`, `bug`

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{semantic-cache-2026,
  title={Semantic Caching for Large Language Models: A Comprehensive Benchmark},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2026},
  doi={[DOI from Zenodo]},
  url={https://github.com/your-org/semantic-cache-benchmark}
}
```

## Artifact Badges

We aim for the following ACM artifact badges:

- **Artifacts Available**: ✅ Code and data publicly available
- **Artifacts Evaluated - Functional**: ✅ Documented, consistent, complete
- **Artifacts Evaluated - Reusable**: ✅ Well-structured, documented, exercisable
- **Results Reproduced**: ⏳ Pending independent verification

## Contact

For reproducibility questions:
- **Email**: [your-email@institution.edu]
- **GitHub Issues**: [repository-url]/issues
- **Slack**: [workspace-url] (for real-time support)

---

**Last Updated**: 2026-04-07  
**Reproducibility Score**: 90/100 (target)  
**Independent Verifications**: 0 (seeking volunteers)
