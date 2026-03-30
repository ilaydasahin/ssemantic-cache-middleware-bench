# ⚡ Quick Start Guide

## 3 Steps to Run

### 1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2
```

### 2. Setup Project
```bash
bash scripts/fetch_embedding_assets.sh
cd scripts && pip install -r requirements.txt && python prepare_datasets.py && cd ..
```

### 3. Run Test
```bash
./run_ollama_test.sh
```

## That's it! 🚀

Results will be in `results/ollama_test.json`

## Full Benchmark
```bash
./run_ollama_full_benchmark.sh
```

## Clean Up
```bash
./clean_all.sh
```
