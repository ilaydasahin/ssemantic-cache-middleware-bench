#!/bin/bash
# ============================================
# Semantic Cache — Altın Standart Reproducibility Script (M.8)
# ============================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results/reproducibility_run"

echo "=== Reproducibility Pipeline Started ==="
mkdir -p "${RESULTS_DIR}"

# 1. Environment Setup Check
if ! command -v mvn &> /dev/null; then
    echo "ERROR: Maven not found. Please install Maven."
    exit 1
fi

# 2. Build & Package
echo "--- Building Artifacts ---"
mvn clean package -DskipTests -q

# 3. Run Controlled Benchmark (MiniLM, MSMarco, Seed 42)
echo "--- Running Controlled Experiments ---"
java -jar target/semantic-cache-benchmark-1.0.0.jar \
    --benchmark.strategy="SEMANTIC" \
    --benchmark.similarityThreshold=0.85 \
    --embedding.model-name="minilm" \
    --benchmark.current-dataset="msmarco" \
    --benchmark.current-seed=42 \
    --benchmark.sampleSize=100 \
    --benchmark.output-file="${RESULTS_DIR}/repro_minilm_t0.85_s42.json" \
    --spring.profiles.active=benchmark,benchmark-mock

# 4. Generate Analysis & Visualizations
echo "--- Generating Scientific Analysis ---"
python3 scripts/analyze_results.py "${RESULTS_DIR}"

echo "=== Reproducibility Pipeline Complete ==="
echo "Results and Plots available in: ${RESULTS_DIR}"
