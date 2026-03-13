#!/bin/bash

# run_baseline_comparison.sh
# Deney 4: SEMANTIC vs EXACT_MATCH vs NONE (No Cache).
# Measures the HNSW and ONNX overhead against brute-force baseline to prove ROI.

RESULTS_DIR="results/20260311_clean/baselines"
mkdir -p "$RESULTS_DIR"

DATASET="msmarco"
MODEL="minilm"
STRATEGIES=("SEMANTIC" "EXACT_MATCH" "MIDDLEWARE_BASELINE" "NONE")
THRESHOLD=0.90
# S1 Fix: Use multiple seeds for variance calculation (M.7)
SEEDS=(42 123 456 789 101)
export PYTHONHASHSEED=42
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"

FAILED_LOG="${RESULTS_DIR}/failed_runs.log"

cleanup_port() {
  lsof -ti:8181 | xargs kill -9 2>/dev/null || true
  # Wait until port is actually free
  for i in $(seq 1 10); do
    if ! lsof -ti:8181 >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
}

echo "=== Deney 4: HNSW Semantic Cache vs EXACT_MATCH Baseline ==="

mvn clean package -DskipTests

TOTAL_RUNS=$(( ${#STRATEGIES[@]} * ${#SEEDS[@]} ))
CURRENT=0

for strategy in "${STRATEGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
    
        PROFILE="benchmark,benchmark-mock"
        OUT_FILE="${RESULTS_DIR}/baseline_${strategy}_${DATASET}_t${THRESHOLD}_s${seed}.json"
        
        echo "[$CURRENT/$TOTAL_RUNS] Running: Strategy=$strategy / Seed=$seed"
        cleanup_port
        
        # Run benchmark
        mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
            -Dlogging.level.root=WARN \
            -Dspring-boot.run.arguments="--server.port=0 \
                                         --benchmark.current-dataset=$DATASET \
                                         --benchmark.current-seed=$seed \
                                         --benchmark.similarity-threshold=$THRESHOLD \
                                         --benchmark.strategy=$strategy \
                                         --benchmark.sampleSize=2000 \
                                         --benchmark.output-file=$OUT_FILE"

        if [ $? -eq 0 ]; then
            echo "  ✅ Done"
        else
            echo "  ❌ Failed"
            echo "strategy=$strategy/seed=$seed" >> "$FAILED_LOG"
        fi
    done
done

echo "=== Baseline Comparison Completed ==="
