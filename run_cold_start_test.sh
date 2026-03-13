#!/bin/bash

# run_cold_start_test.sh
# Deney 2: Time-to-effectiveness. Maps the Cumulative Hit Rate as the cache goes from empty to warm.

RESULTS_DIR="results/20260311_clean/cold_start"
mkdir -p "$RESULTS_DIR"

# Test all 3 datasets
DATASETS=("msmarco" "natural-questions" "quora-pairs")
MODEL="mpnet"
THRESHOLD=0.90
STRATEGY="SEMANTIC"

# Fixed SEEDS for reproducibility (M.7 Variance)
SEEDS=(42 123 456)

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

echo "=== Deney 2: Cold Start Analysis (Time-to-effectiveness) ==="

mvn clean package -DskipTests

TOTAL_RUNS=${#DATASETS[@]}
CURRENT=0

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # mpnet profile since it's the high-accuracy model
        PROFILE="benchmark,benchmark-mock,mpnet"
        OUT_FILE="${RESULTS_DIR}/coldstart_${dataset}_t${THRESHOLD}_s${seed}.json"
        
        echo "[$CURRENT/$TOTAL_RUNS] Running: Dataset=$dataset (No Warmup)"
        cleanup_port
        
        # M.6 Fix: warmup-ratio=0.0 meaning completely empty cache at start
        mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
            -Dlogging.level.root=WARN \
            -Dspring-boot.run.arguments="--server.port=0 \
                                         --benchmark.current-dataset=$dataset \
                                         --benchmark.current-seed=$seed \
                                         --benchmark.similarity-threshold=$THRESHOLD \
                                         --benchmark.strategy=$STRATEGY \
                                         --benchmark.warmup-ratio=0.0 \
                                         --benchmark.sampleSize=5000 \
                                         --benchmark.output-file=$OUT_FILE"

        if [ $? -eq 0 ]; then
            echo "  ✅ Done"
        else
            echo "  ❌ Failed"
            echo "dataset=$dataset" >> "$FAILED_LOG"
        fi
    done
done

echo "=== Cold Start Study Completed ==="
