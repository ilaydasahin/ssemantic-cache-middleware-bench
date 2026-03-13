#!/bin/bash

# run_hybrid_comparison.sh
# Compares HYBRID (Cascaded) vs SEMANTIC strategy.

RESULTS_DIR="results/20260311_clean"
mkdir -p "$RESULTS_DIR"

DATASETS=("msmarco" "natural-questions" "quora-pairs")
STRATEGIES=("SEMANTIC" "HYBRID")
THRESHOLD=0.85
MODEL="minilm"
# S1 fix: multiple seeds for Wilcoxon statistical significance
SEEDS=(42 123 456)
SAMPLE_SIZE=1000
FAILED_LOG="${RESULTS_DIR}/failed_runs.log"  # R3

cleanup_port() {
  lsof -ti:8181 | xargs kill -9 2>/dev/null || true
}

echo "=== Hybrid Caching Comparison Started ==="

TOTAL_RUNS=$(( ${#DATASETS[@]} * ${#STRATEGIES[@]} * ${#SEEDS[@]} ))
CURRENT=0
mvn clean package -DskipTests -q

for dataset in "${DATASETS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))

            PROFILE="benchmark,benchmark-mock"
            OUT_FILE="${RESULTS_DIR}/hybrid_${strategy}_${dataset}_t${THRESHOLD}_s${seed}.json"

            echo "[$CURRENT/$TOTAL_RUNS] Running: $dataset / Strategy=$strategy / Seed=$seed"

            # R2 fix: use logging.level instead of -q so errors are visible
            mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
                -Dlogging.level.root=WARN \
                -Dspring-boot.run.arguments="--server.port=0 \
                                             --benchmark.current-dataset=$dataset \
                                             --benchmark.current-seed=$seed \
                                             --benchmark.similarity-threshold=$THRESHOLD \
                                             --benchmark.strategy=$strategy \
                                             --benchmark.sampleSize=$SAMPLE_SIZE \
                                             --benchmark.output-file=$OUT_FILE"

            if [ $? -eq 0 ]; then
                echo "  ✅ Done"
            else
                echo "  ❌ Failed — logged to ${FAILED_LOG}"
                echo "$dataset/$strategy/seed=$seed" >> "$FAILED_LOG"  # R3
            fi
        done
    done
done

echo "=== Hybrid Comparison Completed ==="
# Only run analysis if no failures
if [ ! -f "$FAILED_LOG" ] || [ ! -s "$FAILED_LOG" ]; then
    python3 scripts/analyze_results.py "$RESULTS_DIR"
else
    echo "⚠️  Some runs failed. Review ${FAILED_LOG} before running analysis."
fi
