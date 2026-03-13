#!/bin/bash

# run_warmup_ablation.sh
# Compares BIDIRECTIONAL vs UNIDIRECTIONAL warmup strategy.
# Focuses on θ=0.85 (balanced) and θ=0.90 (high precision) for all 3 datasets.

RESULTS_DIR="results/20260311_clean"
mkdir -p "$RESULTS_DIR"

DATASETS=("msmarco" "natural-questions" "quora-pairs")
STRATEGIES=("BIDIRECTIONAL" "UNIDIRECTIONAL")
THRESHOLDS=(0.85 0.90)
MODELS=("minilm" "mpnet")
# Fix #4: Use multiple seeds for statistical significance (Wilcoxon needs ≥5 pairs)
SEEDS=(42 123 456)
# M.6 Gold Standard: Expand TCP connection queues
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

# M.6 Fix: Sub-millisecond Z Garbage Collector prevents GC-induced tail latency spikes
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"
# Fix #8: use sample size to keep runtime feasible (~minutes not hours)
SAMPLE_SIZE=1000

FAILED_LOG="${RESULTS_DIR}/failed_runs.log"  # R3
echo "=== Warmup Ablation Study Started ==="
echo "Output directory: $RESULTS_DIR"

# Rebuild to apply Java changes
mvn clean package -DskipTests

TOTAL_RUNS=$(( ${#DATASETS[@]} * ${#STRATEGIES[@]} * ${#THRESHOLDS[@]} * ${#MODELS[@]} * ${#SEEDS[@]} ))
CURRENT=0

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

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for threshold in "${THRESHOLDS[@]}"; do
            for strategy in "${STRATEGIES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    CURRENT=$((CURRENT + 1))

                    # Setup model profile
                    PROFILE="benchmark,benchmark-mock"
                    if [ "$model" == "mpnet" ]; then
                        PROFILE="benchmark,benchmark-mock,mpnet"
                    fi

                    STRAT_LOWER=$(echo "$strategy" | tr '[:upper:]' '[:lower:]')
                    OUT_FILE="${RESULTS_DIR}/ablation_${STRAT_LOWER}_${dataset}_${model}_t${threshold}_s${seed}.json"

                    echo "[$CURRENT/$TOTAL_RUNS] Running: $dataset / $model / θ=$threshold / Strategy=$strategy / Seed=$seed"
                    cleanup_port

                    # R2 fix: -Dlogging.level.root=WARN keeps Spring app logs, silences Maven output
                    # M.6 Gold Standard: Use 'nice' for consistent warmup ablation measurement
                    nice -n -10 mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
                        -Dlogging.level.root=WARN \
                        -Dspring-boot.run.arguments="--server.port=0 \
                                                     --benchmark.current-dataset=$dataset \
                                                     --benchmark.current-seed=$seed \
                                                     --benchmark.similarity-threshold=$threshold \
                                                     --benchmark.warmup-strategy=$strategy \
                                                     --benchmark.sampleSize=$SAMPLE_SIZE \
                                                     --benchmark.output-file=$OUT_FILE"

                    if [ $? -eq 0 ]; then
                        echo "  ✅ Done"
                    else
                        echo "  ❌ Failed — logged to ${FAILED_LOG}"
                        echo "$dataset/$model/t=$threshold/strategy=$strategy/seed=$seed" >> "$FAILED_LOG"  # R3
                    fi
                done
            done
        done
    done
done

echo "=== Ablation Study Completed ==="
echo "Generating preliminary analysis..."
python3 scripts/analyze_results.py "$RESULTS_DIR"
