#!/bin/bash

# run_pareto_optimality.sh
# Deney 3: Cost vs p99 Latency vs SBERT similarity. (M.4 Pareto Front Analysis).
# Generates data for the Semantic Performance vs Speed scatter plot.

RESULTS_DIR="results/20260311_clean/pareto"
mkdir -p "$RESULTS_DIR"

DATASETS=("natural-questions" "msmarco")
MODELS=("minilm" "mpnet")
# For a pareto front we need multiple operating points (threshold sweeps)
THRESHOLDS=(0.80 0.85 0.90 0.95)
# M.6 Gold Standard: Expand TCP connection queues
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

# S1 Fix: Use multiple seeds for variance calculation (M.7)
SEEDS=(42 123 456)
# M.6 Fix: Sub-millisecond Z Garbage Collector prevents GC-induced tail latency spikes
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

echo "=== Deney 3: Multi-Model Pareto Optimality (Cost vs Latency) ==="

mvn clean package -DskipTests

TOTAL_RUNS=$(( ${#DATASETS[@]} * ${#MODELS[@]} * ${#THRESHOLDS[@]} * ${#SEEDS[@]} ))
CURRENT=0

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for threshold in "${THRESHOLDS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                CURRENT=$((CURRENT + 1))
            
            PROFILE="benchmark,benchmark-mock"
            if [ "$model" == "mpnet" ]; then
                PROFILE="benchmark,benchmark-mock,mpnet"
            fi
            
                OUT_FILE="${RESULTS_DIR}/pareto_${model}_${dataset}_t${threshold}_s${seed}.json"
                
                echo "[$CURRENT/$TOTAL_RUNS] Running: $dataset / Model=$model / θ=$threshold / Seed=$seed"
                cleanup_port
                
                # M.6 Gold Standard: Use 'nice' for consistent Pareto curve measurement
                nice -n -10 mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
                    -Dlogging.level.root=WARN \
                    -Dspring-boot.run.arguments="--server.port=0 \
                                                 --benchmark.current-dataset=$dataset \
                                                 --benchmark.current-seed=$seed \
                                                 --benchmark.similarity-threshold=$threshold \
                                                 --embedding.model-name=$model \
                                                 --benchmark.strategy=SEMANTIC \
                                                 --benchmark.sampleSize=1000 \
                                                 --benchmark.output-file=$OUT_FILE"

                if [ $? -eq 0 ]; then
                    echo "  ✅ Done"
                else
                    echo "  ❌ Failed"
                    echo "$dataset/$model/$threshold/$seed" >> "$FAILED_LOG"
                fi
            done
        done
    done
done

echo "=== Pareto Optimality Study Completed ==="
