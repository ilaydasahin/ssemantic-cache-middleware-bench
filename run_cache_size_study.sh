#!/bin/bash

# run_cache_size_study.sh
# Deney 1: Evaluates Hit Rate and LLM Cost Savings as a function of Cache Size (Capacity Planning).
# Sweeps --benchmark.max-entries to find the saturation point (Pareto optimality).

RESULTS_DIR="results/20260311_clean/capabilities"
mkdir -p "$RESULTS_DIR"

# Target dataset: Since NLP questions are highly diverse, MSMARCO is a good representative.
DATASET="msmarco"
MODEL="minilm"
THRESHOLD=0.85
THRESHOLD=0.85
STRATEGY="SEMANTIC"
# M.6 Gold Standard: Expand TCP connection queues
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

# M.6 Fix: Sub-millisecond Z Garbage Collector prevents GC-induced tail latency spikes
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"
# S1 Fix: Use multiple seeds for variance calculation (M.7)
SEEDS=(42 123 456)

# Sweep over Cache Capacity (1K to 100K)
CACHE_SIZES=(1000 5000 10000 25000 50000 100000)

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

echo "=== Deney 1: Cache Size Sensitivity Analysis (Capacity Planning) ==="
echo "Output directory: $RESULTS_DIR"

mvn clean package -DskipTests

TOTAL_RUNS=$(( ${#CACHE_SIZES[@]} * ${#SEEDS[@]} ))
CURRENT=0

for size in "${CACHE_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
    
    PROFILE="benchmark,benchmark-mock"
    OUT_FILE="${RESULTS_DIR}/cachesize_${size}_${DATASET}_t${THRESHOLD}_s${SEED}.json"
    
    echo "[$CURRENT/$TOTAL_RUNS] Running: Cache Size=$size"
    cleanup_port
    
        # Run benchmark with explicit max-entries override
        # M.6 Gold Standard: Use 'nice' for consistent cache size scalability measurement
        nice -n -10 mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
            -Dlogging.level.root=WARN \
            -Dspring-boot.run.arguments="--server.port=0 \
                                         --benchmark.current-dataset=$DATASET \
                                         --benchmark.current-seed=$seed \
                                         --benchmark.similarity-threshold=$THRESHOLD \
                                         --benchmark.strategy=SEMANTIC \
                                         --benchmark.max-entries=$size \
                                         --benchmark.sampleSize=2000 \
                                         --benchmark.output-file=$OUT_FILE"

        if [ $? -eq 0 ]; then
            echo "  ✅ Done"
        else
            echo "  ❌ Failed — logged to ${FAILED_LOG}"
            echo "size=$size/seed=$seed" >> "$FAILED_LOG"
        fi
    done
done

echo "=== Cache Size Study Completed ==="
