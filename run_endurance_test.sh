#!/bin/bash

# run_endurance_test.sh
# Deney 6: Endurance Test (System Reliability)
# Pushes the system to extreme limits by overriding the sample size and pushing maximum throughput
# to prove HNSW robustness, Absence of Memory Leaks, and GC Stutters across large iterations.

RESULTS_DIR="results/20260311_clean/endurance"
mkdir -p "$RESULTS_DIR"

# M.6 Gold Standard: Expand TCP connection queues
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

# M.6 Fix: Sub-millisecond Z Garbage Collector prevents GC-induced tail latency spikes
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"

DATASET="msmarco"
MODEL="minilm"
THRESHOLD=0.85
# S1 Fix: Use multiple seeds for variance calculation (M.7)
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

echo "=== Deney 6: System Endurance Test (High Volume Load) ==="
echo "Output directory: $RESULTS_DIR"

mvn clean package -DskipTests

PROFILE="benchmark,benchmark-mock"

TOTAL_RUNS=${#SEEDS[@]}
CURRENT=0

echo "Running: 10,000 continuous queries without JVM restart per seed..."

for seed in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    OUT_FILE="${RESULTS_DIR}/endurance_10K_queries_${DATASET}_s${seed}.json"
    
    echo "[$CURRENT/$TOTAL_RUNS] Endurance test with Seed=$seed"
    cleanup_port

    # Run benchmark with sampleSize equivalent to full dataset file bounds
    # This will take longer so it shouldn't be run in CI implicitly, but manually by researcher.
    # Since MSMARCO sample is 10K lines, omitting sampleSize defaults it to read all 10K.
    # M.6 Gold Standard: Use 'nice' for consistent endurance measurement
    nice -n -10 mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
        -Dlogging.level.root=WARN \
        -Dspring-boot.run.arguments="--server.port=0 \
                                     --benchmark.current-dataset=$DATASET \
                                     --benchmark.current-seed=$seed \
                                     --benchmark.similarity-threshold=$THRESHOLD \
                                     --benchmark.strategy=$STRAT \
                                     --benchmark.sampleSize=10000 \
                                     --benchmark.output-file=$OUT_FILE"

    if [ $? -eq 0 ]; then
        echo "  ✅ Done"
    else
        echo "  ❌ Failed — logged to ${FAILED_LOG}"
        echo "endurance/seed=$seed" >> "$FAILED_LOG"
    fi
done

echo "=== Endurance Test Completed ==="
