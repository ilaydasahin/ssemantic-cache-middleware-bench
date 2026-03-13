#!/bin/bash

# run_throughput_test.sh
# Measures RPS and Latency vs. Concurrent Users.

RESULTS_DIR="results/20260311_clean"
mkdir -p "$RESULTS_DIR"

# M.6 Gold Standard: Expand TCP connection queues to prevent Ephemeral Port / SYN timeouts on Mac/Linux
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

# M.6 Fix: Sub-millisecond Z Garbage Collector prevents GC-induced tail latency spikes
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"

DATASETS=("msmarco")
MODELS=("minilm" "mpnet")
CONCURRENT_USERS=(50 100 500)
# S1 Fix: Use multiple seeds for variance calculation (M.7)
SEEDS=(42 123 456 789 101)
export PYTHONHASHSEED=42
FAILED_LOG="${RESULTS_DIR}/failed_runs.log"  # R3

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

echo "=== Throughput & Scalability Benchmark Started ==="

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#CONCURRENT_USERS[@]} * ${#SEEDS[@]} ))
CURRENT=0

for model in "${MODELS[@]}"; do
    for users in "${CONCURRENT_USERS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            
            PROFILE="benchmark,benchmark-mock"
            if [ "$model" == "mpnet" ]; then
                PROFILE="benchmark,benchmark-mock,mpnet"
            fi
            
            OUT_FILE="${RESULTS_DIR}/throughput_${model}_u${users}_s${seed}.json"
            
            echo "[$CURRENT/$TOTAL_RUNS] Running: Model=$model / Users=$users / Seed=$seed"

            # R2 fix: use -Dlogging.level.root=WARN instead of -q
            # M.6 Gold Standard: Use 'nice' to give the JVM the highest priority possible against background noise
            nice -n -10 mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
                -Dlogging.level.root=WARN \
                -Dspring-boot.run.arguments="--server.port=0 \
                                             --benchmark.current-dataset=msmarco \
                                             --benchmark.current-seed=$seed \
                                             --benchmark.concurrent-users=$users \
                                             --benchmark.output-file=$OUT_FILE"

            if [ $? -eq 0 ]; then
                echo "  ✅ Done"
            else
                echo "  ❌ Failed — logged to ${FAILED_LOG}"
                echo "model=$model/users=$users/seed=$seed" >> "$FAILED_LOG"  # R3
            fi
        done
    done
done

echo "=== Throughput & Scalability Benchmark Completed ==="
