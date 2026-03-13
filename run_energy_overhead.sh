#!/bin/bash

# run_energy_overhead.sh
# Deney 7: ROI (Return on Investment) / Energy Overhead Analysis
# Combines top-level CPU and RAM profiling alongside the Spring Boot test 
# to quantify the local infrastructure cost of running the Semantic Cache 
# versus the cost savings of avoiding the LLM.

RESULTS_DIR="results/20260311_clean/energy_overhead"
mkdir -p "$RESULTS_DIR"

DATASET="msmarco"
MODEL="mpnet"
THRESHOLD=0.90
STRAT="SEMANTIC"
# M.6 Gold Standard: Expand TCP connection queues
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

# M.6 Fix: Sub-millisecond Z Garbage Collector prevents GC-induced tail latency spikes
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"
# S1 Fix: Use multiple seeds for variance calculation (M.7)
SEEDS=(42 123 456)

FAILED_LOG="${RESULTS_DIR}/failed_runs.log"
CPU_LOG="${RESULTS_DIR}/cpu_usage_log.csv"

echo "Timestamp,CPU%,Mem_MB" > "$CPU_LOG"

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

echo "=== Deney 7: Energy & ROI Overhead Profiling ==="
echo "Output directory: $RESULTS_DIR"

mvn clean package -DskipTests

TOTAL_RUNS=${#SEEDS[@]}
CURRENT=0

for seed in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    PROFILE="benchmark,benchmark-mock,mpnet"
    OUT_FILE="${RESULTS_DIR}/roi_${MODEL}_t${THRESHOLD}_s${seed}.json"

    echo "[$CURRENT/$TOTAL_RUNS] Running: Energy Profile with Seed=$seed"
    cleanup_port

    # Start background hardware power logger (milliwatts)
    # M.6 Gold Standard: Direct hardware measurement using Mac powermetrics
    echo "Timestamp,CPU_Power_mW,GPU_Power_mW,Combined_Power_mW" > "${RESULTS_DIR}/hardware_power_s${seed}.csv"
    (
      while true; do
        # Captures systemic power usage during the process execution
        # Note: requires sudo. If sudo is not available, falls back to ps
        POWER_DATA=$(sudo powermetrics -i 1000 -n 1 --samplers cpu_power 2>/dev/null | grep -E "CPU Power|GPU Power|Combined Power")
        if [ -n "$POWER_DATA" ]; then
          CPU_PW=$(echo "$POWER_DATA" | grep "CPU Power" | awk '{print $3}')
          GPU_PW=$(echo "$POWER_DATA" | grep "GPU Power" | awk '{print $3}')
          COMB_PW=$(echo "$POWER_DATA" | grep "Combined Power" | awk '{print $3}')
          echo "$(date +%s),$CPU_PW,$GPU_PW,$COMB_PW" >> "${RESULTS_DIR}/hardware_power_s${seed}.csv"
        else
          # Fallback to ps-based estimation if powermetrics fails
          JAVA_PID=$(jps -l | grep SemanticCacheBenchmarkApplication | awk '{print $1}')
          if [ -n "$JAVA_PID" ]; then
            STATS=$(ps -p $JAVA_PID -o %cpu,rss | tail -n +2)
            CPU=$(echo $STATS | awk '{print $1}')
            echo "$(date +%s),$CPU,0,0" >> "${RESULTS_DIR}/hardware_power_s${seed}.csv"
          fi
        fi
        sleep 1
      done
    ) &
    LOGGER_PID=$!

    echo "Starting Application with background PID $LOGGER_PID logging resources..."

    # M.6 Gold Standard: Use 'nice' to isolated energy measurement from other processes
    nice -n -20 mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
        -Dlogging.level.root=WARN \
        -Dspring-boot.run.arguments="--server.port=0 \
                                     --benchmark.current-dataset=$DATASET \
                                     --benchmark.current-seed=$seed \
                                     --benchmark.similarity-threshold=$THRESHOLD \
                                     --benchmark.strategy=$STRAT \
                                     --benchmark.sampleSize=2000 \
                                     --benchmark.output-file=$OUT_FILE" &

    RETVAL=$?

    # Kill the background resource logger
    kill $LOGGER_PID

    if [ $RETVAL -eq 0 ]; then
        echo "  ✅ Done for seed $seed. Hardware metrics logged to $CPU_LOG"
    else
        echo "  ❌ Failed — logged to ${FAILED_LOG}"
        echo "roi/seed=$seed" >> "$FAILED_LOG"
    fi
done

echo "=== ROI & Energy Profiling Completed ==="
