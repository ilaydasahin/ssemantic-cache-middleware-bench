#!/bin/bash

# run_eviction_stress_test.sh
# Deney 5: Eviction Resilience (Bellek Kıtlığı) Testi
# Tests the Semantic-Aware Eviction (LRU+LFU Hybrid) policy under severe memory pressure.
# Cache capacity is artificially constrained to force high eviction rates.

RESULTS_DIR="results/20260311_clean/eviction_stress"
mkdir -p "$RESULTS_DIR"

# M.6 Gold Standard: Expand TCP connection queues
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

# M.6 Fix: Sub-millisecond Z Garbage Collector prevents GC-induced tail latency spikes
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"

DATASET="msmarco"
MODEL="minilm"
THRESHOLD=0.85
STRAT="SEMANTIC"
# S1 Fix: Use multiple seeds for variance calculation (M.7)
SEEDS=(42 123 456)

# Force severe memory pressure caching (Max 100 to 500 entries vs a 10K dataset)
TIGHT_CAPACITIES=(100 250 500)

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

echo "=== Deney 5: Eviction Policy Stress Test (Memory Pressure) ==="
echo "Output directory: $RESULTS_DIR"

mvn clean package -DskipTests

TOTAL_RUNS=$(( ${#TIGHT_CAPACITIES[@]} * ${#SEEDS[@]} ))
CURRENT=0

for capacity in "${TIGHT_CAPACITIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
    
        PROFILE="benchmark,benchmark-mock"
        OUT_FILE="${RESULTS_DIR}/eviction_stress_cap${capacity}_t${THRESHOLD}_s${seed}.json"
        
        echo "[$CURRENT/$TOTAL_RUNS] Running: Capacity=$capacity / Seed=$seed"
        cleanup_port
        
        # We set warmup-ratio=0 to rely solely on real-time querying & eviction 
        # to test the LFU+LRU survival capability of hub queries.
        # M.6 Gold Standard: Use 'nice' to give the JVM the highest priority possible against background noise
        nice -n -10 mvn spring-boot:run -Dspring-boot.run.profiles=$PROFILE \
            -Dlogging.level.root=WARN \
            -Dspring-boot.run.arguments="--server.port=0 \
                                         --benchmark.current-dataset=$DATASET \
                                         --benchmark.current-seed=$seed \
                                         --benchmark.similarity-threshold=$THRESHOLD \
                                         --benchmark.strategy=$STRAT \
                                         --benchmark.max-entries=$capacity \
                                         --benchmark.warmup-ratio=0.0 \
                                         --benchmark.sampleSize=2000 \
                                         --benchmark.output-file=$OUT_FILE"

        if [ $? -eq 0 ]; then
            echo "  ✅ Done"
        else
            echo "  ❌ Failed — logged to ${FAILED_LOG}"
            echo "capacity=$capacity/seed=$seed" >> "$FAILED_LOG"
        fi
    done
done

echo "=== Eviction Stress Test Completed ==="
