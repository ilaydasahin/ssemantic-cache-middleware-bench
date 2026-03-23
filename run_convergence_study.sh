#!/bin/bash

# run_convergence_study.sh
# Deney 8: Steady-State Convergence Analysis (§5.5)
# Measures how quickly hit rate stabilizes from a cold-start.

RESULTS_DIR="results/20260311_clean/convergence"
mkdir -p "$RESULTS_DIR"

DATASET="msmarco"
# M.6 Gold Standard: Large sequence (N=5000) for smooth convergence plotting
SAMPLE_SIZE=5000 
THRESHOLD=0.90
SEED=42

# M.6 Gold Standard: OS Tuning and ZGC
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"

cleanup_port() {
  lsof -ti:0 | xargs kill -9 2>/dev/null || true
}

echo "=== Deney 8: Steady-State Convergence Analysis ==="
cleanup_port

OUT_FILE="${RESULTS_DIR}/convergence_study.json"

# Note: We use Zipfian skew=0.8 for the most realistic convergence curve
nice -n -10 mvn spring-boot:run -Dspring-boot.run.profiles=benchmark,benchmark-mock \
      -Dlogging.level.root=WARN \
      -Dspring-boot.run.arguments="--server.port=0 \
                                   --benchmark.current-dataset=$DATASET \
                                   --benchmark.current-seed=$SEED \
                                   --benchmark.zipfian-skew=0.8 \
                                   --benchmark.sampleSize=$SAMPLE_SIZE \
                                   --benchmark.warmup-ratio=0.1 \
                                   --benchmark.output-file=$OUT_FILE"

echo "=== Convergence Study Completed — Results in $OUT_FILE ==="
