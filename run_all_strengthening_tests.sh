#!/bin/bash

# run_all_strengthening_tests.sh
# Master script to execute all Phase 2.5 experiments with enriched data.

RESULTS_DIR="results/20260311_clean"
mkdir -p "$RESULTS_DIR"

echo "=== PHASE 2.5: STRENGTHENING EXPERIMENTS STARTED ==="
echo "Timestamp: $(date)"

# M.6 Gold Standard: Expand TCP connection queues globally for the suite
sudo sysctl -w kern.ipc.somaxconn=1024 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

# 1. Warmup Ablation
echo ">>> Running Warmup Ablation (Proposed vs Legacy)..."
bash run_warmup_ablation.sh

# 2. Hybrid Comparison
echo ">>> Running Hybrid (Cascaded) Caching Comparison..."
bash run_hybrid_comparison.sh

# 3. Throughput & Scalability
echo ">>> Running Throughput & Scalability Tests..."
bash run_throughput_test.sh

echo "=== ALL EXPERIMENTS COMPLETED ==="
echo "Date: $(date)"
python3 scripts/analyze_results.py "$RESULTS_DIR"
