#!/bin/bash
# Quick GPTCache Baseline Test
# Tests SOTA baseline comparison

set -e

echo "=========================================="
echo "GPTCache SOTA Baseline Test"
echo "=========================================="
echo ""

DATASET="msmarco"
SEEDS=(42 123 456)
STRATEGIES=("SEMANTIC" "EXACT_MATCH" "GPTCACHE_BASELINE" "NONE")

OUTPUT_DIR="results/gptcache_baseline_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Seeds: ${SEEDS[@]}"
echo "Strategies: ${STRATEGIES[@]}"
echo ""

for SEED in "${SEEDS[@]}"; do
  for STRATEGY in "${STRATEGIES[@]}"; do
    echo "----------------------------------------"
    echo "Running: $STRATEGY (seed=$SEED)"
    echo "----------------------------------------"
    
    mvn spring-boot:run \
      -q \
      -Dspring-boot.run.profiles=benchmark \
      -Dbenchmark.current-dataset="$DATASET" \
      -Dbenchmark.current-seed="$SEED" \
      -Dcache.strategy="$STRATEGY" \
      -Dresults.output-dir="$OUTPUT_DIR" \
      2>&1 | grep -E "(Hit Rate|Latency|Cost|Completed)"
    
    echo "✅ Completed: $STRATEGY (seed=$SEED)"
    echo ""
  done
done

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  cd scripts"
echo "  python3 analyze_results.py ../$OUTPUT_DIR"
echo ""
