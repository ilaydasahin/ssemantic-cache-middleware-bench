#!/bin/bash
# Q1 Publication Benchmark - Comprehensive Statistical Power
# 26 seeds for d=0.8 detection with 80% power (minimum acceptable)

set -e

DATASET="msmarco"
OUTPUT_DIR="results/q1_comprehensive"
STRATEGIES=("EXACT_MATCH" "SEMANTIC" "HYBRID" "MIDDLEWARE_BASELINE" "GPTCACHE_BASELINE")

# 26 seeds for 80% power at d=0.8 (Q1 minimum)
SEEDS=(42 123 999 1024 2048 3141 5926 5358 9793 2384 6264 3383 2795 288 4197 1693 9937 5105 8209 7494 4592 3078 1640 6286 2089 9862)

echo "====================================================="
echo " Q1 COMPREHENSIVE BENCHMARK (26 seeds, 80% power)"
echo " Dataset: $DATASET"
echo " Strategies: ${STRATEGIES[*]}"
echo " Total runs: $((${#SEEDS[@]} * ${#STRATEGIES[@]}))"
echo "====================================================="

# Compile once
echo "Compiling project..."
mvn clean compile package -DskipTests

mkdir -p "$OUTPUT_DIR"

total_runs=$((${#SEEDS[@]} * ${#STRATEGIES[@]}))
current_run=0

for seed in "${SEEDS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        current_run=$((current_run + 1))
        output_file="$OUTPUT_DIR/${DATASET}_${strategy}_${seed}.json"
        
        echo "-----------------------------------------------------"
        echo "Run $current_run/$total_runs: Strategy=$strategy | Seed=$seed"
        echo "Output -> $output_file"
        
        mvn spring-boot:run \
            -Dspring-boot.run.profiles=benchmark,benchmark-mock \
            -Dspring-boot.run.arguments="--benchmark.current-dataset=$DATASET --benchmark.current-seed=$seed --benchmark.strategy=$strategy --benchmark.output-file=$output_file"
        
        if [ $? -ne 0 ]; then
            echo "❌ Error running experiment for $strategy with seed $seed."
            exit 1
        fi
    done
done

echo "====================================================="
echo " ✅ All experiments completed successfully!"
echo " Results saved to: $OUTPUT_DIR/"
echo " Total runs: $total_runs"
echo "====================================================="
echo ""
echo "Next steps:"
echo "  1. Run statistical analysis: python3 scripts/analyze_results.py --input-dir $OUTPUT_DIR"
echo "  2. Generate figures: python3 scripts/generate_publication_figures.py --input-dir $OUTPUT_DIR"
echo "  3. Validate results: python3 scripts/validate_experiment.py --input-dir $OUTPUT_DIR"
