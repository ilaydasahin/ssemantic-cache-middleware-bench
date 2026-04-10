#!/bin/bash
# Q1+ MEGA BENCHMARK - Robust Medium Effect Detection
# 64 seeds for d=0.5 detection with 80% power (recommended for top-tier journals)

set -e

DATASET="msmarco"
OUTPUT_DIR="results/q1plus_mega"
STRATEGIES=("EXACT_MATCH" "SEMANTIC" "HYBRID" "MIDDLEWARE_BASELINE" "GPTCACHE_BASELINE")

# 64 seeds for 80% power at d=0.5 (robust medium effect detection)
SEEDS=(42 123 999 1024 2048 3141 5926 5358 9793 2384 6264 3383 2795 288 4197 1693 9937 5105 8209 7494 4592 3078 1640 6286 2089 9862 8034 8253 4211 7067 9821 4808 6513 2823 664 7093 8446 955 582 2317 2535 9408 1284 8111 7450 2841 2701 9385 2110 5559 6446 2294 8954 9301 4644 2881 9771 1399 3751 0577 2185 6959 4639 3038 1964)

echo "====================================================="
echo " Q1+ MEGA BENCHMARK (64 seeds, 80% power for d=0.5)"
echo " Dataset: $DATASET"
echo " Strategies: ${STRATEGIES[*]}"
echo " Total runs: $((${#SEEDS[@]} * ${#STRATEGIES[@]}))"
echo " Estimated time: ~8-12 hours"
echo "====================================================="

# Compile once
echo "Compiling project..."
mvn clean compile package -DskipTests

mkdir -p "$OUTPUT_DIR"

total_runs=$((${#SEEDS[@]} * ${#STRATEGIES[@]}))
current_run=0
start_time=$(date +%s)

for seed in "${SEEDS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        current_run=$((current_run + 1))
        output_file="$OUTPUT_DIR/${DATASET}_${strategy}_${seed}.json"
        
        elapsed=$(($(date +%s) - start_time))
        eta=$(( (elapsed * total_runs / current_run) - elapsed ))
        eta_hours=$((eta / 3600))
        eta_mins=$(( (eta % 3600) / 60 ))
        
        echo "-----------------------------------------------------"
        echo "Run $current_run/$total_runs: Strategy=$strategy | Seed=$seed"
        echo "Progress: $(( current_run * 100 / total_runs ))% | ETA: ${eta_hours}h ${eta_mins}m"
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

end_time=$(date +%s)
total_time=$((end_time - start_time))
total_hours=$((total_time / 3600))
total_mins=$(( (total_time % 3600) / 60 ))

echo "====================================================="
echo " ✅ MEGA BENCHMARK COMPLETED!"
echo " Results saved to: $OUTPUT_DIR/"
echo " Total runs: $total_runs"
echo " Total time: ${total_hours}h ${total_mins}m"
echo "====================================================="
echo ""
echo "Next steps:"
echo "  1. Run statistical analysis: python3 scripts/analyze_results.py --input-dir $OUTPUT_DIR"
echo "  2. Generate figures: python3 scripts/generate_publication_figures.py --input-dir $OUTPUT_DIR"
echo "  3. Validate results: python3 scripts/validate_experiment.py --input-dir $OUTPUT_DIR"
echo "  4. Calculate effect sizes: python3 scripts/effect_size_calculator.py --input-dir $OUTPUT_DIR"
