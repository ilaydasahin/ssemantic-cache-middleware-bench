#!/bin/bash
# Semantic Cache Benchmark - M.7 Statistical Variance Orchestrator
# Runs the benchmark across multiple independent seeds to ensure statistical significance.

set -e

# Default Parameters
DATASET="msmarco"
OUTPUT_DIR="results/raw"

# M.7: Minimum 5 runs required for significance testing
SEEDS=(42 123 999 1024 2048)
STRATEGIES=("EXACT_MATCH" "SEMANTIC" "HYBRID")

echo "====================================================="
echo " Starting M.7 Statistical Significance Benchmark run "
echo " Dataset: $DATASET "
echo " Strategies: ${STRATEGIES[*]} "
echo " Seeds: ${SEEDS[*]} "
echo "====================================================="

# Clean compile the project once before running
echo "Compiling project..."
mvn clean compile package -DskipTests

mkdir -p "$OUTPUT_DIR"

for seed in "${SEEDS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        output_file="$OUTPUT_DIR/${DATASET}_${strategy}_${seed}.json"
        echo "-----------------------------------------------------"
        echo "Running Strategy: $strategy | Seed: $seed"
        echo "Output -> $output_file"
        
        mvn spring-boot:run -Dspring-boot.run.profiles=benchmark,benchmark-mock -Dspring-boot.run.arguments="--benchmark.current-dataset=$DATASET --benchmark.current-seed=$seed --benchmark.strategy=$strategy --benchmark.output-file=$output_file"
        
        if [ $? -ne 0 ]; then
            echo "Error running experiment for $strategy with seed $seed."
            exit 1
        fi
    done
done

echo "====================================================="
echo " All experiments completed successfully. Results saved to $OUTPUT_DIR/"
echo "====================================================="
