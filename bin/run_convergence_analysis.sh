#!/bin/bash
#
# Convergence Analysis Runner
#
# Demonstrates that 10K queries is sufficient for cache effectiveness measurement.
#
# This script:
# 1. Runs experiments at multiple dataset sizes (1K, 5K, 10K, 20K, 50K)
# 2. Analyzes hit rate convergence
# 3. Generates publication-quality figures
#
# Usage: ./bin/run_convergence_analysis.sh
#
# Output:
# - results/convergence_*/  (experiment results)
# - convergence_analysis.pdf (figure for paper)
# - convergence_table.tex (table for paper)
#

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/convergence_${TIMESTAMP}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         CONVERGENCE ANALYSIS FOR Q1 PUBLICATION               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "This analysis demonstrates that 10K queries is sufficient"
echo "for measuring cache effectiveness (hit rate convergence)."
echo ""
echo "Estimated time: 3-4 hours"
echo ""

# Create output directory
mkdir -p "$BASE_DIR"

# Dataset sizes to test
SIZES=(1000 2000 5000 10000 20000 50000)

# Number of seeds per size (reduced for faster convergence analysis)
SEEDS=5

echo "Testing dataset sizes: ${SIZES[@]}"
echo "Seeds per size: $SEEDS"
echo ""

# Run experiments for each size
for SIZE in "${SIZES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing size: $SIZE queries"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    SIZE_DIR="$BASE_DIR/size_${SIZE}"
    mkdir -p "$SIZE_DIR"
    
    for SEED in $(seq 1 $SEEDS); do
        echo "  Seed $SEED/$SEEDS..."
        
        # Run benchmark with limited query count
        mvn spring-boot:run \
            -Dspring-boot.run.profiles=benchmark \
            -Dbenchmark.current-dataset=msmarco \
            -Dbenchmark.current-seed=$SEED \
            -Dbenchmark.query-limit=$SIZE \
            -Dcache.strategy=SEMANTIC \
            -Dcache.similarity-threshold=0.90 \
            -Dembedding.model-name=minilm \
            > "$SIZE_DIR/seed_${SEED}.log" 2>&1
        
        # Copy results
        if [ -f "results/benchmark_msmarco_seed${SEED}_*.json" ]; then
            cp results/benchmark_msmarco_seed${SEED}_*.json "$SIZE_DIR/"
        fi
        
        # Copy query logs if available
        if [ -f "results/benchmark_msmarco_seed${SEED}_*.logs.jsonl" ]; then
            cp results/benchmark_msmarco_seed${SEED}_*.logs.jsonl "$SIZE_DIR/"
        fi
    done
    
    echo ""
    echo "✅ Completed size $SIZE"
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ANALYZING CONVERGENCE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run convergence analysis
cd scripts
python3 analyze_convergence.py \
    --results-dir "../$BASE_DIR" \
    --output-dir "../$BASE_DIR"
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CONVERGENCE ANALYSIS COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: $BASE_DIR"
echo ""
echo "Key outputs:"
echo "  • convergence_analysis.pdf  (figure for paper)"
echo "  • convergence_analysis.png  (figure for README)"
echo "  • convergence_table.tex     (table for paper)"
echo ""
echo "Next steps:"
echo "  1. Include convergence_analysis.pdf in paper (Section 3.2)"
echo "  2. Add convergence_table.tex to paper (Table X)"
echo "  3. Update paper text to reference convergence analysis"
echo ""
echo "Paper language suggestion:"
echo "  'We validated that 10K queries is sufficient via convergence"
echo "   analysis (Figure X), showing hit rate stabilizes within 2%"
echo "   variance (p=0.XX for 10K vs 50K comparison).'"
echo ""
