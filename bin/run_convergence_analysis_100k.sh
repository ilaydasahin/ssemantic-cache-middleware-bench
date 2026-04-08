#!/bin/bash
#
# Convergence Analysis with 100K Dataset
#
# Tests dataset sizes: 1K, 5K, 10K, 20K, 50K, 100K
# Demonstrates that hit rate converges well before 100K
#
# Usage: ./bin/run_convergence_analysis_100k.sh
#

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/convergence_100k_${TIMESTAMP}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      CONVERGENCE ANALYSIS WITH 100K DATASET                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "This analysis uses 100K queries to demonstrate convergence."
echo ""
echo "Estimated time: 6-8 hours"
echo ""

mkdir -p "$BASE_DIR"

# Dataset sizes to test (now including 100K)
SIZES=(1000 2000 5000 10000 20000 50000 100000)
SEEDS=5

echo "Testing dataset sizes: ${SIZES[@]}"
echo "Seeds per size: $SEEDS"
echo ""

for SIZE in "${SIZES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing size: $SIZE queries"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    SIZE_DIR="$BASE_DIR/size_${SIZE}"
    mkdir -p "$SIZE_DIR"
    
    for SEED in $(seq 1 $SEEDS); do
        echo "  Seed $SEED/$SEEDS..."
        
        # Use 100K dataset file
        DATASET_FILE="data/msmarco_100k_with_paraphrases.jsonl"
        
        if [ ! -f "$DATASET_FILE" ]; then
            echo "❌ Error: $DATASET_FILE not found"
            echo "   Run: ./bin/prepare_100k_datasets.sh"
            exit 1
        fi
        
        mvn spring-boot:run \
            -Dspring-boot.run.profiles=benchmark \
            -Dbenchmark.current-dataset=msmarco_100k \
            -Dbenchmark.current-seed=$SEED \
            -Dbenchmark.query-limit=$SIZE \
            -Dcache.strategy=SEMANTIC \
            -Dcache.similarity-threshold=0.90 \
            -Dembedding.model-name=minilm \
            > "$SIZE_DIR/seed_${SEED}.log" 2>&1
        
        if [ -f "results/benchmark_msmarco_100k_seed${SEED}_*.json" ]; then
            cp results/benchmark_msmarco_100k_seed${SEED}_*.json "$SIZE_DIR/"
        fi
        
        if [ -f "results/benchmark_msmarco_100k_seed${SEED}_*.logs.jsonl" ]; then
            cp results/benchmark_msmarco_100k_seed${SEED}_*.logs.jsonl "$SIZE_DIR/"
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

cd scripts
python3 analyze_convergence.py \
    --results-dir "../$BASE_DIR" \
    --output-dir "../$BASE_DIR"
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CONVERGENCE ANALYSIS COMPLETE (100K)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: $BASE_DIR"
echo ""
echo "Key findings (expected):"
echo "  • 10K vs 100K: p > 0.05 (no significant difference)"
echo "  • Convergence achieved at: ~10K queries"
echo "  • Variance at 100K: < 1.5%"
echo ""
echo "Paper language:"
echo "  'We validated convergence using datasets up to 100K queries."
echo "   Statistical analysis shows hit rate stabilizes at 10K queries"
echo "   (p=0.XX for 10K vs 100K), with diminishing returns beyond"
echo "   this point. This justifies our use of 10K for controlled"
echo "   experiments while demonstrating scalability to 100K+.'"
echo ""
