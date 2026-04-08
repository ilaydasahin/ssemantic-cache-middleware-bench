#!/bin/bash
#
# Q1 Comprehensive Benchmark with 100K Dataset
#
# Full-scale benchmark using 100K queries per dataset:
# - 26 seeds (80% power for d=0.8)
# - 3 datasets (MS MARCO, NQ, QQP)
# - 3 strategies (SEMANTIC, EXACT_MATCH, GPTCACHE_BASELINE)
# - 3 embedding models (MiniLM, MPNet, TinyBERT)
# - 4 thresholds (0.80, 0.85, 0.90, 0.95)
#
# Total experiments: 26 × 3 × 3 × 3 × 4 = 2,808 runs
# Estimated time: 48-72 hours (2-3 days)
#
# Usage: ./bin/run_q1_comprehensive_benchmark_100k.sh
#

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/q1_comprehensive_100k_${TIMESTAMP}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      Q1 COMPREHENSIVE BENCHMARK (100K DATASET)                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  WARNING: This benchmark will run for 48-72 hours!"
echo ""
echo "Configuration:"
echo "  • Seeds: 26 (80% power for d=0.8)"
echo "  • Datasets: 3 (MS MARCO, NQ, QQP) × 100K queries each"
echo "  • Strategies: 3 (SEMANTIC, EXACT_MATCH, GPTCACHE_BASELINE)"
echo "  • Embedding models: 3 (MiniLM, MPNet, TinyBERT)"
echo "  • Thresholds: 4 (0.80, 0.85, 0.90, 0.95)"
echo "  • Total experiments: 2,808"
echo ""
echo "Press Ctrl+C within 10 seconds to cancel..."
sleep 10

mkdir -p "$RESULTS_DIR"

# Check datasets exist
for DATASET in msmarco_100k nq_100k qqp_100k; do
    if [ ! -f "data/${DATASET}_with_paraphrases.jsonl" ]; then
        echo "❌ Error: data/${DATASET}_with_paraphrases.jsonl not found"
        echo "   Run: ./bin/prepare_100k_datasets.sh"
        exit 1
    fi
done

echo "✅ All 100K datasets found"
echo ""

# Configuration
SEEDS=($(seq 1 26))
DATASETS=(msmarco_100k nq_100k qqp_100k)
STRATEGIES=(SEMANTIC EXACT_MATCH GPTCACHE_BASELINE)
MODELS=(minilm mpnet tinybert)
THRESHOLDS=(0.80 0.85 0.90 0.95)

TOTAL_RUNS=$((26 * 3 * 3 * 3 * 4))
CURRENT_RUN=0

echo "Starting benchmark at: $(date)"
echo ""

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for STRATEGY in "${STRATEGIES[@]}"; do
            for MODEL in "${MODELS[@]}"; do
                for THRESHOLD in "${THRESHOLDS[@]}"; do
                    CURRENT_RUN=$((CURRENT_RUN + 1))
                    PROGRESS=$((CURRENT_RUN * 100 / TOTAL_RUNS))
                    
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "Run $CURRENT_RUN/$TOTAL_RUNS ($PROGRESS%)"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "  Seed: $SEED"
                    echo "  Dataset: $DATASET (100K queries)"
                    echo "  Strategy: $STRATEGY"
                    echo "  Model: $MODEL"
                    echo "  Threshold: $THRESHOLD"
                    echo ""
                    
                    LOG_FILE="$RESULTS_DIR/${DATASET}_${STRATEGY}_${MODEL}_${THRESHOLD}_seed${SEED}.log"
                    
                    mvn spring-boot:run \
                        -Dspring-boot.run.profiles=benchmark \
                        -Dbenchmark.current-dataset=$DATASET \
                        -Dbenchmark.current-seed=$SEED \
                        -Dcache.strategy=$STRATEGY \
                        -Dcache.similarity-threshold=$THRESHOLD \
                        -Dembedding.model-name=$MODEL \
                        > "$LOG_FILE" 2>&1
                    
                    # Copy results
                    if [ -f "results/benchmark_${DATASET}_seed${SEED}_*.json" ]; then
                        cp results/benchmark_${DATASET}_seed${SEED}_*.json "$RESULTS_DIR/"
                    fi
                    
                    echo "✅ Run $CURRENT_RUN complete"
                    echo ""
                done
            done
        done
    done
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BENCHMARK COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Finished at: $(date)"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Validate results:"
echo "     cd scripts && python3 q1_validation_comprehensive.py --results-dir ../$RESULTS_DIR"
echo ""
echo "  2. Statistical analysis:"
echo "     cd scripts && python3 analyze_results.py ../$RESULTS_DIR"
echo ""
echo "  3. Bias analysis:"
echo "     cd scripts && python3 bias_analysis.py --results-dir ../$RESULTS_DIR"
echo ""
echo "  4. Generate figures:"
echo "     cd scripts && python3 generate_publication_figures.py ../$RESULTS_DIR"
echo ""
echo "Paper claim:"
echo "  'We evaluated our approach on 100K queries per dataset across"
echo "   three diverse domains (MS MARCO, Natural Questions, Quora),"
echo "   totaling 300K queries. With 26 independent seeds, our study"
echo "   comprises 7.8M query evaluations, providing robust statistical"
echo "   evidence for our claims.'"
echo ""
