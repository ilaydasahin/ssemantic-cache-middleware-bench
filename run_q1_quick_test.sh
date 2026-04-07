#!/bin/bash
# Q1 Quick Test (30-45 minutes)
# Validates Q1 setup with 3 seeds

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Q1 Publication - Quick Validation Test            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}
SEEDS=(42 123 456)
DATASETS=(msmarco)
EMBEDDING_MODELS=(minilm mpnet)
THRESHOLDS=(0.90)

echo "📋 Q1 Quick Test Configuration:"
echo "   LLM Model: $OLLAMA_MODEL"
echo "   Seeds: ${SEEDS[@]} (3 seeds)"
echo "   Datasets: ${DATASETS[@]}"
echo "   Embedding Models: ${EMBEDDING_MODELS[@]}"
echo "   Thresholds: ${THRESHOLDS[@]}"
echo "   Estimated time: 30-45 minutes"
echo ""
echo "⚠️  This is a QUICK validation test"
echo "   For full Q1 publication, run: ./run_q1_comprehensive_benchmark.sh"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Prerequisites
echo "🔍 Checking prerequisites..."
bash scripts/collect_system_info.sh > /dev/null

# Create results directory
RESULTS_DIR="results/q1_quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Collect system info
bash scripts/collect_system_info.sh > "$RESULTS_DIR/system_info.txt"

# Total experiments
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]} * ${#EMBEDDING_MODELS[@]} * ${#THRESHOLDS[@]} * 2))
CURRENT=0

echo ""
echo "🚀 Starting Q1 quick test..."
echo "   Total experiments: $TOTAL"
echo ""

# Run experiments
for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for EMB_MODEL in "${EMBEDDING_MODELS[@]}"; do
            for THRESHOLD in "${THRESHOLDS[@]}"; do
                for STRATEGY in SEMANTIC EXACT_MATCH; do
                    CURRENT=$((CURRENT + 1))
                    
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "Experiment $CURRENT/$TOTAL"
                    echo "  Dataset: $DATASET | Seed: $SEED | Embedding: $EMB_MODEL"
                    echo "  θ: $THRESHOLD | Strategy: $STRATEGY"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    
                    mvn spring-boot:run \
                        -q \
                        -Dspring-boot.run.profiles=benchmark \
                        -Dspring-boot.run.jvmArguments="--enable-native-access=ALL-UNNAMED -Xmx4g" \
                        -Dbenchmark.current-dataset="$DATASET" \
                        -Dbenchmark.current-seed="$SEED" \
                        -Dembedding.model-name="$EMB_MODEL" \
                        -Dllm.provider=ollama \
                        -Dllm.ollama.model="$OLLAMA_MODEL" \
                        -Dcache.strategy="$STRATEGY" \
                        -Dcache.similarity-threshold="$THRESHOLD" \
                        -Dresults.output-dir="$RESULTS_DIR" \
                        || echo "⚠️  Experiment failed, continuing..."
                    
                    echo ""
                done
            done
        done
    done
done

echo ""
echo "✅ Q1 quick test completed!"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""
echo "📈 Run statistical analysis:"
echo "   cd scripts"
echo "   python3 power_analysis.py --effect-size 0.8"
echo "   python3 analyze_results.py ../$RESULTS_DIR"
echo "   python3 statistical_validation.py --results-dir ../$RESULTS_DIR"
echo "   python3 bias_analysis.py --results-dir ../$RESULTS_DIR"
echo ""
echo "⚠️  Note: 3 seeds is insufficient for Q1 publication"
echo "   Run full benchmark with 26 seeds: ./run_q1_comprehensive_benchmark.sh"
echo ""
