#!/bin/bash
# Full Ollama Benchmark (2-4 hours)
# Comprehensive evaluation with multiple seeds and configurations

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Semantic Cache - Full Ollama Benchmark               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}
SEEDS=(42 123 456 789 1024)
DATASETS=(msmarco natural-questions quora-pairs)
THRESHOLDS=(0.85 0.90 0.95)
STRATEGIES=(SEMANTIC EXACT_MATCH)

echo "📋 Configuration:"
echo "   Model: $OLLAMA_MODEL"
echo "   Seeds: ${SEEDS[@]}"
echo "   Datasets: ${DATASETS[@]}"
echo "   Thresholds: ${THRESHOLDS[@]}"
echo "   Strategies: ${STRATEGIES[@]}"
echo "   Estimated time: 2-4 hours"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Create results directory
RESULTS_DIR="results/ollama_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Collect system info
echo "📊 Collecting system information..."
bash scripts/collect_system_info.sh > "$RESULTS_DIR/system_info.txt"

# Total experiments
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]} * ${#THRESHOLDS[@]} * ${#STRATEGIES[@]}))
CURRENT=0

echo ""
echo "🚀 Starting benchmark..."
echo "   Total experiments: $TOTAL"
echo ""

# Run experiments
for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for THRESHOLD in "${THRESHOLDS[@]}"; do
            for STRATEGY in "${STRATEGIES[@]}"; do
                CURRENT=$((CURRENT + 1))
                
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "Experiment $CURRENT/$TOTAL"
                echo "  Dataset: $DATASET | Seed: $SEED | θ: $THRESHOLD | Strategy: $STRATEGY"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                
                mvn spring-boot:run \
                    -q \
                    -Dspring-boot.run.profiles=benchmark \
                    -Dspring-boot.run.jvmArguments="--enable-native-access=ALL-UNNAMED -Xmx4g" \
                    -Dbenchmark.current-dataset="$DATASET" \
                    -Dbenchmark.current-seed="$SEED" \
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

echo ""
echo "✅ Benchmark completed!"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""
echo "📈 Analyze results:"
echo "   cd scripts"
echo "   python3 analyze_results.py ../$RESULTS_DIR"
echo "   python3 visualize_results.py ../$RESULTS_DIR"
echo ""
