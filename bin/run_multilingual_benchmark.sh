#!/bin/bash
# Multilingual Benchmark - Q1 Publication Requirement
# Tests semantic caching across multiple languages

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Multilingual Benchmark - Q1 Publication          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}
SEEDS=(42 123 456 789 1024)  # 5 seeds for multilingual test
LANGUAGES=(tr de)  # Turkish, German
EMBEDDING_MODEL="multilingual-minilm"  # Multilingual embedding model

echo "📋 Multilingual Configuration:"
echo "   LLM Model: $OLLAMA_MODEL"
echo "   Embedding Model: $EMBEDDING_MODEL"
echo "   Languages: ${LANGUAGES[@]}"
echo "   Seeds: ${SEEDS[@]}"
echo ""

# Check if multilingual datasets exist
echo "🔍 Checking multilingual datasets..."
MISSING_DATASETS=()
for LANG in "${LANGUAGES[@]}"; do
    if [ ! -f "data/${LANG}_sample_with_paraphrases.jsonl" ]; then
        MISSING_DATASETS+=("$LANG")
    fi
done

if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
    echo "❌ Missing datasets for: ${MISSING_DATASETS[@]}"
    echo ""
    echo "📥 Prepare datasets first:"
    echo "   python3 scripts/prepare_multilingual_datasets.py \\"
    echo "     --languages ${MISSING_DATASETS[@]} \\"
    echo "     --sample-size 10000 \\"
    echo "     --with-paraphrases"
    echo ""
    exit 1
fi

echo "✅ All datasets found"
echo ""

# Check if multilingual model exists
if [ ! -d "models/paraphrase-multilingual-MiniLM-L12-v2" ]; then
    echo "❌ Multilingual embedding model not found"
    echo ""
    echo "📥 Download model first:"
    echo "   ./scripts/fetch_multilingual_model.sh"
    echo ""
    exit 1
fi

echo "✅ Multilingual model found"
echo ""

# Create results directory
RESULTS_DIR="results/multilingual_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "🚀 Starting multilingual benchmark..."
echo ""

# Run benchmark for each language
for LANG in "${LANGUAGES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Language: ${LANG^^}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for SEED in "${SEEDS[@]}"; do
        echo "  [${LANG^^}] Seed: $SEED"
        
        mvn spring-boot:run \
            -q \
            -Dspring-boot.run.profiles=benchmark \
            -Dbenchmark.current-dataset="$LANG" \
            -Dbenchmark.current-seed="$SEED" \
            -Dcache.strategy=SEMANTIC \
            -Dembedding.model-name="$EMBEDDING_MODEL" \
            -Dresults.output-dir="$RESULTS_DIR/$LANG"
    done
    
    echo ""
done

echo "✅ Multilingual benchmark completed!"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""
echo "📈 Analyze results:"
echo "   cd scripts"
echo "   python3 analyze_multilingual_results.py ../$RESULTS_DIR"
echo ""
echo "Expected findings:"
echo "  • Hit rates should be similar across languages (±5%)"
echo "  • Latencies should be comparable"
echo "  • Demonstrates generalizability beyond English"
echo ""
