#!/bin/bash
# Quick Ollama Test (5-10 minutes)
# Tests semantic cache with local Ollama LLM

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Semantic Cache - Quick Ollama Test                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Ollama
echo "🔍 Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found!"
    echo "   Install: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "❌ Ollama not running!"
    echo "   Start: ollama serve &"
    exit 1
fi

# Check model
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}
echo "🔍 Checking model: $OLLAMA_MODEL..."
if ! ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "⚠️  Model not found: $OLLAMA_MODEL"
    echo "   Downloading..."
    ollama pull "$OLLAMA_MODEL"
fi

# Check embedding models
echo "🔍 Checking embedding models..."
if [ ! -f "models/all-MiniLM-L6-v2/model.onnx" ]; then
    echo "⚠️  Embedding models not found"
    echo "   Fetching..."
    bash scripts/fetch_embedding_assets.sh
fi

# Check datasets
echo "🔍 Checking datasets..."
if [ ! -f "data/msmarco_sample_with_paraphrases.jsonl" ]; then
    echo "⚠️  Datasets not prepared"
    echo "   Preparing..."
    cd scripts
    pip install -q -r requirements.txt
    python prepare_datasets.py
    cd ..
fi

echo ""
echo "✅ All prerequisites ready!"
echo ""
echo "🚀 Starting quick test..."
echo "   Dataset: msmarco (sample)"
echo "   Seeds: 1 (seed=42)"
echo "   Model: $OLLAMA_MODEL"
echo "   Duration: ~5-10 minutes"
echo ""

# Create results directory
mkdir -p results/ollama_quick_test
RESULTS_DIR="results/ollama_quick_test/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Run benchmark
mvn spring-boot:run \
    -Dspring-boot.run.profiles=benchmark \
    -Dspring-boot.run.jvmArguments="--enable-native-access=ALL-UNNAMED -Xmx4g" \
    -Dbenchmark.current-dataset=msmarco \
    -Dbenchmark.current-seed=42 \
    -Dbenchmark.pilot-sample-size=100 \
    -Dllm.provider=ollama \
    -Dllm.ollama.model="$OLLAMA_MODEL" \
    -Dcache.strategy=SEMANTIC \
    -Dcache.similarity-threshold=0.90 \
    -Dresults.output-dir="$RESULTS_DIR"

echo ""
echo "✅ Test completed!"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""
echo "📈 View results:"
echo "   cat $RESULTS_DIR/*.json | jq '.hitRate, .p99Latency, .costSavings'"
echo ""
echo "🔬 Next steps:"
echo "   • Full benchmark: ./run_ollama_full_benchmark.sh"
echo "   • Q1 benchmark: ./run_q1_comprehensive_benchmark.sh"
echo ""
