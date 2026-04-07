#!/bin/bash
# Q1 Comprehensive Benchmark (12-16 hours)
# Full statistical rigor with 26 seeds for d=0.8 at 80% power

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Q1 Publication - Comprehensive Benchmark             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration - 26 seeds for 80% power at d=0.8
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}
SEEDS=(42 123 456 789 1024 2048 3141 4096 5555 6789 7777 8888 9999 10101 11111 12345 13579 14641 15873 16384 17777 18888 19999 20202 21212 22222)
DATASETS=(msmarco natural-questions quora-pairs)
EMBEDDING_MODELS=(minilm mpnet tinybert)
THRESHOLDS=(0.85 0.90 0.95)
STRATEGIES=(SEMANTIC EXACT_MATCH NONE)

echo "📋 Q1 Comprehensive Configuration:"
echo "   LLM Model: $OLLAMA_MODEL"
echo "   Seeds: ${#SEEDS[@]} seeds (26 for 80% power at d=0.8)"
echo "   Datasets: ${DATASETS[@]}"
echo "   Embedding Models: ${EMBEDDING_MODELS[@]}"
echo "   Thresholds: ${THRESHOLDS[@]}"
echo "   Strategies: ${STRATEGIES[@]}"
echo ""

# Calculate total experiments
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]} * ${#EMBEDDING_MODELS[@]} * ${#THRESHOLDS[@]} * ${#STRATEGIES[@]}))

echo "📊 Experiment Statistics:"
echo "   Total experiments: $TOTAL"
echo "   Estimated time: 12-16 hours"
echo "   Estimated storage: ~2-3 GB"
echo ""
echo "⚠️  This is a LONG-RUNNING benchmark"
echo "   Consider running overnight or on a dedicated machine"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Prerequisites check
echo ""
echo "🔍 Checking prerequisites..."

# Check Ollama
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "❌ Ollama not running!"
    echo "   Start: ollama serve &"
    exit 1
fi

# Check model
if ! ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "⚠️  Model not found: $OLLAMA_MODEL"
    echo "   Downloading..."
    ollama pull "$OLLAMA_MODEL"
fi

# Check Python dependencies
if ! python3 -c "import statsmodels, scipy, numpy, pandas" 2>/dev/null; then
    echo "⚠️  Python dependencies missing"
    echo "   Installing..."
    cd scripts
    pip install -r requirements.txt
    cd ..
fi

echo "✅ All prerequisites ready!"
echo ""

# Create results directory
RESULTS_DIR="results/q1_comprehensive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Collect system info
echo "📊 Collecting system information..."
bash scripts/collect_system_info.sh > "$RESULTS_DIR/system_info.txt"

# Run power analysis
echo "📊 Running power analysis..."
cd scripts
python3 power_analysis.py --effect-size 0.8 > "$RESULTS_DIR/power_analysis.txt"
cd ..

# Create experiment log
LOG_FILE="$RESULTS_DIR/experiment.log"
echo "Q1 Comprehensive Benchmark" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Total experiments: $TOTAL" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

CURRENT=0
START_TIME=$(date +%s)

echo ""
echo "🚀 Starting Q1 comprehensive benchmark..."
echo "   Progress will be logged to: $LOG_FILE"
echo ""

# Run experiments
for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for EMB_MODEL in "${EMBEDDING_MODELS[@]}"; do
            for THRESHOLD in "${THRESHOLDS[@]}"; do
                for STRATEGY in "${STRATEGIES[@]}"; do
                    CURRENT=$((CURRENT + 1))
                    
                    # Calculate progress
                    PERCENT=$((CURRENT * 100 / TOTAL))
                    ELAPSED=$(($(date +%s) - START_TIME))
                    if [ $CURRENT -gt 1 ]; then
                        AVG_TIME=$((ELAPSED / (CURRENT - 1)))
                        REMAINING=$(((TOTAL - CURRENT) * AVG_TIME))
                        ETA=$(date -d "@$(($(date +%s) + REMAINING))" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -r $(($(date +%s) + REMAINING)) "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "N/A")
                    else
                        ETA="Calculating..."
                    fi
                    
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "Experiment $CURRENT/$TOTAL ($PERCENT%)"
                    echo "  Dataset: $DATASET | Seed: $SEED | Embedding: $EMB_MODEL"
                    echo "  θ: $THRESHOLD | Strategy: $STRATEGY"
                    echo "  ETA: $ETA"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    
                    # Log experiment
                    echo "[$CURRENT/$TOTAL] $DATASET seed=$SEED emb=$EMB_MODEL θ=$THRESHOLD strategy=$STRATEGY" >> "$LOG_FILE"
                    
                    # Run experiment
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
                        && echo "  ✅ Success" >> "$LOG_FILE" \
                        || echo "  ❌ Failed" >> "$LOG_FILE"
                    
                    echo ""
                done
            done
        done
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "" >> "$LOG_FILE"
echo "Completed: $(date)" >> "$LOG_FILE"
echo "Duration: ${HOURS}h ${MINUTES}m" >> "$LOG_FILE"

echo ""
echo "✅ Q1 comprehensive benchmark completed!"
echo "   Duration: ${HOURS}h ${MINUTES}m"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""
echo "📈 Run statistical analysis:"
echo "   cd scripts"
echo "   python3 analyze_results.py ../$RESULTS_DIR"
echo "   python3 statistical_validation.py --results-dir ../$RESULTS_DIR"
echo "   python3 bias_analysis.py --results-dir ../$RESULTS_DIR"
echo "   python3 visualize_results.py ../$RESULTS_DIR"
echo "   python3 generate_publication_figures.py ../$RESULTS_DIR"
echo ""
echo "✅ Your results are ready for Q1 publication!"
echo ""
