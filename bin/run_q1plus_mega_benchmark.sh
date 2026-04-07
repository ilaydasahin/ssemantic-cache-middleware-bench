#!/bin/bash
# Q1+ MEGA Benchmark (4-5 DAYS)
# Nature/Science level with 64 seeds for d=0.5 at 80% power

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Q1+ MEGA BENCHMARK - Nature/Science Level            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration - 64 seeds for 80% power at d=0.5
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}
SEEDS=($(seq 1 64))
DATASETS=(msmarco natural-questions quora-pairs)
EMBEDDING_MODELS=(minilm mpnet tinybert)
THRESHOLDS=(0.80 0.85 0.90 0.95)
STRATEGIES=(SEMANTIC EXACT_MATCH NONE)

echo "📋 Q1+ MEGA Configuration:"
echo "   LLM Model: $OLLAMA_MODEL"
echo "   Seeds: 64 seeds (for 80% power at d=0.5)"
echo "   Datasets: ${DATASETS[@]}"
echo "   Embedding Models: ${EMBEDDING_MODELS[@]}"
echo "   Thresholds: ${THRESHOLDS[@]}"
echo "   Strategies: ${STRATEGIES[@]}"
echo ""

# Calculate total experiments
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]} * ${#EMBEDDING_MODELS[@]} * ${#THRESHOLDS[@]} * ${#STRATEGIES[@]}))

echo "📊 Experiment Statistics:"
echo "   Total experiments: $TOTAL"
echo "   Estimated time: 4-5 DAYS"
echo "   Estimated storage: ~10-15 GB"
echo ""
echo "⚠️  WARNING: This is an EXTREMELY LONG benchmark"
echo "   Recommended for:"
echo "   • Nature/Science submissions"
echo "   • PhD dissertations"
echo "   • Dedicated compute clusters"
echo ""
echo "   NOT recommended for:"
echo "   • Regular Q1 journals (use run_q1_comprehensive_benchmark.sh)"
echo "   • Laptops (use dedicated server)"
echo "   • Quick validation (use run_q1_quick_test.sh)"
echo ""

read -p "Are you ABSOLUTELY SURE you want to run MEGA benchmark? (yes/no) " -r
echo
if [[ ! $REPLY == "yes" ]]; then
    echo "Cancelled. Use run_q1_comprehensive_benchmark.sh for regular Q1 journals."
    exit 0
fi

# Prerequisites check
echo ""
echo "🔍 Checking prerequisites..."

if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "❌ Ollama not running!"
    exit 1
fi

if ! ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "⚠️  Model not found: $OLLAMA_MODEL"
    ollama pull "$OLLAMA_MODEL"
fi

echo "✅ All prerequisites ready!"
echo ""

# Create results directory
RESULTS_DIR="results/q1plus_mega_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Collect system info
echo "📊 Collecting system information..."
bash scripts/collect_system_info.sh > "$RESULTS_DIR/system_info.txt"

# Run power analysis
echo "📊 Running power analysis..."
cd scripts
python3 power_analysis.py --effect-size 0.5 > "$RESULTS_DIR/power_analysis.txt"
cd ..

# Create experiment log
LOG_FILE="$RESULTS_DIR/experiment.log"
echo "Q1+ MEGA Benchmark" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Total experiments: $TOTAL" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

CURRENT=0
START_TIME=$(date +%s)

echo ""
echo "🚀 Starting Q1+ MEGA benchmark..."
echo "   This will take 4-5 DAYS"
echo "   Progress logged to: $LOG_FILE"
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
                        DAYS=$((REMAINING / 86400))
                        HOURS=$(((REMAINING % 86400) / 3600))
                        ETA_DATE=$(date -d "@$(($(date +%s) + REMAINING))" "+%Y-%m-%d %H:%M" 2>/dev/null || date -r $(($(date +%s) + REMAINING)) "+%Y-%m-%d %H:%M" 2>/dev/null || echo "N/A")
                    else
                        DAYS="?"
                        HOURS="?"
                        ETA_DATE="Calculating..."
                    fi
                    
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "MEGA Experiment $CURRENT/$TOTAL ($PERCENT%)"
                    echo "  Dataset: $DATASET | Seed: $SEED | Embedding: $EMB_MODEL"
                    echo "  θ: $THRESHOLD | Strategy: $STRATEGY"
                    echo "  Remaining: ${DAYS}d ${HOURS}h | ETA: $ETA_DATE"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    
                    echo "[$CURRENT/$TOTAL] $DATASET seed=$SEED emb=$EMB_MODEL θ=$THRESHOLD strategy=$STRATEGY" >> "$LOG_FILE"
                    
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
                    
                    # Checkpoint every 100 experiments
                    if [ $((CURRENT % 100)) -eq 0 ]; then
                        echo "📊 Checkpoint: $CURRENT/$TOTAL completed" >> "$LOG_FILE"
                        echo "   Elapsed: $((ELAPSED / 3600))h" >> "$LOG_FILE"
                    fi
                done
            done
        done
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DAYS=$((DURATION / 86400))
HOURS=$(((DURATION % 86400) / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "" >> "$LOG_FILE"
echo "Completed: $(date)" >> "$LOG_FILE"
echo "Duration: ${DAYS}d ${HOURS}h ${MINUTES}m" >> "$LOG_FILE"

echo ""
echo "✅ Q1+ MEGA benchmark completed!"
echo "   Duration: ${DAYS}d ${HOURS}h ${MINUTES}m"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""
echo "📈 Run comprehensive statistical analysis:"
echo "   cd scripts"
echo "   python3 analyze_results.py ../$RESULTS_DIR"
echo "   python3 statistical_validation.py --results-dir ../$RESULTS_DIR"
echo "   python3 bias_analysis.py --results-dir ../$RESULTS_DIR --mega-mode"
echo "   python3 cross_validation_analysis.py ../$RESULTS_DIR"
echo "   python3 visualize_results.py ../$RESULTS_DIR"
echo "   python3 generate_publication_figures.py ../$RESULTS_DIR"
echo ""
echo "🎉 Your results are ready for Nature/Science submission!"
echo ""
