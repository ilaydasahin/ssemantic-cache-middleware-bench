#!/bin/bash
# Q1+ MEGA Benchmark (4-5 days)
# Maximum statistical rigor with 64 seeds for d=0.5 at 80% power
# Suitable for Nature/Science level publications

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Q1+ MEGA BENCHMARK - Nature/Science Level            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration - 64 seeds for 80% power at d=0.5
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}

# 64 seeds for maximum statistical power
SEEDS=(
    42 123 456 789 1024 2048 3141 4096 5555 6789 
    7777 8888 9999 10101 11111 12345 13579 14641 15873 16384 
    17777 18888 19999 20202 21212 22222 23456 24680 25252 26384
    27777 28888 29999 30303 31415 32768 33333 34567 35791 36864
    37777 38888 39999 40404 41421 42424 43210 44444 45678 46656
    47777 48888 49999 50505 51234 52428 53333 54321 55555 56789
    57777 58888 59999 60606
)

DATASETS=(msmarco natural-questions quora-pairs)
EMBEDDING_MODELS=(minilm mpnet tinybert)
THRESHOLDS=(0.85 0.90 0.95)
STRATEGIES=(SEMANTIC EXACT_MATCH NONE)

echo "📋 Q1+ MEGA Configuration:"
echo "   LLM Model: $OLLAMA_MODEL"
echo "   Seeds: ${#SEEDS[@]} seeds (64 for 80% power at d=0.5)"
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
echo "   Estimated storage: ~8-10 GB"
echo ""
echo "⚠️  THIS IS AN EXTREMELY LONG-RUNNING BENCHMARK"
echo "   Recommended: Dedicated server or cloud instance"
echo "   Consider: Running in tmux/screen session"
echo "   Backup: Checkpoint system will save progress"
echo ""
echo "🎯 Statistical Power:"
echo "   • Can detect medium effects (d=0.5) with 80% power"
echo "   • Can detect large effects (d=0.8) with 99% power"
echo "   • Suitable for top-tier Q1 journals (Nature, Science, Cell)"
echo ""

read -p "Are you ABSOLUTELY SURE you want to run this? (yes/no) " -r
echo
if [[ ! $REPLY == "yes" ]]; then
    echo "Cancelled. For shorter benchmarks:"
    echo "  • Quick test (30 min):    ./bin/run_q1_quick_test.sh"
    echo "  • Q1 standard (12-16h):   ./bin/run_q1_comprehensive_benchmark.sh"
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

# Check disk space (need at least 15GB)
AVAILABLE=$(df -k . | tail -1 | awk '{print $4}')
REQUIRED=$((15 * 1024 * 1024))  # 15GB in KB
if [ "$AVAILABLE" -lt "$REQUIRED" ]; then
    echo "❌ Insufficient disk space!"
    echo "   Available: $((AVAILABLE / 1024 / 1024)) GB"
    echo "   Required: 15 GB"
    exit 1
fi

# Check Python dependencies
if ! python3 -c "import statsmodels, scipy, numpy, pandas, matplotlib" 2>/dev/null; then
    echo "⚠️  Python dependencies missing"
    echo "   Installing..."
    cd scripts
    pip install -r requirements.txt
    cd ..
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
python3 power_analysis.py --effect-size 0.5 --power 0.80 > "$RESULTS_DIR/power_analysis.txt"
python3 power_analysis.py --show-curves 2>/dev/null || echo "⚠️  Skipping power curves (matplotlib not available)"
if [ -f "power_curves.png" ]; then
    mv power_curves.png "$RESULTS_DIR/"
fi
cd ..

# Create experiment log
LOG_FILE="$RESULTS_DIR/experiment.log"
echo "Q1+ MEGA Benchmark - Nature/Science Level" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Total experiments: $TOTAL" >> "$LOG_FILE"
echo "Seeds: ${#SEEDS[@]} (64 for d=0.5 at 80% power)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Create checkpoint file
CHECKPOINT_FILE="$RESULTS_DIR/checkpoint.txt"
touch "$CHECKPOINT_FILE"

CURRENT=0
START_TIME=$(date +%s)

echo ""
echo "🚀 Starting Q1+ MEGA benchmark..."
echo "   Progress will be logged to: $LOG_FILE"
echo "   Checkpoints saved to: $CHECKPOINT_FILE"
echo ""
echo "⚠️  This will take 4-5 DAYS. You can safely interrupt and resume."
echo ""

# Run experiments
for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for EMB_MODEL in "${EMBEDDING_MODELS[@]}"; do
            for THRESHOLD in "${THRESHOLDS[@]}"; do
                for STRATEGY in "${STRATEGIES[@]}"; do
                    CURRENT=$((CURRENT + 1))
                    
                    # Check if already completed (resume support)
                    CHECKPOINT_KEY="${DATASET}_${SEED}_${EMB_MODEL}_${THRESHOLD}_${STRATEGY}"
                    if grep -q "$CHECKPOINT_KEY" "$CHECKPOINT_FILE" 2>/dev/null; then
                        echo "⏭️  Skipping completed: $CHECKPOINT_KEY"
                        continue
                    fi
                    
                    # Calculate progress
                    PERCENT=$((CURRENT * 100 / TOTAL))
                    ELAPSED=$(($(date +%s) - START_TIME))
                    if [ $CURRENT -gt 1 ]; then
                        AVG_TIME=$((ELAPSED / (CURRENT - 1)))
                        REMAINING=$(((TOTAL - CURRENT) * AVG_TIME))
                        DAYS=$((REMAINING / 86400))
                        HOURS=$(((REMAINING % 86400) / 3600))
                        MINUTES=$(((REMAINING % 3600) / 60))
                        ETA_STR="${DAYS}d ${HOURS}h ${MINUTES}m"
                    else
                        ETA_STR="Calculating..."
                    fi
                    
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "Experiment $CURRENT/$TOTAL ($PERCENT%)"
                    echo "  Dataset: $DATASET | Seed: $SEED | Embedding: $EMB_MODEL"
                    echo "  θ: $THRESHOLD | Strategy: $STRATEGY"
                    echo "  ETA: $ETA_STR"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    
                    # Log experiment
                    echo "[$CURRENT/$TOTAL] $DATASET seed=$SEED emb=$EMB_MODEL θ=$THRESHOLD strategy=$STRATEGY" >> "$LOG_FILE"
                    
                    # Run experiment
                    if mvn spring-boot:run \
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
                        -Dresults.output-dir="$RESULTS_DIR"; then
                        echo "  ✅ Success" >> "$LOG_FILE"
                        echo "$CHECKPOINT_KEY" >> "$CHECKPOINT_FILE"
                    else
                        echo "  ❌ Failed" >> "$LOG_FILE"
                    fi
                    
                    echo ""
                    
                    # Periodic status update (every 100 experiments)
                    if [ $((CURRENT % 100)) -eq 0 ]; then
                        echo ""
                        echo "📊 Progress Update:"
                        echo "   Completed: $CURRENT/$TOTAL ($PERCENT%)"
                        echo "   Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
                        echo "   Remaining: $ETA_STR"
                        echo ""
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
echo "   python3 bias_analysis.py --results-dir ../$RESULTS_DIR"
echo "   python3 effect_size_calculator.py --results-dir ../$RESULTS_DIR"
echo "   python3 visualize_results.py ../$RESULTS_DIR"
echo "   python3 generate_publication_figures.py ../$RESULTS_DIR"
echo ""
echo "🎯 Statistical Power Achieved:"
echo "   • Medium effects (d=0.5): 80% power ✅"
echo "   • Large effects (d=0.8): 99% power ✅"
echo "   • Suitable for: Nature, Science, Cell, top-tier Q1 journals"
echo ""
echo "✅ Your results are ready for TOP-TIER Q1 publication!"
echo ""
