#!/bin/bash
# Quick test with 3 seeds to verify everything works before full run
# Estimated time: 30-45 minutes

set -e

SEEDS=(42 123 456)
DATASETS=("msmarco")
STRATEGIES=("SEMANTIC" "EXACT_MATCH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/q1_quick_test_${TIMESTAMP}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Q1 QUICK TEST - Verify Setup Before Full Run             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  • Seeds: ${#SEEDS[@]} (quick test)"
echo "  • Datasets: ${#DATASETS[@]} (MS MARCO only)"
echo "  • Strategies: ${#STRATEGIES[@]} (SEMANTIC, EXACT_MATCH)"
echo "  • Total experiments: $((${#SEEDS[@]} * ${#DATASETS[@]} * ${#STRATEGIES[@]}))"
echo "  • Estimated time: 30-45 minutes"
echo ""

# Validation
echo "[1/4] Pre-flight validation..."
python3 scripts/validate_experiment.py
if [ $? -ne 0 ]; then
    echo "❌ Validation failed"
    exit 1
fi
echo "✅ Validation passed"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Run experiments
echo "[2/4] Running quick test benchmark..."
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]} * ${#STRATEGIES[@]}))
CURRENT=0

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for STRATEGY in "${STRATEGIES[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo "[${CURRENT}/${TOTAL}] Seed=${SEED}, Dataset=${DATASET}, Strategy=${STRATEGY}"
            
            LOG_FILE="${RESULTS_DIR}/${DATASET}_${SEED}_${STRATEGY}.log"
            
            mvn spring-boot:run \
                -Dspring-boot.run.profiles=benchmark,ollama \
                -Dspring-boot.run.arguments="--mode=throughput --dataset=${DATASET} --seed=${SEED} --strategy=${STRATEGY} --concurrent-users=50" \
                > "${LOG_FILE}" 2>&1
            
            if [ $? -eq 0 ]; then
                echo "  ✅ Success"
            else
                echo "  ❌ Failed"
            fi
            
            sleep 2
        done
    done
done

echo ""
echo "[3/4] Analyzing results..."
python3 scripts/analyze_results.py "${RESULTS_DIR}" > "${RESULTS_DIR}/analysis.txt" 2>&1 || echo "⚠️  Analysis had issues"

echo ""
echo "[4/4] Running bias check..."
python3 scripts/bias_analysis.py --results-dir "${RESULTS_DIR}" > "${RESULTS_DIR}/bias.txt" 2>&1 || echo "⚠️  Bias analysis had issues"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  QUICK TEST COMPLETE                                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Results: ${RESULTS_DIR}"
echo ""
echo "If everything looks good, run the full benchmark:"
echo "  ./run_q1_comprehensive_benchmark.sh"
echo ""
