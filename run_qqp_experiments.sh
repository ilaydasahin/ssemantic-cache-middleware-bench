#!/bin/bash
# ============================================
# QQP Experiment Runner — Extends main benchmark with Quora dataset
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
OUTPUT_DIR="${PROJECT_DIR}/results/20260310_162645"  # Append to existing results

echo "=== QQP Experiment Extension ==="
echo "Output directory: ${OUTPUT_DIR}"

# Experimental Matrix (same as main benchmark)
THRESHOLDS=(0.75 0.80 0.85 0.90 0.95)
MODELS=("minilm" "mpnet")
SEEDS=(42 123 456 789 101)
export PYTHONHASHSEED=42
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"
SAMPLE_SIZE=1000

# Prerequisites Check
if ! redis-cli ping > /dev/null 2>&1; then
    echo "ERROR: Redis is not running. Start with: redis-server"
    exit 1
fi

# Build project
echo "--- Building project ---"
cd "${PROJECT_DIR}"
mvn clean package -DskipTests -q

TOTAL_SEMANTIC=$((${#THRESHOLDS[@]} * ${#MODELS[@]} * ${#SEEDS[@]}))
TOTAL_BASELINE=$((${#SEEDS[@]}))
TOTAL_REMOTE=$((${#THRESHOLDS[@]} * ${#MODELS[@]} * ${#SEEDS[@]}))
TOTAL=$((TOTAL_SEMANTIC + TOTAL_BASELINE + TOTAL_REMOTE))
CURRENT=0

# --- PHASE 1: EXACT_MATCH BASELINE for QQP ---
echo ""
echo "=== Phase 1: QQP Baseline (EXACT_MATCH) ==="
for SEED in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    RESULT_FILE="${OUTPUT_DIR}/baseline_quora-pairs_s${SEED}.json"
    
    java -jar target/semantic-cache-benchmark-1.0.0.jar \
        --server.port=0 \
        --benchmark.strategy="EXACT_MATCH" \
        --benchmark.current-dataset="quora-pairs" \
        --benchmark.current-seed="${SEED}" \
        --benchmark.sampleSize="${SAMPLE_SIZE}" \
        --benchmark.output-file="${RESULT_FILE}" \
        --spring.profiles.active=benchmark,benchmark-mock > /dev/null 2>&1
    
    echo "[${CURRENT}/${TOTAL}] BASELINE: quora-pairs (Seed ${SEED}) ✅"
done

# --- PHASE 2: SEMANTIC CACHE (LOCAL PARALLEL) for QQP ---
echo ""
echo "=== Phase 2: QQP Semantic Cache (Local Parallel) ==="
for THRESHOLD in "${THRESHOLDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            RESULT_FILE="${OUTPUT_DIR}/quora-pairs_${MODEL}_t${THRESHOLD}_s${SEED}.json"
            
            java -jar target/semantic-cache-benchmark-1.0.0.jar \
                --server.port=0 \
                --benchmark.strategy="SEMANTIC" \
                --benchmark.hnsw-enabled="false" \
                --benchmark.similarityThreshold="${THRESHOLD}" \
                --embedding.model-name="${MODEL}" \
                --benchmark.current-dataset="quora-pairs" \
                --benchmark.current-seed="${SEED}" \
                --benchmark.sampleSize="${SAMPLE_SIZE}" \
                --benchmark.output-file="${RESULT_FILE}" \
                --spring.profiles.active=benchmark,benchmark-mock >> "${OUTPUT_DIR}/experiment_qqp_local.log" 2>&1
            
            echo "[${CURRENT}/${TOTAL}] LOCAL: quora-pairs/${MODEL} @ θ=${THRESHOLD} (Seed ${SEED}) ✅"
        done
    done
done

# --- PHASE 3: REMOTE INDEX BASELINE (HNSW) for QQP ---
echo ""
echo "=== Phase 3: QQP Remote Index (Redis HNSW) ==="
for THRESHOLD in "${THRESHOLDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            RESULT_FILE="${OUTPUT_DIR}/remote_hnsw_quora-pairs_${MODEL}_t${THRESHOLD}_s${SEED}.json"
            
            java -jar target/semantic-cache-benchmark-1.0.0.jar \
                --server.port=0 \
                --benchmark.strategy="SEMANTIC" \
                --benchmark.hnsw-enabled="true" \
                --benchmark.similarityThreshold="${THRESHOLD}" \
                --embedding.model-name="${MODEL}" \
                --benchmark.current-dataset="quora-pairs" \
                --benchmark.current-seed="${SEED}" \
                --benchmark.sampleSize="${SAMPLE_SIZE}" \
                --benchmark.output-file="${RESULT_FILE}" \
                --spring.profiles.active=benchmark,benchmark-mock >> "${OUTPUT_DIR}/experiment_qqp_remote.log" 2>&1
            
            echo "[${CURRENT}/${TOTAL}] REMOTE HNSW: quora-pairs/${MODEL} @ θ=${THRESHOLD} (Seed ${SEED}) ✅"
        done
    done
done

echo ""
echo "=== QQP Experiments Completed ==="
echo "Total runs: ${CURRENT}"
echo "Results appended to: ${OUTPUT_DIR}"
echo ""
echo "--- Running Statistical Analysis (including QQP) ---"
python3 "${SCRIPT_DIR}/scripts/analyze_results.py" "${OUTPUT_DIR}"
echo "=== Done ==="
