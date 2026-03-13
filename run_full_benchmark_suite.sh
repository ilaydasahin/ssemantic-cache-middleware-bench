#!/bin/bash
# ============================================
# Semantic Cache Benchmark — Full Experiment Runner
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROJECT_DIR}/results/${TIMESTAMP}"

echo "=== Semantic Cache Benchmark - High Fidelity Matrix ==="
echo "Output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# 1. Experimental Matrix (M.6 Compliance)
THRESHOLDS=(0.75 0.80 0.85 0.90 0.95)
MODELS=("minilm" "mpnet")
DATASETS=("msmarco" "natural-questions")
SEEDS=(42 123 456)
STRATEGIES=("SEMANTIC" "EXACT_MATCH")
SAMPLE_SIZE=1000 # Sample size for each run (M.7 Statistical Significance requirement)

# 2. Prerequisites Check
if ! redis-cli ping > /dev/null 2>&1; then
    echo "ERROR: Redis is not running. Start with: redis-server"
    exit 1
fi

if [ -z "${GEMINI_API_KEY}" ]; then
    echo "WARNING: GEMINI_API_KEY not set. Using actual LLM will fail."
fi

# 3. Build project
echo "--- Building project ---"
cd "${PROJECT_DIR}"
mvn clean package -DskipTests -q

# 4. Run Experiments
TOTAL_SEMANTIC=$((${#THRESHOLDS[@]} * ${#MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
TOTAL_BASELINE=$((${#DATASETS[@]} * ${#SEEDS[@]}))
TOTAL_REMOTE=$((${#THRESHOLDS[@]} * ${#MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]})) # Expanded for fairness
TOTAL_ABLATION=2
TOTAL_ZIPFIAN=4
TOTAL_ROBUSTNESS=2
TOTAL=$((TOTAL_SEMANTIC + TOTAL_BASELINE + TOTAL_REMOTE + TOTAL_ABLATION + TOTAL_ZIPFIAN + TOTAL_ROBUSTNESS))
CURRENT=0

# --- PHASE 1: EXACT_MATCH BASELINE ---
echo ""
echo "=== Phase 1: Baseline (EXACT_MATCH) ==="
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        RESULT_FILE="${OUTPUT_DIR}/baseline_${DATASET}_s${SEED}.json"
        
        java -jar target/semantic-cache-benchmark-1.0.0.jar \
            --server.port=0 \
            --benchmark.strategy="EXACT_MATCH" \
            --benchmark.current-dataset="${DATASET}" \
            --benchmark.current-seed="${SEED}" \
            --benchmark.sampleSize="${SAMPLE_SIZE}" \
            --benchmark.output-file="${RESULT_FILE}" \
            --spring.profiles.active=benchmark,benchmark-mock > /dev/null 2>&1
        
        echo "[${CURRENT}/${TOTAL}] BASELINE: ${DATASET} (Seed ${SEED}) ✅"
    done
done

# --- PHASE 2: SEMANTIC CACHE (PRIMARY: LOCAL PARALLEL) ---
echo ""
echo "=== Phase 2: Semantic Cache (Local Parallel Search) ==="
for THRESHOLD in "${THRESHOLDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                CURRENT=$((CURRENT + 1))
                RESULT_FILE="${OUTPUT_DIR}/${DATASET}_${MODEL}_t${THRESHOLD}_s${SEED}.json"
                
                # Ensure HNSW is DISABLED to test the paper's local parallel claim
                java -jar target/semantic-cache-benchmark-1.0.0.jar \
                    --server.port=0 \
                    --benchmark.strategy="SEMANTIC" \
                    --benchmark.hnsw-enabled="false" \
                    --benchmark.similarityThreshold="${THRESHOLD}" \
                    --embedding.model-name="${MODEL}" \
                    --benchmark.current-dataset="${DATASET}" \
                    --benchmark.current-seed="${SEED}" \
                    --benchmark.sampleSize="${SAMPLE_SIZE}" \
                    --benchmark.output-file="${RESULT_FILE}" \
                    --spring.profiles.active=benchmark,benchmark-mock >> "${OUTPUT_DIR}/experiment_local.log" 2>&1
                
                echo "[${CURRENT}/${TOTAL}] LOCAL: ${DATASET}/${MODEL} @ θ=${THRESHOLD} (Seed ${SEED}) ✅"
            done
        done
    done
done

# --- PHASE 3: REMOTE INDEX BASELINE (HNSW) ---
echo ""
echo "=== Phase 3: Remote Index Baseline (Redis HNSW) ==="
for THRESHOLD in "${THRESHOLDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                CURRENT=$((CURRENT + 1))
                RESULT_FILE="${OUTPUT_DIR}/remote_hnsw_${DATASET}_${MODEL}_t${THRESHOLD}_s${SEED}.json"
                
                # Test with HNSW ENABLED
                java -jar target/semantic-cache-benchmark-1.0.0.jar \
                    --server.port=0 \
                    --benchmark.strategy="SEMANTIC" \
                    --benchmark.hnsw-enabled="true" \
                    --benchmark.similarityThreshold="${THRESHOLD}" \
                    --embedding.model-name="${MODEL}" \
                    --benchmark.current-dataset="${DATASET}" \
                    --benchmark.current-seed="${SEED}" \
                    --benchmark.sampleSize="${SAMPLE_SIZE}" \
                    --benchmark.output-file="${RESULT_FILE}" \
                    --spring.profiles.active=benchmark,benchmark-mock >> "${OUTPUT_DIR}/experiment_remote.log" 2>&1
                
                echo "[${CURRENT}/${TOTAL}] REMOTE HNSW: ${DATASET}/${MODEL} @ θ=${THRESHOLD} (Seed ${SEED}) ✅"
            done
        done
    done
done

# --- PHASE 4: ABLATION STUDY (Parallel vs. Serial) ---
echo ""
echo "=== Phase 4: Ablation Study (Local Parallelism) ==="
for MODE in "true" "false"; do
    CURRENT=$((CURRENT + 1))
    RESULT_FILE="${OUTPUT_DIR}/ablation_parallel_${MODE}.json"
    echo "Testing ParallelStream=${MODE}..."
    java -jar target/semantic-cache-benchmark-1.0.0.jar \
        --server.port=0 \
        --benchmark.strategy="SEMANTIC" \
        --benchmark.hnsw-enabled="false" \
        --benchmark.parallel-enabled="${MODE}" \
        --benchmark.sampleSize=500 \
        --benchmark.output-file="${RESULT_FILE}" \
        --spring.profiles.active=benchmark,benchmark-mock >> "${OUTPUT_DIR}/ablation.log" 2>&1
    echo "[${CURRENT}/${TOTAL}] Ablation: Parallel=${MODE} ✅"
done

# --- PHASE 5: ZIPFIAN SKEW IMPACT (§5.4) ---
echo ""
echo "=== Phase 5: Zipfian Skew Impact (Realistic Traffic) ==="
SKEWS=(0.0 0.8) # Uniform vs. Realistic skew
for SKEW in "${SKEWS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        CURRENT=$((CURRENT + 1))
        RESULT_FILE="${OUTPUT_DIR}/zipfian_skew_${SKEW}_${DATASET}.json"
        
        java -jar target/semantic-cache-benchmark-1.0.0.jar \
            --server.port=0 \
            --benchmark.strategy="SEMANTIC" \
            --benchmark.zipfian-skew="${SKEW}" \
            --benchmark.current-dataset="${DATASET}" \
            --benchmark.current-seed=42 \
            --benchmark.sampleSize="${SAMPLE_SIZE}" \
            --benchmark.output-file="${RESULT_FILE}" \
            --spring.profiles.active=benchmark,benchmark-mock >> "${OUTPUT_DIR}/zipfian.log" 2>&1
            
        echo "[${CURRENT}/${TOTAL}] ZIPFIAN: Skew=${SKEW} Dataset=${DATASET} ✅"
    done
done

# --- PHASE 6: ADVERSARIAL ROBUSTNESS (§5.6) ---
echo ""
echo "=== Phase 6: Adversarial Robustness (Noise Injection) ==="
PROBS=(0.0 0.2) # Clean vs. 20% Noise
for PROB in "${PROBS[@]}"; do
    CURRENT=$((CURRENT + 1))
    RESULT_FILE="${OUTPUT_DIR}/robustness_noise_${PROB}.json"
    
    java -jar target/semantic-cache-benchmark-1.0.0.jar \
        --server.port=0 \
        --benchmark.strategy="SEMANTIC" \
        --benchmark.noise-probability="${PROB}" \
        --benchmark.current-dataset="msmarco" \
        --benchmark.current-seed=42 \
        --benchmark.sampleSize=1000 \
        --benchmark.output-file="${RESULT_FILE}" \
        --spring.profiles.active=benchmark,benchmark-mock >> "${OUTPUT_DIR}/robustness.log" 2>&1
        
    echo "[${CURRENT}/${TOTAL}] ROBUSTNESS: NoiseProb=${PROB} ✅"
done

echo ""
echo "=== Experiments Completed ==="
echo "--- Running Statistical Analysis (SBERT + Wilcoxon) ---"
python3 "${SCRIPT_DIR}/scripts/analyze_results.py" "${OUTPUT_DIR}"

echo ""
echo "Verification tables generated in ${OUTPUT_DIR}/table4_summary.csv"
echo "=== Done ==="
