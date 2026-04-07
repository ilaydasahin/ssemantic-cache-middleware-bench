#!/bin/bash
# Ablation Study - Component-wise Impact Analysis
# Tests each component's contribution to overall performance

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Ablation Study - Q1 Publication              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}
SEEDS=(42 123 456)  # 3 seeds for ablation
DATASET="msmarco"   # Single dataset for ablation

echo "📋 Ablation Study Configuration:"
echo "   LLM Model: $OLLAMA_MODEL"
echo "   Seeds: ${SEEDS[@]}"
echo "   Dataset: $DATASET"
echo ""

# Create results directory
RESULTS_DIR="results/ablation_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "🔬 Testing Component Contributions..."
echo ""

# ============================================================================
# ABLATION 1: HNSW vs Brute-Force
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ABLATION 1: HNSW vs Brute-Force"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for SEED in "${SEEDS[@]}"; do
    # With HNSW
    echo "  [1a] HNSW enabled (seed=$SEED)"
    mvn spring-boot:run \
        -q \
        -Dspring-boot.run.profiles=benchmark \
        -Dbenchmark.current-dataset="$DATASET" \
        -Dbenchmark.current-seed="$SEED" \
        -Dcache.strategy=SEMANTIC \
        -Dcache.hnsw-enabled=true \
        -Dresults.output-dir="$RESULTS_DIR/hnsw_enabled"
    
    # Without HNSW (brute-force only)
    echo "  [1b] HNSW disabled (seed=$SEED)"
    mvn spring-boot:run \
        -q \
        -Dspring-boot.run.profiles=benchmark \
        -Dbenchmark.current-dataset="$DATASET" \
        -Dbenchmark.current-seed="$SEED" \
        -Dcache.strategy=SEMANTIC \
        -Dcache.hnsw-enabled=false \
        -Dresults.output-dir="$RESULTS_DIR/hnsw_disabled"
done

echo ""

# ============================================================================
# ABLATION 2: Embedding Model Comparison
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ABLATION 2: Embedding Model Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EMBEDDING_MODELS=(minilm mpnet tinybert)

for SEED in "${SEEDS[@]}"; do
    for EMB_MODEL in "${EMBEDDING_MODELS[@]}"; do
        echo "  [2] Embedding: $EMB_MODEL (seed=$SEED)"
        mvn spring-boot:run \
            -q \
            -Dspring-boot.run.profiles=benchmark \
            -Dbenchmark.current-dataset="$DATASET" \
            -Dbenchmark.current-seed="$SEED" \
            -Dcache.strategy=SEMANTIC \
            -Dembedding.model-name="$EMB_MODEL" \
            -Dresults.output-dir="$RESULTS_DIR/embedding_$EMB_MODEL"
    done
done

echo ""

# ============================================================================
# ABLATION 3: Threshold Sensitivity
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ABLATION 3: Threshold Sensitivity (0.70 - 0.99)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

THRESHOLDS=(0.70 0.75 0.80 0.85 0.90 0.95 0.99)

for SEED in "${SEEDS[@]}"; do
    for THRESHOLD in "${THRESHOLDS[@]}"; do
        echo "  [3] Threshold: $THRESHOLD (seed=$SEED)"
        mvn spring-boot:run \
            -q \
            -Dspring-boot.run.profiles=benchmark \
            -Dbenchmark.current-dataset="$DATASET" \
            -Dbenchmark.current-seed="$SEED" \
            -Dcache.strategy=SEMANTIC \
            -Dcache.similarity-threshold="$THRESHOLD" \
            -Dresults.output-dir="$RESULTS_DIR/threshold_$THRESHOLD"
    done
done

echo ""

# ============================================================================
# ABLATION 4: Strategy Comparison
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ABLATION 4: Strategy Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STRATEGIES=(SEMANTIC HYBRID EXACT_MATCH GPTCACHE_BASELINE NONE)

for SEED in "${SEEDS[@]}"; do
    for STRATEGY in "${STRATEGIES[@]}"; do
        echo "  [4] Strategy: $STRATEGY (seed=$SEED)"
        mvn spring-boot:run \
            -q \
            -Dspring-boot.run.profiles=benchmark \
            -Dbenchmark.current-dataset="$DATASET" \
            -Dbenchmark.current-seed="$SEED" \
            -Dcache.strategy="$STRATEGY" \
            -Dresults.output-dir="$RESULTS_DIR/strategy_$STRATEGY"
    done
done

echo ""
echo "✅ Ablation study completed!"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""
echo "📈 Analyze results:"
echo "   cd scripts"
echo "   python3 analyze_ablation_study.py ../$RESULTS_DIR"
echo ""
