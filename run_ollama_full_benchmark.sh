#!/bin/bash

# ============================================
# Ollama ile Tam Benchmark Scripti
# ============================================
# 16 GB RAM için optimize edilmiş
# Tamamen ücretsiz ve yerel çalışır

set -e

echo "🚀 Ollama Tam Benchmark Başlatılıyor..."
echo ""

# Ollama kontrolü
echo "1️⃣ Ollama servis kontrolü..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama çalışmıyor!"
    echo "💡 Başlatmak için: ollama serve"
    exit 1
fi
echo "✅ Ollama çalışıyor"
echo ""

# Model seçimi
MODEL="${OLLAMA_MODEL:-llama3.2}"
echo "2️⃣ Model: $MODEL"
if ! ollama list | grep -q "$MODEL"; then
    echo "⚠️  Model bulunamadı: $MODEL"
    echo "📥 Model indiriliyor..."
    ollama pull $MODEL
fi
echo "✅ Model hazır"
echo ""

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/ollama_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "3️⃣ Benchmark parametreleri:"
echo "   📁 Sonuç dizini: $RESULTS_DIR"
echo "   🔢 Seeds: 42, 123, 456"
echo "   📊 Datasets: msmarco, natural-questions, quora-pairs"
echo "   🎯 Strategies: SEMANTIC, EXACT_MATCH, HYBRID"
echo "   📈 Thresholds: 0.85, 0.90, 0.95"
echo ""

# Deney sayacı
TOTAL_EXPERIMENTS=27  # 3 datasets × 3 seeds × 3 strategies
CURRENT=0

# Datasets
DATASETS=("msmarco" "natural-questions" "quora-pairs")
SEEDS=(42 123 456)
STRATEGIES=("SEMANTIC" "EXACT_MATCH" "HYBRID")

echo "4️⃣ Benchmark başlatılıyor..."
echo "⏱️  Tahmini süre: 2-4 saat (modele göre değişir)"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for STRATEGY in "${STRATEGIES[@]}"; do
            CURRENT=$((CURRENT + 1))
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📊 Deney $CURRENT/$TOTAL_EXPERIMENTS"
            echo "   Dataset: $DATASET"
            echo "   Seed: $SEED"
            echo "   Strategy: $STRATEGY"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            OUTPUT_FILE="$RESULTS_DIR/${DATASET}_${SEED}_${STRATEGY}.json"
            
            mvn spring-boot:run \
              -Dspring-boot.run.profiles=benchmark,ollama \
              -Dbenchmark.current-dataset=$DATASET \
              -Dbenchmark.current-seed=$SEED \
              -Dcache.strategy=$STRATEGY \
              -Dbenchmark.output-file=$OUTPUT_FILE \
              -Dllm.ollama.model=$MODEL \
              -Dcache.max-entries=10000 \
              -Dbenchmark.warmup-ratio=0.30 \
              > "$RESULTS_DIR/${DATASET}_${SEED}_${STRATEGY}.log" 2>&1
            
            echo "✅ Tamamlandı: $OUTPUT_FILE"
            echo ""
        done
    done
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 TÜM BENCHMARK TAMAMLANDI!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Sonuçlar: $RESULTS_DIR"
echo "📊 Toplam deney: $TOTAL_EXPERIMENTS"
echo ""
echo "📈 Analiz için:"
echo "   cd scripts"
echo "   python analyze_results.py ../$RESULTS_DIR"
echo "   python visualize_results.py ../$RESULTS_DIR"
echo ""
echo "💰 Maliyet: 0 TL (Tamamen ücretsiz!)"
echo "✅ Başarılar!"
