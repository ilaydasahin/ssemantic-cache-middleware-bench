#!/bin/bash

# ============================================
# Ollama ile Tam Benchmark Scripti
# ============================================
# 16 GB RAM için optimize edilmiş
# Tamamen ücretsiz ve yerel çalışır
# Watchdog + Checkpoint ile kesintisiz çalışma

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
echo "2️⃣ Model kontrolü..."
# Yüklü modeli otomatik tespit et
AVAILABLE_MODEL=$(ollama list | grep -E "llama3.2|phi3|mistral|gemma2" | head -1 | awk '{print $1}')
if [ -z "$AVAILABLE_MODEL" ]; then
    echo "⚠️  Uygun model bulunamadı!"
    echo "📥 llama3.2:3b indiriliyor..."
    ollama pull llama3.2:3b
    AVAILABLE_MODEL="llama3.2:3b"
fi
MODEL="${OLLAMA_MODEL:-$AVAILABLE_MODEL}"
echo "✅ Model hazır: $MODEL"
echo ""

# Watchdog başlat
echo "3️⃣ Ollama watchdog başlatılıyor..."
if pgrep -f "ollama_watchdog.sh" > /dev/null; then
    echo "✅ Watchdog zaten çalışıyor"
else
    bash scripts/ollama_watchdog.sh > /tmp/ollama_watchdog.log 2>&1 &
    WATCHDOG_PID=$!
    echo "✅ Watchdog başlatıldı (PID: $WATCHDOG_PID)"
    echo "   Log: /tmp/ollama_watchdog.log"
    echo "   🛡️  Ollama çökerse otomatik yeniden başlatılacak"
fi
echo ""

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/ollama_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "4️⃣ Benchmark parametreleri:"
echo "   📁 Sonuç dizini: $RESULTS_DIR"
echo "   🔢 Seeds: 42, 123, 456"
echo "   📊 Datasets: msmarco, natural-questions, quora-pairs"
echo "   🎯 Strategies: SEMANTIC, EXACT_MATCH, HYBRID"
echo "   📈 Thresholds: 0.85, 0.90, 0.95"
echo "   🔄 Timeout: 10 dakika/query (yerel model için)"
echo "   💾 Checkpoint: Her 50 query'de kayıt"
echo ""

# Deney sayacı
TOTAL_EXPERIMENTS=27  # 3 datasets × 3 seeds × 3 strategies
CURRENT=0
FAILED=0

# Datasets
DATASETS=("msmarco" "natural-questions" "quora-pairs")
SEEDS=(42 123 456)
STRATEGIES=("SEMANTIC" "EXACT_MATCH" "HYBRID")

echo "5️⃣ Benchmark başlatılıyor..."
echo "⏱️  Tahmini süre: 2-4 saat (modele göre değişir)"
echo "💡 Kesintiye uğrarsa checkpoint'ten devam eder"
echo ""

# Trap for cleanup
trap 'echo ""; echo "⚠️  Benchmark durduruldu. Checkpoint kaydedildi."; echo "🔄 Devam etmek için scripti tekrar çalıştırın."; exit 130' INT TERM

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
            LOG_FILE="$RESULTS_DIR/${DATASET}_${SEED}_${STRATEGY}.log"
            
            # Retry mekanizması (Ollama çökerse)
            MAX_RETRIES=3
            RETRY=0
            SUCCESS=false
            
            while [ $RETRY -lt $MAX_RETRIES ] && [ "$SUCCESS" = false ]; do
                if [ $RETRY -gt 0 ]; then
                    echo "⚠️  Deneme $((RETRY + 1))/$MAX_RETRIES (önceki deneme başarısız)"
                    sleep 10
                fi
                
                if mvn spring-boot:run \
                  -Dspring-boot.run.profiles=benchmark,ollama \
                  -Dbenchmark.current-dataset=$DATASET \
                  -Dbenchmark.current-seed=$SEED \
                  -Dcache.strategy=$STRATEGY \
                  -Dbenchmark.output-file=$OUTPUT_FILE \
                  -Dllm.ollama.model=$MODEL \
                  -Dllm.ollama.timeout-seconds=600 \
                  -Dcache.max-entries=10000 \
                  -Dbenchmark.warmup-ratio=0.30 \
                  > "$LOG_FILE" 2>&1; then
                    SUCCESS=true
                    echo "✅ Tamamlandı: $OUTPUT_FILE"
                else
                    RETRY=$((RETRY + 1))
                    echo "❌ Hata oluştu (log: $LOG_FILE)"
                fi
            done
            
            if [ "$SUCCESS" = false ]; then
                echo "❌ Deney başarısız oldu (max retry aşıldı)"
                FAILED=$((FAILED + 1))
            fi
            
            echo ""
        done
    done
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 BENCHMARK TAMAMLANDI!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Sonuçlar: $RESULTS_DIR"
echo "📊 Toplam deney: $TOTAL_EXPERIMENTS"
echo "✅ Başarılı: $((TOTAL_EXPERIMENTS - FAILED))"
if [ $FAILED -gt 0 ]; then
    echo "❌ Başarısız: $FAILED"
fi
echo ""
echo "📈 Analiz için:"
echo "   cd scripts"
echo "   python analyze_results.py ../$RESULTS_DIR"
echo "   python visualize_results.py ../$RESULTS_DIR"
echo ""
echo "💰 Maliyet: 0 TL (Tamamen ücretsiz!)"
echo ""
echo "🛑 Watchdog'u durdurmak için:"
echo "   pkill -f ollama_watchdog.sh"
echo ""
echo "✅ Başarılar!"
