#!/bin/bash

echo "🚀 Semantic Cache Benchmark - Tam Deney Başlatılıyor"
echo "=================================================="
echo ""

# Parametreler
DATASET="msmarco"
SEED=42
SAMPLE_SIZE=450000
PROFILE="benchmark,benchmark-mock"  # Mock API ile başla
OUTPUT_FILE="results/full_experiment_${DATASET}_seed${SEED}_${SAMPLE_SIZE}.json"

echo "📋 Deney Parametreleri:"
echo "   Dataset: $DATASET"
echo "   Random Seed: $SEED"
echo "   Sample Size: $SAMPLE_SIZE"
echo "   Output: $OUTPUT_FILE"
echo "   Profile: $PROFILE"
echo ""

# API anahtarlarını kontrol et
if [ -z "$GEMINI_API_KEYS" ]; then
    echo "⚠️  GEMINI_API_KEYS ayarlanmamış - Mock API kullanılacak"
    echo "   Gerçek API için: export GEMINI_API_KEYS='key1,key2,...'"
    PROFILE="benchmark,benchmark-mock"
else
    echo "✅ GEMINI_API_KEYS bulundu - Gerçek API kullanılacak"
    KEY_COUNT=$(echo $GEMINI_API_KEYS | tr ',' '\n' | wc -l | xargs)
    echo "   Toplam anahtar sayısı: $KEY_COUNT"
    PROFILE="benchmark"
fi

echo ""
echo "🔧 Sistem Hazırlığı:"

# Heap boyutunu ayarla
export MAVEN_OPTS="-Xmx8g -Xms4g"
echo "   ✅ JVM Heap: 8GB max, 4GB initial"

# Directories
mkdir -p results checkpoints logs
echo "   ✅ Klasörler oluşturuldu"

echo ""
echo "⏰ Tahmini Süre:"
if [[ "$PROFILE" == *"mock"* ]]; then
    echo "   Mock API: ~2-4 saat (450K sorgu)"
else
    echo "   Gerçek API: ~24-48 saat (77 anahtar ile)"
fi

echo ""
echo "💾 Checkpoint Sistemi:"
echo "   ✅ Otomatik kayıt aktif"
echo "   ✅ Kesintide kaldığı yerden devam eder"
echo "   📁 Checkpoint: checkpoints/${DATASET}_seed${SEED}_t0.90.json"

echo ""
echo "🎯 Deneyi Durdurmak İçin: Ctrl+C"
echo "🔄 Devam Ettirmek İçin: Bu scripti tekrar çalıştırın"
echo ""
echo "=================================================="
echo "🚀 Başlatılıyor..."
echo ""

# Caffeinate ile çalıştır (Mac uyumaz)
caffeinate -i mvn spring-boot:run \
  -Dspring-boot.run.profiles=$PROFILE \
  -Dbenchmark.current-dataset=$DATASET \
  -Dbenchmark.current-seed=$SEED \
  -Dbenchmark.sample-size=$SAMPLE_SIZE \
  -Dbenchmark.output-file=$OUTPUT_FILE

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Deney başarıyla tamamlandı!"
    echo "📊 Sonuçlar: $OUTPUT_FILE"
    echo ""
    echo "📈 Sonuçları analiz etmek için:"
    echo "   cd scripts"
    echo "   python analyze_results.py ../results/"
    echo "   python visualize_results.py ../results/"
else
    echo "⚠️  Deney durduruldu (Exit code: $EXIT_CODE)"
    echo "🔄 Devam ettirmek için scripti tekrar çalıştırın"
    echo "   Checkpoint'ten otomatik devam edecek"
fi
echo "=================================================="
