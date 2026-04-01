#!/bin/bash

# ============================================
# Ollama ile Hızlı Test Scripti
# ============================================
# 16 GB RAM için optimize edilmiş
# Tamamen ücretsiz ve yerel çalışır
# Watchdog ile kesintisiz çalışma garantisi

set -e

echo "🚀 Ollama Test Başlatılıyor..."
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

# Model kontrolü
echo "2️⃣ Model kontrolü..."
# Yüklü modeli otomatik tespit et
AVAILABLE_MODEL=$(ollama list | grep -E "llama3.2|phi3|mistral|gemma2" | head -1 | awk '{print $1}')
if [ -z "$AVAILABLE_MODEL" ]; then
    echo "⚠️  Uygun model bulunamadı!"
    echo "📥 llama3.2:3b indiriliyor (bu birkaç dakika sürebilir)..."
    ollama pull llama3.2:3b
    AVAILABLE_MODEL="llama3.2:3b"
fi
MODEL="${OLLAMA_MODEL:-$AVAILABLE_MODEL}"
echo "✅ Model hazır: $MODEL"
echo ""

# Watchdog başlat (opsiyonel ama önerilen)
echo "3️⃣ Ollama watchdog başlatılıyor..."
if pgrep -f "ollama_watchdog.sh" > /dev/null; then
    echo "✅ Watchdog zaten çalışıyor"
else
    bash scripts/ollama_watchdog.sh > /tmp/ollama_watchdog.log 2>&1 &
    WATCHDOG_PID=$!
    echo "✅ Watchdog başlatıldı (PID: $WATCHDOG_PID)"
    echo "   Log: /tmp/ollama_watchdog.log"
fi
echo ""

# Test çalıştırma
echo "4️⃣ Hızlı test başlatılıyor..."
echo "📊 Dataset: msmarco (küçük örnek)"
echo "🔢 Seed: 42"
echo "⏱️  Tahmini süre: 5-10 dakika"
echo "🔄 Timeout: 10 dakika (yerel model için)"
echo ""

mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark,ollama \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42 \
  -Dbenchmark.warmup-ratio=0.1 \
  -Dbenchmark.output-file=results/ollama_test.json \
  -Dllm.ollama.model=$MODEL \
  -Dllm.ollama.timeout-seconds=600 \
  -Dcache.max-entries=5000

echo ""
echo "✅ Test tamamlandı!"
echo "📁 Sonuçlar: results/ollama_test.json"
echo ""
echo "📊 Sonuçları görüntülemek için:"
echo "   cat results/ollama_test.json | jq '.'"
echo ""
echo "🛑 Watchdog'u durdurmak için:"
echo "   pkill -f ollama_watchdog.sh"
