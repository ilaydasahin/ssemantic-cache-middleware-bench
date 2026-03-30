#!/bin/bash

# ============================================
# Ollama ile Hızlı Test Scripti
# ============================================
# 16 GB RAM için optimize edilmiş
# Tamamen ücretsiz ve yerel çalışır

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
MODEL="llama3.2"
if ! ollama list | grep -q "$MODEL"; then
    echo "⚠️  Model bulunamadı: $MODEL"
    echo "📥 Model indiriliyor (bu birkaç dakika sürebilir)..."
    ollama pull $MODEL
fi
echo "✅ Model hazır: $MODEL"
echo ""

# Test çalıştırma
echo "3️⃣ Hızlı test başlatılıyor..."
echo "📊 Dataset: msmarco (küçük örnek)"
echo "🔢 Seed: 42"
echo "⏱️  Tahmini süre: 5-10 dakika"
echo ""

mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark,ollama \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42 \
  -Dbenchmark.warmup-ratio=0.1 \
  -Dbenchmark.output-file=results/ollama_test.json \
  -Dllm.ollama.model=$MODEL \
  -Dcache.max-entries=5000

echo ""
echo "✅ Test tamamlandı!"
echo "📁 Sonuçlar: results/ollama_test.json"
echo ""
echo "📊 Sonuçları görüntülemek için:"
echo "   cat results/ollama_test.json | jq '.metrics'"
