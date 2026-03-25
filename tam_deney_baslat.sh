#!/bin/bash

echo "═══════════════════════════════════════════════════════════"
echo "🚀 TAM KAPSAMLI DENEY BAŞLATILIYOR (16GB RAM Optimized)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# 1. Load API keys from .env
echo "📋 API keyleri yükleniyor..."
export GEMINI_API_KEYS=$(grep GEMINI_API_KEYS .env | cut -d'"' -f2)
KEY_COUNT=$(echo $GEMINI_API_KEYS | grep -o 'AIzaSy' | wc -l | tr -d ' ')

if [ "$KEY_COUNT" -lt 100 ]; then
    echo "❌ HATA: Yeterli API key yok ($KEY_COUNT < 100)"
    exit 1
fi

echo "✅ $KEY_COUNT API key yüklendi"
echo ""

# 2. Check prerequisites
echo "🔍 Ön kontroller..."
REDIS_STATUS=$(redis-cli ping 2>/dev/null || echo "FAIL")
if [ "$REDIS_STATUS" != "PONG" ]; then
    echo "❌ HATA: Redis çalışmıyor"
    echo "   Başlatmak için: redis-server &"
    exit 1
fi
echo "✅ Redis: $REDIS_STATUS"
echo ""

# 3. Clean old results
echo "🗑️  Eski loglar temizleniyor..."
rm -f experiment.log
rm -f .experiment_pid
echo "✅ Temizlendi"
echo ""

# 4. Set RAM optimization
echo "⚙️  RAM optimizasyonu ayarlanıyor..."
export MAVEN_OPTS="-Xms2G -Xmx4G -XX:+UseZGC"
echo "✅ JVM heap: 4GB max"
echo ""

# 5. Start experiment (background)
echo "🚀 Deney başlatılıyor..."
echo "   • Toplam deney: 218"
echo "   • Toplam LLM çağrısı: 217,000"
echo "   • Max paralel thread: 10"
echo "   • JVM heap: 4GB max"
echo "   • Tahmini süre: 3-4 saat (keyler tazeyse)"
echo ""

# Export keys and run
export GEMINI_API_KEYS
nohup ./run_full_benchmark_suite.sh > experiment.log 2>&1 &
EXPERIMENT_PID=$!
echo $EXPERIMENT_PID > .experiment_pid

sleep 5

# Verify it started
if ps -p $EXPERIMENT_PID > /dev/null; then
    echo "✅ Deney başlatıldı!"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "📊 DENEY BİLGİLERİ"
    echo "═══════════════════════════════════════════════════════════"
    echo "  • PID: $EXPERIMENT_PID"
    echo "  • Log: experiment.log"
    echo "  • Benchmark log: logs/benchmark-current.log"
    echo "  • API Keys: $KEY_COUNT keys"
    echo "  • RAM: 4GB heap (optimized for 16GB system)"
    echo ""
    echo "📋 TAKİP KOMUTLARI:"
    echo "  • tail -f experiment.log              # Ana log"
    echo "  • tail -f logs/benchmark-current.log  # Detaylı log"
    echo "  • ps -p $EXPERIMENT_PID               # Process durumu"
    echo "  • ls -lh results/*/                   # Sonuçlar"
    echo ""
    echo "🛑 DURDURMA:"
    echo "  • kill $EXPERIMENT_PID"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "🎯 Deney çalışıyor!"
    echo "═══════════════════════════════════════════════════════════"
else
    echo "❌ HATA: Deney başlatılamadı"
    echo ""
    echo "Log:"
    tail -20 experiment.log
    exit 1
fi
