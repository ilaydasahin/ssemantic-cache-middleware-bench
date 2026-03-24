#!/bin/bash

echo "═══════════════════════════════════════════════════════════"
echo "🚀 TAM KAPSAMLI DENEY BAŞLATILIYOR"
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
echo "🗑️  Eski sonuçlar temizleniyor..."
rm -rf results/20260324_*
rm -f experiment.log
rm -f nohup.out
rm -f .experiment_pid
echo "✅ Temizlendi"
echo ""

# 4. Build project
echo "🔨 Proje derleniyor..."
mvn clean package -DskipTests -q
if [ $? -ne 0 ]; then
    echo "❌ HATA: Proje derlenemedi"
    exit 1
fi
echo "✅ Proje derlendi"
echo ""

# 5. Start experiment
echo "🚀 Deney başlatılıyor..."
echo "   • Toplam deney: 218"
echo "   • Toplam LLM çağrısı: 217,000"
echo "   • Tahmini süre: 1.5 saat"
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
    echo "  • API Keys: $KEY_COUNT keys"
    echo ""
    echo "📋 TAKİP KOMUTLARI:"
    echo "  • tail -f experiment.log"
    echo "  • ps -p $EXPERIMENT_PID"
    echo "  • ls -lh results/*/"
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
