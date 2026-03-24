#!/bin/bash

echo "═══════════════════════════════════════════════════════════"
echo "🚀 TAM KAPSAMLI DENEY BAŞLATILIYOR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Load API keys from .env
export GEMINI_API_KEYS=$(grep GEMINI_API_KEYS .env | cut -d'"' -f2)
KEY_COUNT=$(echo $GEMINI_API_KEYS | grep -o 'AIzaSy' | wc -l | tr -d ' ')

echo "✅ Ön Kontrol:"
echo "  • Redis: $(redis-cli ping 2>/dev/null || echo 'HATA')"
echo "  • API Keys: $KEY_COUNT keys"
echo "  • Java: $(java -version 2>&1 | head -1)"
echo ""

if [ "$KEY_COUNT" -lt 100 ]; then
    echo "❌ HATA: Yeterli API key yok ($KEY_COUNT < 100)"
    exit 1
fi

echo "📊 Deney Bilgileri:"
echo "  • Toplam deney: 218 konfigürasyon"
echo "  • Toplam LLM çağrısı: 217,000"
echo "  • Tahmini süre: 1 saat 29 dakika"
echo "  • Günlük kapasite: 234,900 çağrı"
echo "  • Buffer: +8%"
echo ""

echo "🔨 Proje derleniyor..."
mvn clean package -DskipTests -q

if [ $? -ne 0 ]; then
    echo "❌ HATA: Proje derlenemedi"
    exit 1
fi

echo "✅ Proje derlendi"
echo ""

echo "🚀 Deney başlatılıyor (arka planda)..."
nohup ./run_full_benchmark_suite.sh > experiment.log 2>&1 &
EXPERIMENT_PID=$!
echo $EXPERIMENT_PID > .experiment_pid

sleep 3

if ps -p $EXPERIMENT_PID > /dev/null; then
    echo "✅ Deney başlatıldı!"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "📊 DENEY BİLGİLERİ"
    echo "═══════════════════════════════════════════════════════════"
    echo "  • PID: $EXPERIMENT_PID"
    echo "  • Log dosyası: experiment.log"
    echo "  • Sonuç dizini: results/$(date +%Y%m%d)_*/"
    echo ""
    echo "📋 TAKİP KOMUTLARI:"
    echo "  • İlerleme: tail -f experiment.log"
    echo "  • Process: ps -p $EXPERIMENT_PID"
    echo "  • Durdur: kill $EXPERIMENT_PID"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "🎯 Deney çalışıyor! Terminal'i kapatabilirsin."
    echo "═══════════════════════════════════════════════════════════"
else
    echo "❌ HATA: Deney başlatılamadı"
    echo "Log: $(tail -20 experiment.log)"
    exit 1
fi
