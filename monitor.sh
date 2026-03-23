#!/bin/bash
# Deney İzleme Scripti

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     SEMANTIC CACHE BENCHMARK - DURUM İZLEME               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Java süreçlerini kontrol et
JAVA_PROCS=$(ps aux | grep -E "semantic-cache-benchmark" | grep -v grep | wc -l | xargs)

if [ "$JAVA_PROCS" -gt 0 ]; then
    echo "✅ Deney çalışıyor ($JAVA_PROCS süreç aktif)"
    echo ""
    
    # Son log satırları
    echo "📊 Son Durum:"
    echo "─────────────────────────────────────────────────────────────"
    tail -15 logs/benchmark-current.log 2>/dev/null | grep -E "Progress|Phase|Metrics|Dataset|Strategy|Hit Rate|Latency|Cost|ERROR|WARNING" || echo "   Log henüz oluşmadı..."
    echo "─────────────────────────────────────────────────────────────"
    echo ""
    
    # Sonuç dosyaları
    RESULT_COUNT=$(find results -name "*.json" 2>/dev/null | wc -l | xargs)
    echo "📁 Tamamlanan Deneyler: $RESULT_COUNT"
    
    if [ "$RESULT_COUNT" -gt 0 ]; then
        echo ""
        echo "Son 5 sonuç:"
        ls -lht results/*.json 2>/dev/null | head -5 | awk '{print "   " $9 " (" $5 ")"}'
    fi
    
else
    echo "⚠️  Deney çalışmıyor"
    echo ""
    echo "Başlatmak için:"
    echo "   bash run_background.sh"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "🔄 Canlı izleme: tail -f logs/benchmark-current.log"
echo "📊 Sonuçlar: ls -lh results/"
echo "🛑 Durdurmak: pkill -f semantic-cache-benchmark"
echo "─────────────────────────────────────────────────────────────"
