#!/bin/bash
# Anlık Canlı İzleme - Her 2 saniyede güncellenir

clear

while true; do
    # Ekranı temizle ve başa dön
    tput cup 0 0
    
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║          SEMANTIC CACHE BENCHMARK - CANLI İZLEME                          ║"
    echo "║          Güncelleme: $(date '+%H:%M:%S')                                           ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Süreç durumu
    JAVA_PROCS=$(ps aux | grep -E "semantic-cache-benchmark" | grep -v grep | wc -l | xargs)
    if [ "$JAVA_PROCS" -gt 0 ]; then
        echo "🟢 Durum: ÇALIŞIYOR"
        
        # CPU ve Memory kullanımı
        ps aux | grep -E "semantic-cache-benchmark" | grep -v grep | awk '{printf "   CPU: %s%%  |  Memory: %s%%\n", $3, $4}' | head -1
    else
        echo "🔴 Durum: DURDURULDU"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 SON LOGLAR:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Son 20 satır log
    tail -20 logs/benchmark-current.log 2>/dev/null | tail -15 || echo "   Log dosyası bekleniyor..."
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📁 SONUÇLAR:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Sonuç sayısı
    RESULT_COUNT=$(find results -name "*.json" 2>/dev/null | wc -l | xargs)
    echo "   Tamamlanan: $RESULT_COUNT deney"
    
    if [ "$RESULT_COUNT" -gt 0 ]; then
        echo ""
        echo "   Son 3 sonuç:"
        ls -lht results/*.json 2>/dev/null | head -3 | awk '{printf "   • %s (%s)\n", $9, $5}'
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⌨️  Çıkmak için: Ctrl+C"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 2 saniye bekle
    sleep 2
done
