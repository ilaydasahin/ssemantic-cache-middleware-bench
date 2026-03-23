#!/bin/bash
# Metrik Odaklı İzleme - Hit rate, latency, cost gibi metrikleri gösterir

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║          SEMANTIC CACHE BENCHMARK - METRİK İZLEME                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔍 Loglardan metrikler çıkarılıyor..."
echo ""

# Log dosyasını bul
LOG_FILE=$(ls -t logs/benchmark_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    LOG_FILE="logs/benchmark-current.log"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log dosyası bulunamadı"
    exit 1
fi

echo "📄 Log: $LOG_FILE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Fazları göster
echo ""
echo "📊 FAZLAR:"
grep -E "Phase [0-9]:|Strategy:" "$LOG_FILE" | tail -10

# Metrikleri göster
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 METRİKLER:"
grep -E "Hit Rate|Latency|Cost|Progress|queries processed" "$LOG_FILE" | tail -15

# Hataları göster
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  HATALAR:"
ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "   Toplam hata: $ERROR_COUNT"
    grep "ERROR" "$LOG_FILE" | tail -5
else
    echo "   ✅ Hata yok"
fi

# Uyarıları göster
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ UYARILAR:"
WARN_COUNT=$(grep -c "WARN" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$WARN_COUNT" -gt 0 ]; then
    echo "   Toplam uyarı: $WARN_COUNT"
    grep "WARN" "$LOG_FILE" | tail -5
else
    echo "   ✅ Uyarı yok"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔄 Canlı izleme için: tail -f $LOG_FILE"
echo "📊 Tüm metrikler için: ./watch_live.sh"
echo ""
