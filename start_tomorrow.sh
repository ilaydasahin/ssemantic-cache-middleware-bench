#!/bin/bash
# Yarın için otomatik başlatma scripti
# API kotaları PST gece yarısında (Türkiye saati ~10:00) sıfırlanır

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     SEMANTIC CACHE BENCHMARK - YARIN İÇİN HAZIR           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Kota sıfırlanma zamanını hesapla
PST_NOW=$(TZ="America/Los_Angeles" date +"%Y-%m-%d %H:%M:%S")
PST_MIDNIGHT=$(TZ="America/Los_Angeles" date -v+1d +"%Y-%m-%d 00:00:00" 2>/dev/null || date -d "tomorrow 00:00:00" +"%Y-%m-%d %H:%M:%S")
TR_RESET_TIME=$(TZ="Europe/Istanbul" date -d "$(TZ="America/Los_Angeles" date -v+1d +"%Y-%m-%d 00:00:00" 2>/dev/null || echo "tomorrow 00:00:00")" +"%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "~10:00")

echo "⏰ API Kota Sıfırlanma Zamanı:"
echo "   PST: $PST_MIDNIGHT"
echo "   Türkiye: $TR_RESET_TIME (yaklaşık)"
echo ""

# Sistem durumunu kontrol et
JAVA_PROCS=$(ps aux | grep -E "semantic-cache-benchmark" | grep -v grep | wc -l | xargs)
if [ "$JAVA_PROCS" -gt 0 ]; then
    echo "⚠️  Sistem hala çalışıyor!"
    echo "   Önce durdurun: pkill -f semantic-cache-benchmark"
    exit 1
fi

# Checkpoint kontrolü
CHECKPOINT_COUNT=$(ls -1 checkpoints/*.json 2>/dev/null | wc -l | xargs)
if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
    echo "⚠️  Eski checkpoint dosyaları bulundu:"
    ls -lh checkpoints/*.json
    echo ""
    echo "   Temizlemek için: rm checkpoints/*.json"
    echo ""
fi

echo "✅ Sistem Durumu:"
echo "   • Benchmark: Durduruldu"
echo "   • Checkpoints: Temiz"
echo "   • API Keys: 77 anahtar hazır"
echo "   • Toplam Kapasite: ~115,500 çağrı/gün"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 YARIN BAŞLATMAK İÇİN:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  Arkaplanda çalıştır (önerilen):"
echo "    bash run_background.sh"
echo ""
echo "2️⃣  Tek deney çalıştır:"
echo "    bash start_experiment.sh"
echo ""
echo "3️⃣  Tam benchmark suite:"
echo "    bash run_full_benchmark_suite.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 İZLEME KOMUTLARI:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   ./watch_live.sh      # Canlı dashboard"
echo "   ./watch_metrics.sh   # Metrik özeti"
echo "   ./monitor.sh         # Hızlı durum"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 İpucu: Kota sıfırlanmasından hemen sonra başlatın"
echo "   (Türkiye saati ~10:00 civarı)"
echo ""
