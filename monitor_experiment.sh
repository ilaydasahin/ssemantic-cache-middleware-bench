#!/bin/bash
# ============================================
# Deney İzleme Paneli (Gelişmiş)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Deney İzleme Paneli - $(date +%H:%M:%S)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# State kontrolü
if [ ! -f ".experiment_state" ]; then
    echo "❌ Aktif deney bulunamadı"
    echo "   Başlatmak için: ./start_full_experiment.sh"
    exit 1
fi

RESULTS_DIR=$(cat .experiment_state)

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Sonuç klasörü bulunamadı: ${RESULTS_DIR}"
    exit 1
fi

# İlerleme hesapla
TOTAL_EXPECTED=108
COMPLETED=$(find "$RESULTS_DIR" -name "*.json" -type f | wc -l | tr -d ' ')
PERCENTAGE=$((COMPLETED * 100 / TOTAL_EXPECTED))
REMAINING=$((TOTAL_EXPECTED - COMPLETED))

echo "📊 İlerleme:"
echo "   Tamamlanan: ${COMPLETED}/${TOTAL_EXPECTED} (${PERCENTAGE}%)"
echo "   Kalan: ${REMAINING} deney"
echo ""

# İlerleme çubuğu
BAR_LENGTH=50
FILLED=$((PERCENTAGE * BAR_LENGTH / 100))
EMPTY=$((BAR_LENGTH - FILLED))
printf "   ["
printf "%${FILLED}s" | tr ' ' '█'
printf "%${EMPTY}s" | tr ' ' '░'
printf "] ${PERCENTAGE}%%\n"
echo ""

# Süreç durumu
if [ -f ".experiment_pid" ]; then
    PID=$(cat .experiment_pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Durum: Çalışıyor (PID: ${PID})"
        
        # CPU ve Memory kullanımı
        if command -v ps &> /dev/null; then
            CPU=$(ps -p "$PID" -o %cpu= 2>/dev/null | tr -d ' ')
            MEM=$(ps -p "$PID" -o %mem= 2>/dev/null | tr -d ' ')
            echo "   CPU: ${CPU}% | Memory: ${MEM}%"
        fi
    else
        echo "⚠️  Durum: Durdurulmuş"
        echo "   Devam ettirmek için: ./auto_resume_experiment.sh"
    fi
else
    echo "⚠️  Durum: Başlatılmamış"
fi
echo ""

# En son log dosyası
LATEST_LOG=$(ls -t "$RESULTS_DIR"/*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "📝 Son Log: $(basename "$LATEST_LOG")"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 Son 5 Tamamlanan Deney:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    grep "✅" "$LATEST_LOG" 2>/dev/null | tail -5 || echo "   Henüz tamamlanan deney yok"
    echo ""
fi

# Tahmini kalan süre
if [ "$COMPLETED" -gt 0 ]; then
    # API key kapasitesi
    if [ -f .env ]; then
        source .env
        KEY_COUNT=$(echo "${GEMINI_API_KEYS}" | tr ',' '\n' | wc -l | tr -d ' ')
        DAILY_CAPACITY=$((KEY_COUNT * 1450))
        
        if [ "$REMAINING" -gt "$DAILY_CAPACITY" ]; then
            DAYS_LEFT=$(( (REMAINING + DAILY_CAPACITY - 1) / DAILY_CAPACITY ))
            echo "⏱️  Tahmini Kalan Süre: ${DAYS_LEFT} gün"
        else
            HOURS_LEFT=$((REMAINING * 2 / 60))  # ~2 dakika/deney
            echo "⏱️  Tahmini Kalan Süre: ~${HOURS_LEFT} saat"
        fi
        echo ""
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Komutlar:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Canlı takip: tail -f ${LATEST_LOG}"
echo "   Sonuçlar: ls -lh ${RESULTS_DIR}/*.json"
echo "   Durdur: kill \$(cat .experiment_pid)"
echo ""
