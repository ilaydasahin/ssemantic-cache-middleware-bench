#!/bin/bash
# ============================================
# Arkaplanda Tam Deney Başlatıcı (Çok Günlük Destek)
# ============================================

echo "🚀 Kapsamlı Benchmark Deneyi Başlatılıyor..."
echo ""

# 1. Ortam Kontrolü
echo "📋 Ön Kontroller:"

# Redis kontrolü
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis çalışmıyor. Başlatılıyor..."
    redis-server --daemonize yes
    sleep 2
    if ! redis-cli ping > /dev/null 2>&1; then
        echo "❌ Redis başlatılamadı. Manuel başlatın: redis-server"
        exit 1
    fi
fi
echo "✅ Redis çalışıyor"

# API Keys kontrolü
if [ -f .env ]; then
    source .env
fi

if [ -z "${GEMINI_API_KEYS}" ]; then
    echo "❌ GEMINI_API_KEYS bulunamadı!"
    echo "   .env dosyasını kontrol edin"
    exit 1
fi

KEY_COUNT=$(echo "${GEMINI_API_KEYS}" | tr ',' '\n' | wc -l | tr -d ' ')
DAILY_CAPACITY=$((KEY_COUNT * 1450))  # Safe buffer: 1450 per key
TOTAL_NEEDED=108000  # 108 configs × 1000 queries
DAYS_NEEDED=$(( (TOTAL_NEEDED + DAILY_CAPACITY - 1) / DAILY_CAPACITY ))

echo "✅ ${KEY_COUNT} API key bulundu"
echo ""
echo "📊 Kapasite Analizi:"
echo "   - Günlük kapasite: ~${DAILY_CAPACITY} çağrı"
echo "   - Toplam ihtiyaç: ~${TOTAL_NEEDED} çağrı"
echo "   - Tahmini süre: ${DAYS_NEEDED} gün"
echo ""

if [ $DAYS_NEEDED -gt 1 ]; then
    echo "⚠️  DİKKAT: Deney ${DAYS_NEEDED} gün sürecek!"
    echo "   - Her gün otomatik devam edecek"
    echo "   - Tamamlanan deneyler atlanacak"
    echo "   - Bilgisayarı kapatabilirsiniz (cron ile devam eder)"
    echo ""
fi

# 3. Log dosyası hazırla
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="experiment_${TIMESTAMP}.log"
RESULTS_DIR="results/${TIMESTAMP}"

# Devam eden deney var mı kontrol et
if [ -f ".experiment_state" ]; then
    EXISTING_DIR=$(cat .experiment_state)
    if [ -d "$EXISTING_DIR" ]; then
        echo "🔄 Devam eden deney bulundu: ${EXISTING_DIR}"
        RESULTS_DIR="$EXISTING_DIR"
        LOG_FILE="${EXISTING_DIR}/experiment_continued_$(date +%Y%m%d_%H%M%S).log"
    fi
fi

# State dosyasını kaydet
echo "$RESULTS_DIR" > .experiment_state

echo "📁 Sonuçlar: ${RESULTS_DIR}"
echo "📝 Log dosyası: ${LOG_FILE}"
echo ""

# 4. Arkaplanda başlat
echo "🔄 Deney arkaplanda başlatılıyor..."
nohup bash run_full_benchmark_suite.sh "${RESULTS_DIR}" > "${LOG_FILE}" 2>&1 &
PID=$!

# PID'yi kaydet
echo "$PID" > .experiment_pid

echo "✅ Deney başlatıldı!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 İzleme Komutları:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  # Canlı log takibi:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "  # İlerleme özeti:"
echo "  ./monitor_experiment.sh"
echo ""
echo "  # Süreç durumu:"
echo "  ps aux | grep ${PID}"
echo ""
echo "  # Durdurma (gerekirse):"
echo "  kill ${PID}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Process ID: ${PID}"
echo "Deney tamamlandığında sonuçlar ${RESULTS_DIR} klasöründe olacak"
echo ""
echo "💡 Bilgisayarı kapatabilirsiniz - cron ile otomatik devam eder"
echo "   Kurulum için: ./setup_auto_resume.sh"
echo ""
