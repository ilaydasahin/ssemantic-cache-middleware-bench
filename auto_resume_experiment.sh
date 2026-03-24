#!/bin/bash
# ============================================
# Otomatik Deney Devam Ettirici
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Otomatik Devam Kontrolü - $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# State dosyasını kontrol et
if [ ! -f ".experiment_state" ]; then
    echo "ℹ️  Devam eden deney yok"
    exit 0
fi

RESULTS_DIR=$(cat .experiment_state)

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Sonuç klasörü bulunamadı: ${RESULTS_DIR}"
    rm -f .experiment_state
    exit 1
fi

# Tamamlanma kontrolü
TOTAL_EXPECTED=108
COMPLETED=$(find "$RESULTS_DIR" -name "*.json" -type f | wc -l | tr -d ' ')

echo "📊 İlerleme: ${COMPLETED}/${TOTAL_EXPECTED} deney tamamlandı"

if [ "$COMPLETED" -ge "$TOTAL_EXPECTED" ]; then
    echo "✅ TÜM DENEYLER TAMAMLANDI!"
    echo ""
    echo "📁 Sonuçlar: ${RESULTS_DIR}"
    echo "📊 Toplam: ${COMPLETED} deney"
    echo ""
    
    # Bildirim gönder (opsiyonel)
    if command -v osascript &> /dev/null; then
        osascript -e 'display notification "Tüm deneyler tamamlandı!" with title "Benchmark Bitti"'
    fi
    
    # State dosyasını temizle
    rm -f .experiment_state .experiment_pid
    
    echo "🎉 Deney başarıyla tamamlandı!"
    exit 0
fi

# Çalışan süreç var mı kontrol et
if [ -f ".experiment_pid" ]; then
    PID=$(cat .experiment_pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Deney zaten çalışıyor (PID: ${PID})"
        echo "   Müdahale edilmedi"
        exit 0
    else
        echo "⚠️  Önceki süreç sonlanmış (PID: ${PID})"
        rm -f .experiment_pid
    fi
fi

# Deneyi devam ettir
echo "🔄 Deney devam ettiriliyor..."
echo ""

# Redis kontrolü
if ! redis-cli ping > /dev/null 2>&1; then
    echo "⚠️  Redis başlatılıyor..."
    redis-server --daemonize yes
    sleep 2
fi

# API Keys yükle
if [ -f .env ]; then
    source .env
fi

# Deneyi başlat
LOG_FILE="${RESULTS_DIR}/experiment_resumed_$(date +%Y%m%d_%H%M%S).log"
nohup bash run_full_benchmark_suite.sh "${RESULTS_DIR}" > "${LOG_FILE}" 2>&1 &
NEW_PID=$!

echo "$NEW_PID" > .experiment_pid

echo "✅ Deney yeniden başlatıldı!"
echo "   PID: ${NEW_PID}"
echo "   Log: ${LOG_FILE}"
echo ""
echo "📊 Kalan: $((TOTAL_EXPECTED - COMPLETED)) deney"
echo ""
