#!/bin/bash
# ============================================
# Otomatik Devam Etme Kurulumu (Cron)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🔧 Otomatik Devam Etme Sistemi Kurulumu"
echo ""
echo "Bu sistem:"
echo "  ✅ Her gün saat 09:00'da deneyi kontrol eder"
echo "  ✅ Tamamlanmamışsa otomatik devam ettirir"
echo "  ✅ Quota yenilendiğinde kaldığı yerden devam eder"
echo "  ✅ Tamamlandığında e-posta gönderir (opsiyonel)"
echo ""

# Cron job oluştur
CRON_JOB="0 9 * * * cd ${SCRIPT_DIR} && bash ${SCRIPT_DIR}/auto_resume_experiment.sh >> ${SCRIPT_DIR}/auto_resume.log 2>&1"

# Mevcut crontab'ı kontrol et
if crontab -l 2>/dev/null | grep -q "auto_resume_experiment.sh"; then
    echo "⚠️  Cron job zaten kurulu"
    echo ""
    read -p "Yeniden kurmak ister misiniz? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ İptal edildi"
        exit 0
    fi
    # Eski job'ı kaldır
    crontab -l 2>/dev/null | grep -v "auto_resume_experiment.sh" | crontab -
fi

# Yeni cron job ekle
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Cron job kuruldu!"
echo ""
echo "📋 Kurulum Detayları:"
echo "   - Çalışma zamanı: Her gün 09:00"
echo "   - Script: ${SCRIPT_DIR}/auto_resume_experiment.sh"
echo "   - Log: ${SCRIPT_DIR}/auto_resume.log"
echo ""
echo "🔍 Kontrol Komutları:"
echo "   - Cron listesi: crontab -l"
echo "   - Log takibi: tail -f ${SCRIPT_DIR}/auto_resume.log"
echo "   - Kaldırma: crontab -e (satırı silin)"
echo ""
echo "💡 İpucu: Bilgisayarınız kapalıysa çalışmaz!"
echo "   Sunucuda çalıştırmanız önerilir."
echo ""
