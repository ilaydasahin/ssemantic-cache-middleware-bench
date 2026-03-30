#!/bin/bash

# ============================================
# Tüm Eski Dosyaları Temizleme Scripti
# ============================================

echo "🧹 Temizlik başlatılıyor..."
echo ""

# Eski sonuçları temizle
echo "1️⃣ Eski deney sonuçlarını temizleme..."
rm -rf results/*.json results/*.jsonl results/*.log results/*/
find results -type f ! -name '.gitkeep' -delete 2>/dev/null
echo "✅ Sonuçlar temizlendi"

# Eski logları temizle
echo "2️⃣ Eski logları temizleme..."
find logs -type f ! -name '.gitkeep' -delete 2>/dev/null
echo "✅ Loglar temizlendi"

# Checkpoint'leri temizle
echo "3️⃣ Checkpoint'leri temizleme..."
find checkpoints -type f -delete 2>/dev/null
echo "✅ Checkpoint'ler temizlendi"

# Archive'i temizle
echo "4️⃣ Archive'i temizleme..."
find results_archive -type f -delete 2>/dev/null
echo "✅ Archive temizlendi"

# PID ve nohup dosyalarını temizle
echo "5️⃣ Geçici dosyaları temizleme..."
rm -f .experiment_pid nohup.out
echo "✅ Geçici dosyalar temizlendi"

# Maven cache temizle (opsiyonel)
echo "6️⃣ Maven cache temizleme..."
mvn clean -q
echo "✅ Maven cache temizlendi"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 TEMİZLİK TAMAMLANDI!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Sistem temiz ve hazır! 🚀"
echo ""
echo "Başlatmak için:"
echo "  ./run_ollama_test.sh        # Hızlı test"
echo "  ./run_ollama_full_benchmark.sh  # Tam benchmark"
