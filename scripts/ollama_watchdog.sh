#!/bin/bash
# Ollama Watchdog - Servisin sürekli çalışmasını garanti eder
# Kullanım: bash scripts/ollama_watchdog.sh &

OLLAMA_URL="http://localhost:11434"
CHECK_INTERVAL=60  # 60 saniyede bir kontrol
MAX_RESTART_ATTEMPTS=5
RESTART_COUNT=0

echo "🔍 Ollama Watchdog başlatıldı"
echo "   URL: $OLLAMA_URL"
echo "   Kontrol aralığı: ${CHECK_INTERVAL}s"
echo ""

while true; do
    # Ollama'nın yanıt verip vermediğini kontrol et
    if curl -s --max-time 5 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        # Başarılı - restart counter'ı sıfırla
        if [ $RESTART_COUNT -gt 0 ]; then
            echo "✅ Ollama tekrar çalışıyor ($(date))"
            RESTART_COUNT=0
        fi
    else
        # Başarısız - restart gerekli
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "⚠️  Ollama yanıt vermiyor! (Deneme: $RESTART_COUNT/$MAX_RESTART_ATTEMPTS) - $(date)"
        
        if [ $RESTART_COUNT -ge $MAX_RESTART_ATTEMPTS ]; then
            echo "❌ Maksimum restart denemesi aşıldı. Manuel müdahale gerekli!"
            echo "   Lütfen kontrol edin: ollama serve"
            exit 1
        fi
        
        echo "🔄 Ollama yeniden başlatılıyor..."
        
        # Mevcut Ollama process'lerini temizle
        pkill -9 ollama 2>/dev/null
        sleep 3
        
        # Ollama'yı yeniden başlat
        nohup ollama serve > /tmp/ollama_watchdog.log 2>&1 &
        sleep 10
        
        # Başarılı başlatıldı mı kontrol et
        if curl -s --max-time 5 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
            echo "✅ Ollama başarıyla yeniden başlatıldı"
        else
            echo "❌ Ollama başlatılamadı, bir sonraki denemede tekrar denenecek"
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
