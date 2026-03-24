# 🚀 DENEY BAŞLATMA REHBERİ - TERMINAL ADIMLARI

## ✅ ÖN HAZIRLIK KONTROL LİSTESİ

Başlamadan önce kontrol et:

```bash
# 1. Redis çalışıyor mu?
redis-cli ping
# Beklenen çıktı: PONG

# 2. API keyleri yüklü mü?
grep -c "AIzaSy" .env
# Beklenen çıktı: 162

# 3. Java kurulu mu?
java -version
# Beklenen çıktı: Java 17 veya üzeri

# 4. Maven kurulu mu?
mvn -version
# Beklenen çıktı: Maven 3.x
```

---

## 🎯 ADIM ADIM BAŞLATMA

### ADIM 1: Terminal Aç

```bash
# Proje dizinine git
cd /path/to/semantic-cache-benchmark

# Veya mevcut dizindeysen:
pwd
# Çıktı: /Users/hlmacos/Desktop/güncel (veya proje dizinin)
```

### ADIM 2: Redis'i Başlat (Eğer çalışmıyorsa)

```bash
# Redis'i başlat
redis-server &

# Veya ayrı bir terminalde:
redis-server

# Kontrol et:
redis-cli ping
# Çıktı: PONG ✅
```

### ADIM 3: Projeyi Derle (İlk kez veya değişiklik yaptıysan)

```bash
# Projeyi derle
mvn clean package -DskipTests

# Beklenen çıktı:
# [INFO] BUILD SUCCESS
# [INFO] Total time: ~30 seconds
```

### ADIM 4: Deneyi Başlat

```bash
# Tam kapsamlı deneyi başlat
./start_full_experiment.sh

# Beklenen çıktı:
# 🚀 Starting full benchmark experiment...
# 📊 Experiment will run in background
# 📝 Output: nohup.out
# 🔍 Monitor: ./monitor_experiment.sh
# ✅ Experiment started with PID: 12345
```

**NOT:** Eğer script çalıştırma izni yoksa:
```bash
chmod +x start_full_experiment.sh
./start_full_experiment.sh
```

---

## 📊 ADIM 5: İlerlemeyi Takip Et

### Seçenek A: Monitor Script Kullan (Önerilen)

```bash
# İlerleme takip script'ini çalıştır
./monitor_experiment.sh

# Beklenen çıktı:
# ═══════════════════════════════════════════
# 🔍 EXPERIMENT MONITOR
# ═══════════════════════════════════════════
# Status: Running ✅
# Progress: 45/218 experiments (20%)
# LLM Calls: 45,000/217,000 (20%)
# Keys Used: 31/162
# Last Activity: 2 minutes ago
# Estimated Time Remaining: 1h 10m
# ═══════════════════════════════════════════
```

### Seçenek B: Log Dosyasını İzle

```bash
# Son 20 satırı göster
tail -20 nohup.out

# Gerçek zamanlı takip (Ctrl+C ile çık)
tail -f nohup.out

# Beklenen çıktı:
# [1/218] BASELINE: msmarco (Seed 42) ✅
# [2/218] BASELINE: msmarco (Seed 123) ✅
# Progress: 1000 total calls across 162 keys (avg 6/key)
# [3/218] BASELINE: msmarco (Seed 456) ✅
# ...
```

### Seçenek C: Process Kontrolü

```bash
# Deney çalışıyor mu?
ps aux | grep "semantic-cache-benchmark"

# PID'yi kontrol et
cat .experiment_pid
# Çıktı: 12345

# Process detayları
ps -p $(cat .experiment_pid)
```

---

## ⏸️ ADIM 6: Deney Yönetimi (Opsiyonel)

### Deneyi Durdur (Gerekirse)

```bash
# Deneyi durdur
kill $(cat .experiment_pid)

# Veya zorla durdur
kill -9 $(cat .experiment_pid)

# NOT: Checkpoint sayesinde kaldığı yerden devam edebilir
```

### Deneyi Yeniden Başlat (Checkpoint'ten devam)

```bash
# Aynı komutla yeniden başlat
./start_full_experiment.sh

# Sistem otomatik olarak:
# ✓ Tamamlanan deneyleri atlar
# ✓ Kaldığı yerden devam eder
# ✓ Aynı sonuç dizinini kullanır
```

---

## 🎉 ADIM 7: Deney Tamamlandığında

### Sonuçları Kontrol Et

```bash
# Deney tamamlandı mı?
./monitor_experiment.sh

# Beklenen çıktı:
# Status: Completed ✅
# Progress: 218/218 experiments (100%)
# LLM Calls: 217,000/217,000 (100%)
# Total Time: 1h 29m

# Sonuç dosyalarını listele
ls -lh results/*/

# Beklenen çıktı:
# results/20260324_100000/
#   ├── baseline_msmarco_s42.json
#   ├── baseline_msmarco_s123.json
#   ├── msmarco_minilm_t0.75_s42.json
#   ├── ...
#   └── table4_summary.csv
```

### İstatistik Analizi

```bash
# Analiz sonuçlarını görüntüle
cat results/*/table4_summary.csv

# Veya Excel/Numbers ile aç
open results/*/table4_summary.csv
```

---

## 🔧 SORUN GİDERME

### Problem 1: Redis Çalışmıyor

```bash
# Hata: "ERROR: Redis is not running"

# Çözüm:
redis-server &
redis-cli ping  # PONG dönmeli
```

### Problem 2: API Keys Yüklenmedi

```bash
# Hata: "WARNING: GEMINI_API_KEYS not set"

# Kontrol:
cat .env | grep GEMINI_API_KEYS

# Çözüm: .env dosyasını kontrol et
# 162 key olmalı
```

### Problem 3: Port Zaten Kullanımda

```bash
# Hata: "Port 8080 already in use"

# Çözüm: Eski process'i bul ve durdur
lsof -ti:8080 | xargs kill -9

# Veya farklı port kullan (otomatik: --server.port=0)
```

### Problem 4: Deney Durdu

```bash
# Kontrol:
./monitor_experiment.sh

# Eğer "Waiting for quota reset" görüyorsan:
# → Normal, sistem otomatik bekliyor
# → PST gece yarısı devam edecek

# Eğer "Stopped" görüyorsan:
# → Yeniden başlat: ./start_full_experiment.sh
# → Checkpoint'ten devam edecek
```

### Problem 5: Log Dosyası Çok Büyüdü

```bash
# Log dosyası boyutunu kontrol et
ls -lh nohup.out

# Son 100 satırı başka dosyaya kaydet
tail -100 nohup.out > recent_logs.txt

# Eski log'u temizle (dikkatli!)
> nohup.out  # Dosyayı boşalt
```

---

## 📋 HIZLI KOMUT REFERANSı

### Başlatma
```bash
./start_full_experiment.sh
```

### Takip
```bash
./monitor_experiment.sh          # Özet bilgi
tail -f nohup.out                # Gerçek zamanlı log
ps -p $(cat .experiment_pid)     # Process durumu
```

### Yönetim
```bash
kill $(cat .experiment_pid)      # Durdur
./start_full_experiment.sh       # Yeniden başlat (checkpoint'ten)
```

### Sonuçlar
```bash
ls results/*/                    # Sonuç dosyaları
cat results/*/table4_summary.csv # Analiz özeti
```

---

## 🎯 TAM ÖRNEK SENARYO

İşte baştan sona tam bir örnek:

```bash
# 1. Proje dizinine git
cd ~/Desktop/güncel

# 2. Redis'i kontrol et
redis-cli ping
# Çıktı: PONG ✅

# 3. API key sayısını kontrol et
grep -c "AIzaSy" .env
# Çıktı: 162 ✅

# 4. Deneyi başlat
./start_full_experiment.sh
# Çıktı: ✅ Experiment started with PID: 12345

# 5. İlerlemeyi takip et (5 dakika sonra)
./monitor_experiment.sh
# Çıktı: Progress: 12/218 experiments (5%)

# 6. Log'u kontrol et
tail -20 nohup.out
# Çıktı: [12/218] LOCAL: msmarco/minilm @ θ=0.75 (Seed 42) ✅

# 7. Kahve molası ☕ (1.5 saat)

# 8. Sonucu kontrol et
./monitor_experiment.sh
# Çıktı: Status: Completed ✅

# 9. Sonuçları görüntüle
ls results/*/
cat results/*/table4_summary.csv

# 10. Bitti! 🎉
```

---

## 💡 İPUÇLARI

### İpucu 1: Arka Planda Çalıştır
```bash
# Deney zaten arka planda çalışıyor (nohup ile)
# Terminal'i kapatabilirsin
# Bilgisayarı kapatma!
```

### İpucu 2: Otomatik Resume Kur
```bash
# Her gün otomatik kontrol için
./setup_auto_resume.sh

# Cron job kuruldu:
# Her gün 09:00'da kontrol eder
# Yarım kalan deney varsa devam ettirir
```

### İpucu 3: Çoklu Terminal Kullan
```bash
# Terminal 1: Deney çalışıyor
./start_full_experiment.sh

# Terminal 2: Log takibi
tail -f nohup.out

# Terminal 3: Periyodik kontrol
watch -n 60 './monitor_experiment.sh'  # Her 60 saniyede
```

### İpucu 4: Bildirim Al (macOS)
```bash
# Deney bitince bildirim al
./start_full_experiment.sh && \
  osascript -e 'display notification "Deney tamamlandı!" with title "Semantic Cache"'
```

---

## 🎉 HAZIR!

Artık deneyi başlatmaya hazırsın!

**Tek yapman gereken:**
```bash
./start_full_experiment.sh
```

Sistem geri kalanını halleder! 🚀

**Sorular?**
- `TAM_KAPSAMLI_DENEY_FINAL.md` - Deney detayları
- `OTOMATIK_KEY_YONETIMI.md` - Key rotasyon sistemi
- `SISTEM_AKIS_DIYAGRAMI.md` - Görsel akış
