# ⚡ HIZLI BAŞLANGIÇ - TEK SAYFA

## 🚀 3 ADIMDA BAŞLAT

### 1️⃣ Ön Kontrol (30 saniye)
```bash
redis-cli ping                    # PONG dönmeli ✅
grep -c "AIzaSy" .env            # 162 olmalı ✅
java -version                     # Java 17+ olmalı ✅
```

### 2️⃣ Başlat (1 komut)
```bash
./start_full_experiment.sh
```

### 3️⃣ Takip Et
```bash
./monitor_experiment.sh          # Her 5 dakikada kontrol et
```

---

## 📊 BEKLENEN ÇIKTILAR

### Başlatma Çıktısı
```
🚀 Starting full benchmark experiment...
📊 Total experiments: 218
📊 Total LLM calls: 217,000
⏱️  Estimated time: 1h 29m
✅ Experiment started with PID: 12345
📝 Output: nohup.out
🔍 Monitor: ./monitor_experiment.sh
```

### İlerleme Çıktısı
```
═══════════════════════════════════════════
🔍 EXPERIMENT MONITOR
═══════════════════════════════════════════
Status: Running ✅
Progress: 45/218 experiments (20%)
LLM Calls: 45,000/217,000 (20%)
Keys Used: 31/162
Last Activity: 2 minutes ago
Estimated Time Remaining: 1h 10m
═══════════════════════════════════════════
```

### Tamamlanma Çıktısı
```
Status: Completed ✅
Progress: 218/218 experiments (100%)
LLM Calls: 217,000/217,000 (100%)
Total Time: 1h 29m
Results: results/20260324_100000/
```

---

## 🔧 SORUN GİDERME (Hızlı)

| Problem | Çözüm |
|---------|-------|
| Redis çalışmıyor | `redis-server &` |
| API keys yok | `.env` dosyasını kontrol et |
| Port kullanımda | `lsof -ti:8080 \| xargs kill -9` |
| Deney durdu | `./start_full_experiment.sh` (checkpoint'ten devam) |
| Log çok büyük | `tail -100 nohup.out > recent.txt` |

---

## 📋 KOMUT REFERANSI

```bash
# BAŞLATMA
./start_full_experiment.sh              # Deneyi başlat

# TAKİP
./monitor_experiment.sh                 # Özet bilgi
tail -f nohup.out                       # Gerçek zamanlı log
ps -p $(cat .experiment_pid)            # Process durumu

# YÖNETİM
kill $(cat .experiment_pid)             # Durdur
./start_full_experiment.sh              # Yeniden başlat

# SONUÇLAR
ls results/*/                           # Sonuç dosyaları
cat results/*/table4_summary.csv        # Analiz özeti
```

---

## ⏱️ ZAMAN ÇİZELGESİ

```
00:00 → Deney başladı
00:30 → %33 tamamlandı (~73 deney)
01:00 → %67 tamamlandı (~146 deney)
01:29 → %100 tamamlandı (218 deney) ✅
```

---

## 🎯 ÖNEMLİ NOTLAR

✅ **Yapman Gerekenler:**
- Deneyi başlat: `./start_full_experiment.sh`
- Periyodik kontrol: `./monitor_experiment.sh`
- Bilgisayarı açık tut

❌ **Yapman Gerekmeyenler:**
- Manuel key değiştirme (otomatik)
- Quota reset bekleme (otomatik)
- Hata müdahalesi (otomatik)
- Checkpoint yönetimi (otomatik)

---

## 🎉 BAŞARI!

Deney tamamlandığında:
```bash
./monitor_experiment.sh
# Status: Completed ✅

ls results/*/
# 218 JSON dosyası + 1 CSV analiz dosyası
```

---

## 📖 DETAYLI DOKÜMANTASYON

- `TERMINAL_BASLAT_REHBERI.md` - Adım adım rehber
- `TAM_KAPSAMLI_DENEY_FINAL.md` - Deney detayları
- `OTOMATIK_KEY_YONETIMI.md` - Key rotasyon sistemi

---

**TEK KOMUT:**
```bash
./start_full_experiment.sh
```

**Sistem geri kalanını halleder!** 🚀
