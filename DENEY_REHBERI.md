# 🚀 Kapsamlı Deney Rehberi (Çok Günlük)

## 📊 Deney Özeti

**Toplam İhtiyaç**: ~108,000 LLM çağrısı
**Sizin Kapasiteniz**: 78 key × 1,450 çağrı/gün = **113,100 çağrı/gün**
**Tahmini Süre**: **1-2 gün** (quota yenilenmesiyle)

## ✅ Sorunuzun Cevapları

### 1. Birkaç gün sürecek mi?
**Evet**, ama sadece 1-2 gün. 78 key ile günlük kapasiteniz yeterli.

### 2. Yarım kalır mı?
**HAYIR!** Sistem:
- ✅ Tamamlanan deneyleri kaydeder
- ✅ Quota dolunca güvenle durur
- ✅ Ertesi gün kaldığı yerden devam eder
- ✅ Hiçbir deney tekrar yapılmaz

### 3. Keyler yenilendikçe devam eder mi?
**EVET!** Otomatik devam sistemi:
- ✅ Her gün saat 09:00'da kontrol eder
- ✅ Quota yenilendiyse devam ettirir
- ✅ Tamamlanana kadar çalışır

### 4. Bitince haber verir mi?
**EVET!** Sistem:
- ✅ Terminal bildirimi gönderir (macOS)
- ✅ Log dosyasına "TAMAMLANDI" yazar
- ✅ `monitor_experiment.sh` ile kontrol edebilirsiniz

### 5. Ücretsiz tamamı yapılır mı?
**EVET!** Tamamen ücretsiz:
- ✅ 78 free key yeterli
- ✅ Toplam maliyet: **$0.00**
- ✅ Tüm deneyler tamamlanır

## 🎯 Hızlı Başlangıç

```bash
# 1. Deneyi başlat
./start_full_experiment.sh

# 2. Otomatik devam sistemini kur (opsiyonel ama önerilen)
./setup_auto_resume.sh

# 3. İlerlemeyi izle
./monitor_experiment.sh
```

## 📅 Günlük Akış

### Gün 1 (Bugün)
```
09:00 - Deney başlatılır
09:00-23:00 - ~100,000 çağrı yapılır
23:00 - Quota doldu, güvenle durur
```

### Gün 2 (Yarın)
```
00:00 - Quotalar yenilenir (otomatik)
09:00 - Cron job devam ettirir
09:00-10:00 - Kalan ~8,000 çağrı tamamlanır
10:00 - ✅ TÜM DENEYLER TAMAMLANDI!
```

## 🔄 Otomatik Devam Sistemi

### Kurulum
```bash
./setup_auto_resume.sh
```

Bu sistem:
- ✅ Her gün 09:00'da çalışır
- ✅ Tamamlanmamış deneyleri kontrol eder
- ✅ Otomatik devam ettirir
- ✅ Tamamlandığında bildirim gönderir

### Manuel Devam
Cron kurmak istemezseniz, manuel devam:
```bash
./auto_resume_experiment.sh
```

## 📊 İzleme

### Hızlı Durum
```bash
./monitor_experiment.sh
```

Çıktı:
```
📊 İlerleme:
   Tamamlanan: 45/108 (42%)
   Kalan: 63 deney

   [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 42%

✅ Durum: Çalışıyor (PID: 12345)
   CPU: 85% | Memory: 12%

⏱️  Tahmini Kalan Süre: ~1 gün
```

### Canlı Log Takibi
```bash
tail -f experiment_*.log
```

### Tamamlanan Deneyler
```bash
ls -lh results/*/
```

## 🛡️ Güvenlik Özellikleri

### 1. Quota Koruması
- Her key için 1,450 çağrı limiti (güvenli buffer)
- Otomatik key rotasyonu
- Quota dolunca güvenle durur

### 2. Devam Etme Mekanizması
- Tamamlanan deneyler atlanır
- Sadece eksik olanlar yapılır
- Hiçbir deney tekrar edilmez

### 3. Hata Yönetimi
- Redis çökmesi → Otomatik yeniden başlatır
- API hatası → Sonraki key'e geçer
- Sistem kapanması → Cron ile devam eder

## 📁 Dosya Yapısı

```
results/20250324_123456/
├── baseline_msmarco_s42.json          ✅ Tamamlandı
├── baseline_msmarco_s123.json         ✅ Tamamlandı
├── msmarco_minilm_t0.75_s42.json     ✅ Tamamlandı
├── ...                                 (108 dosya)
├── experiment_local.log                (Detaylı loglar)
├── experiment_remote.log
└── table4_summary.csv                  (Özet sonuçlar)
```

## 🔍 Sorun Giderme

### "Deney durdu, devam etmiyor"
```bash
# Manuel devam ettir
./auto_resume_experiment.sh

# Veya cron'u kontrol et
crontab -l
```

### "Quota doldu" hatası
```bash
# Normal! Yarın otomatik devam eder
# Veya manuel kontrol:
./monitor_experiment.sh
```

### "Redis bağlantı hatası"
```bash
# Redis'i başlat
redis-server --daemonize yes

# Deneyi devam ettir
./auto_resume_experiment.sh
```

## 💡 İpuçları

### Bilgisayarı Kapatabilir miyim?
**Hayır!** Ama:
- Sunucuda çalıştırırsanız kapatabilirsiniz
- Veya her gün manuel başlatın: `./auto_resume_experiment.sh`

### Daha Hızlı Tamamlamak İçin
```bash
# Daha fazla key ekleyin (100+ key ile aynı gün biter)
# .env dosyasına yeni keyler ekleyin
```

### İlerlemeyi Takip
```bash
# Her 5 dakikada bir kontrol et
watch -n 300 ./monitor_experiment.sh
```

## 🎉 Tamamlanma

Deney tamamlandığında:

```
✅ TÜM DENEYLER TAMAMLANDI!

📁 Sonuçlar: results/20250324_123456/
📊 Toplam: 108 deney
💰 Maliyet: $0.00

🎉 Deney başarıyla tamamlandı!
```

Sonuçlar:
- `results/*/` klasöründe JSON dosyaları
- `table4_summary.csv` özet tablo
- Analiz için: `python scripts/analyze_results.py results/*/`

## 📞 Destek

Sorun yaşarsanız:
```bash
# Logları kontrol edin
tail -100 experiment_*.log

# Durum kontrolü
./monitor_experiment.sh

# Manuel devam
./auto_resume_experiment.sh
```

---

**Hazırsınız!** `./start_full_experiment.sh` ile başlayın! 🚀
