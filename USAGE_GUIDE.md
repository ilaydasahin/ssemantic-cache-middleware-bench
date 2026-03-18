# Kullanım Kılavuzu - 20 Key ile Arkaplanda Çalıştırma

## Hızlı Başlangıç

### 1. Keyler Hazır ✅

20 Gemini API key'iniz `.env` ve `application-local.yml` dosyalarına kaydedildi.

### 2. Hızlı Test (30 saniye)

```bash
bash quick_test.sh
```

10 sorgu ile sistem testini yapar. Çıktı:
```
✅ Loaded 20 API keys
📊 Total capacity: ~240 RPM, ~29000 RPD
🧪 Testing with 10 queries...
✅ Test completed!
```

### 3. Arkaplanda Tam Benchmark Başlat

```bash
bash run_background.sh
```

Çıktı:
```
╔════════════════════════════════════════════════════════════╗
║        SEMANTIC CACHE BENCHMARK - BACKGROUND MODE          ║
╠════════════════════════════════════════════════════════════╣
║  📊 Configuration:                                         ║
║     • 20 API keys loaded                                   ║
║     • Capacity: ~240 RPM, ~29,000 RPD                     ║
║  🎯 Running full benchmark suite...                        ║
║     • MS MARCO (10K queries)                              ║
║     • Natural Questions (10K queries)                     ║
║     • Quora Pairs (10K queries)                           ║
║  ⏱️  Estimated time: ~3-4 hours                            ║
║  💰 Cost: $0.00 (free tier)                               ║
╚════════════════════════════════════════════════════════════╝

✅ Benchmark started in background
   PID: 12345
   Log: logs/benchmark_20250118_143022.log
```

### 4. İlerlemeyi İzle

```bash
bash monitor.sh
```

Canlı çıktı:
```
📊 Live progress:
🔑 Multi-key mode: 20 keys detected. Total capacity: ~240RPM
📈 Progress: 100 total calls across 20 keys (avg 5/key)
📈 Progress: 500 total calls across 20 keys (avg 25/key)
📈 Progress: 1000 total calls across 20 keys (avg 50/key)
✅ Metrics computed: hitRate=85.2%, p50=12ms, p99=45ms
```

### 5. Sonuçları Kontrol Et

```bash
ls -lh results/
```

Her deney için JSON dosyası oluşturulur:
```
results/msmarco_seed42_threshold0.90.json
results/nq_seed42_threshold0.90.json
results/qqp_seed42_threshold0.90.json
```

## Komutlar

### Başlatma
```bash
# Hızlı test (10 sorgu, 30 saniye)
bash quick_test.sh

# Arkaplanda tam benchmark (10K sorgu, 3-4 saat)
bash run_background.sh

# Ön planda çalıştır (log ekranda)
bash run_full_benchmark_suite.sh
```

### İzleme
```bash
# Canlı monitoring
bash monitor.sh

# Log dosyasını direkt izle
tail -f logs/benchmark_*.log

# Sadece progress göster
tail -f logs/benchmark_*.log | grep Progress

# Hataları göster
tail -f logs/benchmark_*.log | grep ERROR
```

### Durdurma
```bash
# PID'yi bul
cat logs/benchmark.pid

# Durdur
kill $(cat logs/benchmark.pid)

# Zorla durdur (gerekirse)
kill -9 $(cat logs/benchmark.pid)
```

## Beklenen Davranış

### Normal Çalışma

```
INFO: ✅ Multi-key mode: 20 keys detected
INFO: Progress: 100 total calls across 20 keys (avg 5/key)
INFO: Progress: 200 total calls across 20 keys (avg 10/key)
...
INFO: Metrics computed: hitRate=85%, p50=12ms, p99=45ms
```

### Key Rotasyonu (Normal)

```
WARN: Key 5 hit quota/rate limit. Usage: 1450/1450. Rotating...
INFO: Progress: 7500 total calls across 20 keys (avg 375/key)
```

Bu normal! Sistem otomatik olarak sonraki key'e geçer.

### Tüm Keyler Tükendi (Günlük Limit)

```
ERROR: ALL 20 keys exhausted daily quota (1450 calls each). 
       Total: 29000 calls today.
```

**Çözüm**: Yarın devam et (kotalar gece yarısı PST'de sıfırlanır)

## Kapasite Planlaması

| Deney | Sorgu Sayısı | Süre | Key Kullanımı |
|-------|--------------|------|---------------|
| Hızlı test | 10 | 30 sn | <1 key |
| Küçük deney | 1,000 | 70 dk | 1 key |
| Orta deney | 5,000 | 90 dk | 4 key |
| Tam benchmark | 10,000 | 180 dk | 7 key |
| Maksimum | 29,000 | 500 dk | 20 key |

## Sorun Giderme

### "No API keys provided"
```bash
# .env dosyasını kontrol et
cat .env

# Yeniden yükle
source .env
echo $GEMINI_API_KEYS
```

### "Benchmark already running"
```bash
# Mevcut PID'yi kontrol et
cat logs/benchmark.pid

# Çalışıyor mu?
ps -p $(cat logs/benchmark.pid)

# Durdur
kill $(cat logs/benchmark.pid)
```

### "Rate limit errors"
Sistem otomatik halleder. Key rotasyonu yapılır.

### Yavaş çalışıyor
Normal! Her key 4.8 saniye bekliyor (rate limit). 20 key ile ~240 RPM hız normal.

## İpuçları

### 1. Gece Çalıştır
```bash
# Akşam başlat, sabah sonuçları al
nohup bash run_background.sh &
```

### 2. Birden Fazla Deney
```bash
# Sırayla çalıştır
for seed in 42 123 456; do
  bash run_full_benchmark_suite.sh
  sleep 60  # Keyler dinlensin
done
```

### 3. Disk Alanı Kontrol
```bash
# Sonuçlar büyüyebilir
du -sh results/
du -sh logs/
```

### 4. Bellek İzleme
```bash
# Java heap kullanımı
ps aux | grep java | grep semantic-cache
```

## Güvenlik

- ✅ `.env` ve `application-local.yml` git'e commit edilmez (.gitignore'da)
- ✅ Keyler asla log dosyalarında görünmez
- ✅ Sonuç dosyaları key içermez

## Özet

```bash
# 1. Test et
bash quick_test.sh

# 2. Arkaplanda başlat
bash run_background.sh

# 3. İzle
bash monitor.sh

# 4. Sonuçları al
ls -lh results/
```

**Toplam maliyet: $0.00** 🎉

## Destek

Sorun yaşarsan:
1. `logs/benchmark_*.log` dosyasını kontrol et
2. `bash monitor.sh` ile canlı izle
3. GitHub issue aç: https://github.com/ilaydasahin/ssemantic-cache-middleware-bench/issues
