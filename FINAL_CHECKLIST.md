# Son Kontrol Listesi - Arkaplanda Çalıştırma

## ✅ Sistem Hazır

### 1. API Key Yönetimi
- ✅ 23 Gemini API key yüklü (EXTENDED CAPACITY)
- ✅ `.env` dosyasında saklanıyor
- ✅ `application-local.yml` backup var
- ✅ `.gitignore` ile korunuyor

### 2. Token Güvenliği
```java
CALL_SPACING = 4800ms        // Her key 4.8s bekler (12.5 RPM)
DAILY_QUOTA_PER_KEY = 1450   // Key başına günlük limit
```

**Toplam Kapasite:**
- Dakikalık: 276 istek (23 × 12 RPM)
- Günlük: 33,350 istek (23 × 1,450 RPD)

### 3. Hata Yönetimi

#### Rate Limit (429)
```
WARN: Key 5 hit quota/rate limit. Rotating...
```
✅ Otomatik sonraki key'e geçer

#### Günlük Kota Aşımı
```
ERROR: ALL 23 keys exhausted daily quota. Total: 33350 calls today.
```
✅ Graceful shutdown, sonuçlar kaydedilir

#### Network Hatası
✅ Retry mekanizması (max 40 deneme)
✅ Hata loglanır, deney devam eder

### 4. Derleme
```bash
mvn compile -q
# Exit code: 0 ✅
```

### 5. Scriptler
- ✅ `quick_test.sh` - 10 sorgu test
- ✅ `run_background.sh` - Arkaplanda çalıştır
- ✅ `monitor.sh` - Canlı izleme

## 🎯 Önerilen Başlangıç

### Adım 1: Hızlı Test (5 dakika)
```bash
bash quick_test.sh
```

**Beklenen çıktı:**
```
✅ Loaded 23 API keys
📊 Total capacity: ~276 RPM, ~33350 RPD
🧪 Testing with 10 queries...
INFO: Multi-key mode: 23 keys detected
INFO: Progress: 10 total calls across 23 keys
✅ Test completed!
```

**Token kullanımı:** ~3,000 token (10 sorgu × ~300 token)

### Adım 2: Küçük Deney (1 saat)
```bash
mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42 \
  -Dbenchmark.sample-size=1000 \
  -Dbenchmark.output-file=results/test_1k.json
```

**Token kullanımı:** ~300,000 token (1,000 sorgu)
**Key kullanımı:** 1 key

### Adım 3: Tam Benchmark (3-4 saat)
```bash
bash run_background.sh
```

**Token kullanımı:** ~3,000,000 token (10,000 sorgu)
**Key kullanımı:** 7 key (16 key kaldı)

## 🔍 İzleme

### Canlı Monitoring
```bash
bash monitor.sh
```

### Manuel Log İzleme
```bash
tail -f logs/benchmark_*.log | grep -E "Progress|ERROR|exhausted"
```

### Beklenen Çıktılar

**Normal:**
```
INFO: Progress: 100 total calls across 23 keys (avg 4/key)
INFO: Progress: 500 total calls across 23 keys (avg 22/key)
INFO: Progress: 1000 total calls across 23 keys (avg 43/key)
```

**Key Rotasyonu (Normal):**
```
WARN: Key 3 hit quota limit. Usage: 1450/1450. Rotating...
```

**Başarılı Tamamlanma:**
```
INFO: Metrics computed: hitRate=85.2%, p50=12ms, p99=45ms
INFO: Result written to: results/msmarco_seed42.json
```

## ⚠️ Olası Sorunlar

### Sorun 1: "No API keys provided"
**Çözüm:**
```bash
source .env
echo $GEMINI_API_KEYS | wc -w  # 23 olmalı
```

### Sorun 2: Yavaş çalışıyor
**Normal!** Her key 4.8s bekliyor. 23 key ile:
- 20 sorgu/dakika → 1 key
- 276 sorgu/dakika → 23 key (paralel)

### Sorun 3: "ALL keys exhausted"
**Çözüm:** Yarın devam et (kotalar gece yarısı PST'de reset)

## �� Token Hesaplaması

| Sorgu Sayısı | Ortalama Token/Sorgu | Toplam Token | Key Kullanımı |
|--------------|---------------------|--------------|---------------|
| 10 | 300 | 3,000 | <1 |
| 100 | 300 | 30,000 | <1 |
| 1,000 | 300 | 300,000 | 1 |
| 5,000 | 300 | 1,500,000 | 4 |
| 10,000 | 300 | 3,000,000 | 7 |
| 33,350 | 300 | 10,005,000 | 23 |

**Not:** Gemini free tier token limiti yok, sadece RPM/RPD limiti var.

## ✅ Güvenlik Onayı

- ✅ Rate limiting aktif (4.8s/key)
- ✅ Günlük kota takibi aktif (1,450/key)
- ✅ Otomatik key rotasyonu aktif
- ✅ Graceful degradation aktif
- ✅ Error logging aktif
- ✅ Retry mekanizması aktif (max 40)

## 🚀 Başlatma Komutu

```bash
# Test
bash quick_test.sh

# Arkaplanda tam benchmark
bash run_background.sh

# İzle
bash monitor.sh
```

## 💰 Maliyet

**Toplam: $0.00** (Tamamen ücretsiz!)

---

**Sonuç:** Sistem arkaplanda token sorunu olmadan çalışmaya hazır! ✅
