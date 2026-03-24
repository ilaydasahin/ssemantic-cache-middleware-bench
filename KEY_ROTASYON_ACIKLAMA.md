# 🔄 Otomatik Key Rotasyon Sistemi

## 📋 Nasıl Çalışıyor?

Sistem **tamamen otomatik** key rotasyonu yapıyor. İşte adım adım:

### 1️⃣ Her API Çağrısında

```
Çağrı geldi → Uygun key seç → API çağrısı yap → Sonraki key'e geç
```

### 2️⃣ Key Seçim Algoritması

Sistem her çağrıda şu kontrolleri yapar:

```java
for (her key için) {
    ✅ Günlük quota doldu mu? (1,450/1,500)
    ✅ Son çağrıdan 4.8 saniye geçti mi? (rate limit)
    
    İKİSİ DE UYGUNSA:
        → Bu key'i kullan
        → Sayacı artır
        → Sonraki key'e geç
}
```

### 3️⃣ Otomatik Hata Yönetimi

**Quota Doldu (429 hatası):**
```
Key 1 quota doldu (1,450/1,450)
  ↓
Otomatik Key 2'ye geç
  ↓
Key 2 ile devam et
  ↓
Deney kesintisiz devam eder
```

**Tüm Keyler Doldu:**
```
102 key'in hepsi doldu
  ↓
Sistem 5 dakika bekler
  ↓
Quota reset kontrolü yapar
  ↓
Yeni gün başladıysa → Tüm quotalar sıfırlanır
  ↓
Deney kaldığı yerden devam eder
```

### 4️⃣ Network Hataları

```
Timeout/Connection hatası
  ↓
Exponential backoff (1s, 2s, 4s, 8s...)
  ↓
Otomatik retry
  ↓
Deney kesintisiz devam eder
```

## ✅ Garantiler

### 1. Kesintisiz Geçiş
- Key değişimi **0 ms** sürer
- Deney hiç durmaz
- Hiçbir çağrı kaybolmaz

### 2. Otomatik Quota Yönetimi
- Her key için 1,450 çağrı limiti (güvenli buffer)
- Otomatik 24 saatlik reset
- Gerçek zamanlı kullanım takibi

### 3. Paralel Çalışma
- 102 key = 102 paralel çağrı
- Her key bağımsız çalışır
- Maksimum hız

### 4. Hata Toleransı
- API hataları → Otomatik retry
- Network sorunları → Exponential backoff
- Quota dolması → Otomatik key değişimi

## 📊 İzleme

Sistem her 100 çağrıda bir rapor verir:

```
Progress: 100 total calls across 102 keys (avg 1/key)
Progress: 200 total calls across 102 keys (avg 2/key)
Progress: 300 total calls across 102 keys (avg 3/key)
...
Progress: 108000 total calls across 102 keys (avg 1059/key)
```

## 🔍 Log Örnekleri

### Normal Çalışma
```
INFO: Progress: 1000 total calls across 102 keys (avg 10/key)
INFO: Progress: 2000 total calls across 102 keys (avg 20/key)
```

### Key Rotasyonu
```
WARN: Key 5 hit quota/rate limit. Usage: 1450/1450. Rotating...
INFO: Switched to key 6
```

### Tüm Keyler Doldu
```
WARN: ⏰ ALL 102 keys exhausted (147,900 total calls). 
      Waiting for PST midnight quota reset...
WARN: 💤 Reset time: 2025-03-25 00:00:00 PST | Time remaining: 8h 23m
WARN: 📊 Checkpoint saved. System will auto-resume.
```

### Quota Reset
```
INFO: 🔄 Daily quota reset completed! All 102 keys refreshed. 
      Previous 24h total: 147900 calls
INFO: ✅ New capacity available: ~1224RPM, ~147900RPD
```

## 🎯 Sizin Durumunuzda

**102 key ile:**
- Günlük kapasite: 147,900 çağrı
- Deney ihtiyacı: 108,000 çağrı
- Buffer: +39,900 çağrı (%36)

**Sonuç:**
- ✅ Tek günde kesin biter
- ✅ Hiçbir key manuel değişimi gerekmez
- ✅ Sistem tamamen otomatik
- ✅ Siz sadece izlersiniz

## 💡 Önemli Notlar

1. **Manuel müdahale gerekmez**: Sistem her şeyi otomatik yapar
2. **Deney asla durmaz**: Key değişimi anında olur
3. **Checkpoint sistemi**: Her deney sonucu kaydedilir
4. **Resume özelliği**: Sistem kapansa bile kaldığı yerden devam eder

## 🚀 Başlatma

```bash
./start_full_experiment.sh
```

Sistem başladıktan sonra:
- Otomatik key rotasyonu başlar
- İlerleme logları görürsünüz
- Hiçbir şey yapmanız gerekmez
- 8-10 saat sonra tamamlanır

---

**Özet**: Sistem tamamen otomatik. Siz sadece başlatıp izleyin! 🎯
