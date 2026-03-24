# 🔄 OTOMATİK KEY YÖNETİMİ - AKIŞ DİYAGRAMI

## 📊 Sistem Çalışma Akışı

```
┌─────────────────────────────────────────────────────────────┐
│  DENEY BAŞLATILDI: ./start_full_experiment.sh               │
│  Toplam: 218 deney × 1000 çağrı = 217,000 LLM çağrısı      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM ÇAĞRISI GEREKLİ                                        │
│  Query: "What is the capital of France?"                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  KEY SEÇİMİ (GeminiService.attemptGenerate)                 │
│                                                              │
│  FOR each key in [Key 1, Key 2, ..., Key 162]:             │
│    ✓ Günlük quota doldu mu? (< 1,450 çağrı)               │
│    ✓ Son çağrıdan 4.8 saniye geçti mi?                     │
│    ✓ Evet ise → Bu key'i kullan                            │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  KEY BULUNDU     │    │  KEY BULUNAMADI  │
    │  ✅ Kullan       │    │  ⚠️ Bekle        │
    └──────────────────┘    └──────────────────┘
                │                       │
                │                       ▼
                │           ┌──────────────────────────┐
                │           │  TÜM KEYLER DOLDU MU?    │
                │           └──────────────────────────┘
                │                       │
                │           ┌───────────┴───────────┐
                │           │                       │
                │           ▼                       ▼
                │   ┌──────────────┐      ┌──────────────┐
                │   │  EVET        │      │  HAYIR       │
                │   │  Tüm keyler  │      │  Sadece rate │
                │   │  exhausted   │      │  limit       │
                │   └──────────────┘      └──────────────┘
                │           │                       │
                │           ▼                       ▼
                │   ┌──────────────────┐   ┌──────────────┐
                │   │  QUOTA RESET     │   │  500ms BEKLE │
                │   │  BEKLENİYOR      │   │  Tekrar dene │
                │   │                  │   └──────────────┘
                │   │  PST gece yarısı │           │
                │   │  hesapla         │           │
                │   │                  │           │
                │   │  Örnek:          │           │
                │   │  "12h 30m kaldı" │           │
                │   │                  │           │
                │   │  5 dakika bekle  │           │
                │   │  Tekrar kontrol  │◄──────────┘
                │   └──────────────────┘
                │           │
                │           ▼
                │   ┌──────────────────┐
                │   │  24 SAAT GEÇTİ?  │
                │   └──────────────────┘
                │           │
                │           ▼
                │   ┌──────────────────┐
                │   │  QUOTA RESET!    │
                │   │  Tüm keyler      │
                │   │  sıfırlandı      │
                │   │  0/1450 → Ready  │
                │   └──────────────────┘
                │           │
                └───────────┴───────────┐
                                        │
                                        ▼
                        ┌──────────────────────────┐
                        │  API ÇAĞRISI YAP         │
                        │  POST /generateContent   │
                        │  key=AIzaSy...           │
                        └──────────────────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
                        ▼                               ▼
            ┌──────────────────┐          ┌──────────────────┐
            │  BAŞARILI ✅     │          │  HATA ❌         │
            │  Response alındı │          │  Error detected  │
            └──────────────────┘          └──────────────────┘
                        │                               │
                        │                   ┌───────────┴───────────┐
                        │                   │                       │
                        │                   ▼                       ▼
                        │       ┌──────────────────┐    ┌──────────────────┐
                        │       │  429 / QUOTA     │    │  NETWORK ERROR   │
                        │       │  Key exhausted   │    │  Timeout, etc    │
                        │       └──────────────────┘    └──────────────────┘
                        │                   │                       │
                        │                   ▼                       ▼
                        │       ┌──────────────────┐    ┌──────────────────┐
                        │       │  KEY ROTASYON    │    │  EXPONENTIAL     │
                        │       │  Sonraki key'e   │    │  BACKOFF         │
                        │       │  geç             │    │  1s→2s→4s→8s...  │
                        │       │  Tekrar dene     │    │  Max 60s         │
                        │       └──────────────────┘    │  Tekrar dene     │
                        │                   │           └──────────────────┘
                        │                   │                       │
                        └───────────────────┴───────────────────────┘
                                            │
                                            ▼
                        ┌──────────────────────────────────┐
                        │  DENEY TAMAMLANDI MI?            │
                        │  217,000 / 217,000 çağrı         │
                        └──────────────────────────────────┘
                                            │
                        ┌───────────────────┴───────────────┐
                        │                                   │
                        ▼                                   ▼
            ┌──────────────────┐              ┌──────────────────┐
            │  HAYIR           │              │  EVET            │
            │  Devam et        │              │  Tamamlandı! 🎉  │
            │  Sonraki çağrı   │              └──────────────────┘
            └──────────────────┘                          │
                        │                                 ▼
                        │                     ┌──────────────────┐
                        └────────────────────►│  İSTATİSTİK      │
                                              │  ANALİZİ         │
                                              │  analyze_results │
                                              └──────────────────┘
```

---

## 🔑 KEY DURUMU TAKİBİ

```
┌─────────────────────────────────────────────────────────────┐
│  KEY USAGE TRACKING (keyDailyUsage Map)                     │
├─────────────────────────────────────────────────────────────┤
│  Key 1:   [████████████████████] 1450/1450 (FULL)          │
│  Key 2:   [████████████████████] 1450/1450 (FULL)          │
│  Key 3:   [████████████░░░░░░░░] 1200/1450 (ACTIVE)        │
│  Key 4:   [░░░░░░░░░░░░░░░░░░░░]    0/1450 (READY)         │
│  ...                                                         │
│  Key 162: [░░░░░░░░░░░░░░░░░░░░]    0/1450 (READY)         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌──────────────────────────────┐
            │  QUOTA RESET (24h sonra)     │
            │  Tüm barlar sıfırlanır       │
            │  [░░░░░░░░░░░░░░░░░░░░]      │
            └──────────────────────────────┘
```

---

## ⏱️ ZAMAN ÇİZELGESİ ÖRNEĞİ

### Senaryo 1: Tek Seferde Tamamlanma (En Olası)

```
10:00 ┌─────────────────────────────────────────────┐
      │  DENEY BAŞLADI                              │
      │  162 key paralel çalışıyor                  │
      │  Throughput: 2,430 çağrı/dakika             │
      └─────────────────────────────────────────────┘
      
10:30 ┌─────────────────────────────────────────────┐
      │  İLERLEME: 72,900 / 217,000 çağrı (33%)    │
      │  Key 1-50: FULL (1450/1450)                 │
      │  Key 51-162: ACTIVE                         │
      └─────────────────────────────────────────────┘
      
11:00 ┌─────────────────────────────────────────────┐
      │  İLERLEME: 145,800 / 217,000 çağrı (67%)   │
      │  Key 1-100: FULL (1450/1450)                │
      │  Key 101-162: ACTIVE                        │
      └─────────────────────────────────────────────┘
      
11:29 ┌─────────────────────────────────────────────┐
      │  ✅ DENEY TAMAMLANDI!                       │
      │  217,000 / 217,000 çağrı (100%)             │
      │  Toplam süre: 1 saat 29 dakika              │
      │  Kullanılan key: 150/162                    │
      │  Kalan buffer: 17,900 çağrı                 │
      └─────────────────────────────────────────────┘
```

### Senaryo 2: Çok Günlük Çalışma (Düşük Olasılık)

```
GÜN 1
─────
10:00 ┌─────────────────────────────────────────────┐
      │  DENEY BAŞLADI                              │
      │  162 key paralel çalışıyor                  │
      └─────────────────────────────────────────────┘
      
11:30 ┌─────────────────────────────────────────────┐
      │  ⚠️ TÜM KEYLER DOLDU                        │
      │  234,900 / 217,000 çağrı kullanıldı         │
      │  (Ağ hataları nedeniyle fazla retry)        │
      │                                              │
      │  💤 BEKLEME MODU                            │
      │  PST gece yarısına kadar: 12h 30m           │
      │  Checkpoint kaydedildi                      │
      └─────────────────────────────────────────────┘

GÜN 2
─────
00:00 ┌─────────────────────────────────────────────┐
  PST │  🔄 QUOTA RESET!                            │
      │  Tüm keyler sıfırlandı                      │
      │  Sistem otomatik devam ediyor               │
      └─────────────────────────────────────────────┘
      
00:15 ┌─────────────────────────────────────────────┐
      │  ✅ DENEY TAMAMLANDI!                       │
      │  Kalan 17,000 çağrı tamamlandı              │
      │  Toplam süre: 1 gün 15 dakika               │
      │  (Çoğu zaman bekleme)                       │
      └─────────────────────────────────────────────┘
```

---

## 📊 CHECKPOINT SİSTEMİ

```
┌─────────────────────────────────────────────────────────────┐
│  DENEY ÇALIŞIRKEN                                           │
├─────────────────────────────────────────────────────────────┤
│  results/20260324_100000/                                   │
│  ├─ baseline_msmarco_s42.json          ✅ TAMAMLANDI       │
│  ├─ baseline_msmarco_s123.json         ✅ TAMAMLANDI       │
│  ├─ baseline_msmarco_s456.json         ✅ TAMAMLANDI       │
│  ├─ msmarco_minilm_t0.75_s42.json      ✅ TAMAMLANDI       │
│  ├─ msmarco_minilm_t0.75_s123.json     🔄 ÇALIŞIYOR        │
│  └─ msmarco_minilm_t0.75_s456.json     ⏳ BEKLEMEDE        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌──────────────────────────────┐
            │  KESİNTİ OLURSA              │
            │  (Bilgisayar kapandı, vb)    │
            └──────────────────────────────┘
                            │
                            ▼
            ┌──────────────────────────────┐
            │  YENİDEN BAŞLATMA            │
            │  ./start_full_experiment.sh  │
            └──────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  CHECKPOINT KONTROLÜ                                        │
├─────────────────────────────────────────────────────────────┤
│  ✅ baseline_msmarco_s42.json EXISTS → SKIP                │
│  ✅ baseline_msmarco_s123.json EXISTS → SKIP               │
│  ✅ baseline_msmarco_s456.json EXISTS → SKIP               │
│  ✅ msmarco_minilm_t0.75_s42.json EXISTS → SKIP            │
│  ❌ msmarco_minilm_t0.75_s123.json MISSING → RUN           │
│  ❌ msmarco_minilm_t0.75_s456.json MISSING → RUN           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌──────────────────────────────┐
            │  KALDĞI YERDEN DEVAM         │
            │  Tamamlanan deneyler atlandı │
            │  Veri kaybı YOK              │
            └──────────────────────────────┘
```

---

## 🎯 ÖZET

Sistem **tam otomatik** çalışıyor:

1. ✅ Key rotasyonu → Otomatik
2. ✅ Quota reset bekleme → Otomatik
3. ✅ Hata yönetimi → Otomatik
4. ✅ Checkpoint → Otomatik
5. ✅ Resume → Otomatik

**Tek yapman gereken:** `./start_full_experiment.sh`

Sistem geri kalanını halleder! 🚀
