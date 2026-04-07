# Q1 Publication - Pre-Experiment Checklist

Bu checklist'i deneyi başlatmadan önce tamamlayın.

## ✅ TAMAMLANMASI GEREKENLER

### 1. Test Coverage (%80 Hedef)

```bash
# Test coverage'ı kontrol et
mvn clean test jacoco:report

# Raporu görüntüle
open target/site/jacoco/index.html
```

**Durum:** ✅ Yeni testler eklendi
- `KeyHealthMonitorTest.java`
- `BaselineComparatorTest.java`
- `CacheStrategyTest.java`
- `CheckpointManagerTest.java`

**Hedef:** %80 line coverage

---

### 2. Dataset Hazırlığı

#### Mevcut Durum
- ✅ 10K sample datasets mevcut
- ⚠️ Q1 için 100K gerekli

#### Aksiyon
```bash
cd scripts

# Basit paraphrase (hızlı test için)
python3 prepare_datasets.py --output-dir ../data --sample-size 10000

# VEYA

# Advanced paraphrase (Q1 kalitesi - UZUN SÜRER)
python3 prepare_datasets_advanced.py --output-dir ../data --sample-size 100000
```

**Önerilen:** İlk önce 10K ile test edin, sonra 100K'ya geçin.

---

### 3. Baseline Karşılaştırma

#### Gerekli Baseline'lar
- ✅ `NONE` - No cache (control)
- ✅ `EXACT_MATCH` - Hash-based cache
- ✅ `GPTCACHE_BASELINE` - SOTA comparison (YENİ EKLENDI)

#### Test
```bash
# GPTCACHE_BASELINE stratejisini test et
mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark \
  -Dcache.strategy=GPTCACHE_BASELINE \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42
```

---

### 4. Statistical Power

#### Gereksinimler
- **Minimum:** 26 seeds (d=0.8, power=0.80)
- **İdeal:** 64 seeds (d=0.5, power=0.80)

#### Power Analysis
```bash
cd scripts
python3 power_analysis.py --effect-size 0.8
python3 power_analysis.py --effect-size 0.5
```

**Karar:** 26 seeds ile başlayın (12-16 saat)

---

### 5. Sistem Gereksinimleri

#### Donanım
- ✅ 16GB RAM minimum
- ✅ 50GB disk space (100K datasets için)
- ✅ CPU: 4+ cores

#### Yazılım
- ✅ Java 21
- ✅ Maven 3.8+
- ✅ Redis Stack (RediSearch)
- ✅ Ollama (llama3.2:3b)
- ✅ Python 3.9+ (analysis scripts)

#### Kontrol
```bash
# Java
java -version  # 21+

# Maven
mvn -version  # 3.8+

# Redis
redis-cli PING  # PONG

# Ollama
curl http://localhost:11434/api/tags

# Python dependencies
cd scripts
pip install -r requirements.txt
```

---

### 6. Disk Space Kontrolü

```bash
# Mevcut disk kullanımı
df -h .

# Tahmini gereksinim
# - 100K datasets: ~5GB
# - 26 seeds × 3 datasets × 3 models × 3 thresholds × 3 strategies = ~2100 experiments
# - Her experiment ~1MB = ~2GB results
# - Checkpoints: ~1GB
# - TOPLAM: ~10GB
```

**Minimum:** 15GB boş alan

---

### 7. Ollama Model Hazırlığı

```bash
# Model'i indir (ilk kez)
ollama pull llama3.2:3b

# Ollama'yı başlat
ollama serve &

# Test et
ollama run llama3.2:3b "Hello"
```

---

### 8. Redis Hazırlığı

```bash
# Redis Stack'i başlat (Docker)
docker-compose up -d redis

# Veya local
redis-server --loadmodule /path/to/redisearch.so

# Test et
redis-cli FT._LIST
```

---

### 9. Reproducibility Package

#### Gerekli Dosyalar
- ✅ `README.md` - Execution instructions
- ✅ `pom.xml` - Locked dependency versions
- ✅ `docker-compose.yml` - Environment setup
- ✅ `scripts/collect_system_info.sh` - System documentation
- ⏳ `CITATION.cff` - Citation metadata
- ⏳ Zenodo DOI (deney sonrası)

---

### 10. Pre-Flight Test

Tam deneyi başlatmadan önce küçük bir test yapın:

```bash
# 3 seed ile hızlı test (30 dakika)
SEEDS=(42 123 456)
DATASETS=(msmarco)
STRATEGIES=(SEMANTIC EXACT_MATCH NONE)

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for STRATEGY in "${STRATEGIES[@]}"; do
      echo "Testing: $DATASET seed=$SEED strategy=$STRATEGY"
      mvn spring-boot:run \
        -q \
        -Dspring-boot.run.profiles=benchmark \
        -Dbenchmark.current-dataset="$DATASET" \
        -Dbenchmark.current-seed="$SEED" \
        -Dcache.strategy="$STRATEGY" \
        -Dresults.output-dir="results/preflight_test"
    done
  done
done

# Sonuçları analiz et
cd scripts
python3 analyze_results.py ../results/preflight_test
```

**Beklenen:** 9 experiment, hepsi başarılı

---

## 🚀 DENEY BAŞLATMA

Tüm checkler ✅ ise:

```bash
# Q1 Comprehensive Benchmark (12-16 saat)
./bin/run_q1_comprehensive_benchmark.sh
```

---

## ⚠️ UYARILAR

1. **Uzun Süre:** 12-16 saat sürecek, gece çalıştırın
2. **Disk Space:** Sürekli kontrol edin
3. **Ollama:** Watchdog script kullanın: `scripts/ollama_watchdog.sh`
4. **Checkpoint:** Otomatik kayıt var, kesinti durumunda devam edebilirsiniz
5. **Logs:** `results/q1_comprehensive_*/experiment.log` takip edin

---

## 📊 DENEY SONRASI

```bash
# 1. Statistical validation
cd scripts
python3 q1_validation_comprehensive.py --results-dir ../results/q1_comprehensive_*

# 2. Bias analysis
python3 bias_analysis.py --results-dir ../results/q1_comprehensive_*

# 3. Generate figures
python3 generate_publication_figures.py ../results/q1_comprehensive_*

# 4. Effect sizes
python3 effect_size_calculator.py --results-dir ../results/q1_comprehensive_*
```

---

## 📝 NOTLAR

- Bu checklist Q1 dergi standartlarına göre hazırlanmıştır
- Tüm adımları atlamamanız önerilir
- Sorular için: docs/Q1_PUBLICATION_ACTION_PLAN.md
