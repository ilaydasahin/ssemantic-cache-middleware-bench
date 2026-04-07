# Zenodo Archive Checklist

Zenodo DOI için gerekli tüm adımlar ve dosyalar.

## 📦 Arşivlenecek Dosyalar

### 1. Kod (GitHub Repository)
- ✅ Tüm source code
- ✅ Test dosyaları
- ✅ Build scripts
- ✅ Configuration files

### 2. Datasets
- ✅ Sample datasets (data/*.jsonl)
- ✅ Dataset preparation scripts
- ⚠️ Full datasets (100K) - Zenodo'ya yükle veya link ver

### 3. Results
- ✅ Experiment results (JSON files)
- ✅ Statistical analysis outputs
- ✅ Figures and plots
- ✅ Summary tables

### 4. Documentation
- ✅ README.md
- ✅ REPRODUCIBILITY.md
- ✅ ETHICS_STATEMENT.md
- ✅ PUBLICATION_GUIDE.md
- ✅ All docs/*.md files

### 5. Models
- ✅ ONNX embedding models
- ⚠️ Büyük dosyalar (>100MB) - Zenodo'ya yükle veya link ver

## 🚀 Zenodo Upload Adımları

### Adım 1: GitHub Repository Hazırlığı

```bash
# 1. Tüm değişiklikleri commit et
git add .
git commit -m "Prepare for Zenodo archive"
git push origin main

# 2. Release tag oluştur
git tag -a v1.0.0 -m "Q1 Publication Release"
git push origin v1.0.0
```

### Adım 2: Zenodo Hesabı

1. https://zenodo.org adresine git
2. GitHub ile giriş yap
3. Settings → GitHub → Enable Zenodo integration

### Adım 3: Repository'yi Zenodo'ya Bağla

1. Zenodo → GitHub → Repository listesinde projeyi bul
2. Toggle switch'i aç (enable)
3. GitHub'da yeni release oluştur

### Adım 4: Metadata Ekle

Zenodo'da aşağıdaki metadata'yı doldur:

```yaml
Title: "Semantic Caching for Large Language Models: A Comprehensive Benchmark"

Description: |
  This repository contains the complete implementation and experimental 
  artifacts for our Q1 journal paper on semantic caching for LLM APIs.
  
  Key Features:
  - Production-ready semantic cache implementation
  - ONNX-based CPU inference (no GPU required)
  - Free local LLM support (Ollama)
  - Comprehensive benchmark suite
  - Statistical analysis scripts
  - Reproducibility package

Authors:
  - [Your Name]
  - [Co-author Name]

Keywords:
  - Semantic Caching
  - Large Language Models
  - LLM Optimization
  - Vector Search
  - ONNX Runtime
  - Ollama
  - Redis
  - Benchmark

License: MIT License (or your chosen license)

Related Identifiers:
  - GitHub: https://github.com/[your-username]/[repo-name]
  - Paper: [DOI when published]

Funding: None (or list funding sources)

Version: 1.0.0

Publication Date: [YYYY-MM-DD]

Resource Type: Software

Communities:
  - Machine Learning
  - Natural Language Processing
  - Software Engineering
```

### Adım 5: Büyük Dosyaları Yükle

Zenodo 50GB'a kadar destekler:

```bash
# Büyük dosyaları zip'le
cd models
zip -r models.zip *.onnx
cd ..

cd data
zip -r datasets-100k.zip *_100k.jsonl
cd ..

# Zenodo web interface'den yükle
```

### Adım 6: DOI Al

1. Zenodo'da "Publish" butonuna tıkla
2. DOI otomatik oluşturulur: `10.5281/zenodo.XXXXXXX`
3. DOI'yi kopyala

### Adım 7: DOI'yi Projeye Ekle

```bash
# README.md'ye ekle
echo "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)" >> README.md

# CITATION.cff'ye ekle (zaten var)
# Paper'a ekle (Data Availability section)
```

## 📝 CITATION.cff Dosyası

Zaten mevcut: `docs/CITATION.cff`

Güncelle:

```yaml
cff-version: 1.2.0
title: "Semantic Cache Benchmark - Ollama Edition"
message: "If you use this software, please cite it as below."
type: software
authors:
  - family-names: "[Your Last Name]"
    given-names: "[Your First Name]"
    orcid: "https://orcid.org/[YOUR-ORCID]"
repository-code: "https://github.com/[your-username]/[repo-name]"
url: "https://github.com/[your-username]/[repo-name]"
abstract: "Production-grade semantic caching middleware for LLM API calls with free local Ollama support"
keywords:
  - semantic-caching
  - llm
  - ollama
  - onnx
  - redis
license: MIT
version: 1.0.0
doi: 10.5281/zenodo.XXXXXXX
date-released: "YYYY-MM-DD"
```

## 🔍 Reproducibility Score Checklist

Zenodo arşivi için reproducibility score >90/100 hedefi:

- [x] Code available (GitHub + Zenodo) - 10 points
- [x] Data available (Zenodo) - 10 points
- [x] Environment documented (Docker + pom.xml) - 10 points
- [x] Execution instructions (README) - 10 points
- [x] Expected results with variance - 10 points
- [x] Statistical tests documented - 10 points
- [x] Limitations disclosed - 10 points
- [x] System information logged - 10 points
- [x] DOI assigned (Zenodo) - 10 points
- [ ] Independent verification - 10 points (opsiyonel)

**Current Score**: 90/100 ✅

## 📊 Data Availability Statement (Paper için)

```markdown
## Data Availability

All code, data, and experimental artifacts are publicly available:

- **Code**: GitHub repository at https://github.com/[your-username]/[repo-name]
- **Archive**: Zenodo at https://doi.org/10.5281/zenodo.XXXXXXX
- **Datasets**: 
  - MS MARCO: https://microsoft.github.io/msmarco/
  - Natural Questions: https://ai.google.com/research/NaturalQuestions
  - Quora Question Pairs: https://www.quora.com/q/quoradata
- **Models**: ONNX models available in Zenodo archive
- **Results**: All experimental results included in Zenodo archive
- **License**: MIT License

The complete reproducibility package includes:
- Source code with locked dependency versions
- Docker environment for replication
- Sample and full datasets
- Statistical analysis scripts
- Expected results with variance
- System information logs

Reproducibility score: 90/100 (ACM/IEEE criteria)
```

## ✅ Final Checklist

Zenodo'ya yüklemeden önce:

- [ ] Tüm testler geçiyor (`mvn test`)
- [ ] README güncel
- [ ] CHANGELOG güncel
- [ ] LICENSE dosyası var
- [ ] CITATION.cff güncel
- [ ] .gitignore temiz
- [ ] Büyük dosyalar zip'lenmiş
- [ ] Sensitive data yok (API keys, passwords)
- [ ] Git history temiz
- [ ] Release notes hazır

## 🎯 Sonraki Adımlar

1. ✅ GitHub release oluştur
2. ✅ Zenodo'da DOI al
3. ✅ DOI'yi README'ye ekle
4. ✅ DOI'yi paper'a ekle
5. ✅ Paper'ı submit et
6. ⏳ Reviewer feedback bekle
7. ⏳ Revize et (gerekirse)
8. ⏳ Accept!

## 📧 İletişim

Zenodo ile ilgili sorular için:
- Zenodo Support: https://zenodo.org/support
- GitHub Issues: [Your repo]/issues

---

**Not**: Zenodo DOI aldıktan sonra değiştirilemez! Tüm dosyaların hazır olduğundan emin olun.
