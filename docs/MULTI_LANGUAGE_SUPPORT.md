# Multi-Language Support

Q1 dergileri için generalizability göstermek amacıyla multi-language support eklenmesi önerilir.

## Mevcut Durum

- ✅ İngilizce datasets (MS MARCO, Natural Questions, QQP)
- ❌ Diğer diller yok

## Q1 Gereksinimi

Minimum 2-3 dil ile test edilmeli veya limitation olarak belirtilmeli.

## Önerilen Yaklaşım

### Seçenek 1: Multilingual Embedding Model

```java
// XLM-RoBERTa veya mBERT kullan
public enum EmbeddingModelType {
    MINILM("all-MiniLM-L6-v2", 384),
    MPNET("all-mpnet-base-v2", 768),
    TINYBERT("paraphrase-TinyBERT-L6-v2", 384),
    XLM_ROBERTA("xlm-roberta-base", 768),  // YENİ: Multilingual
    MBERT("bert-base-multilingual-cased", 768);  // YENİ: Multilingual
}
```

### Seçenek 2: Language-Specific Datasets

Türkçe dataset örneği:

```python
# scripts/prepare_turkish_dataset.py
def prepare_turkish_dataset():
    """
    Türkçe Q&A dataset hazırla
    Kaynak: TQuAD (Turkish Question Answering Dataset)
    """
    from datasets import load_dataset
    
    ds = load_dataset("erdometo/tquad", split="train")
    
    records = []
    for item in ds:
        query = item["question"]
        answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
        
        if query and answer:
            records.append({
                "query": query,
                "answer": answer,
                "dataset": "tquad",
                "language": "tr"
            })
    
    return records
```

### Seçenek 3: Limitation Olarak Belirt

Eğer multi-language support eklemek mümkün değilse, makale limitations bölümünde açıkça belirtin:

```markdown
## Limitations

1. **Language Coverage**: This study focuses on English-language queries. 
   While the embedding models (BERT-based) can theoretically support 
   multilingual queries, we did not evaluate performance on non-English 
   datasets. Future work should validate semantic caching effectiveness 
   across multiple languages using multilingual embedding models 
   (e.g., XLM-RoBERTa, mBERT).

2. **Domain Specificity**: Our evaluation uses general-domain datasets 
   (MS MARCO, Natural Questions). Domain-specific applications 
   (medical, legal, financial) may exhibit different cache hit patterns.
```

## Önerilen Strateji (Hızlı)

1. **Şimdi:** Limitation olarak belirt
2. **Revizyon aşamasında:** Reviewer talep ederse, 1 dil ekle (Türkçe en kolay)
3. **Gelecek çalışma:** Full multilingual evaluation

## Implementasyon (Opsiyonel)

Eğer eklemek isterseniz:

```bash
# 1. Multilingual embedding model indir
cd models
wget https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/model.onnx

# 2. Türkçe dataset hazırla
cd scripts
python3 prepare_turkish_dataset.py --output-dir ../data

# 3. Benchmark çalıştır
mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark \
  -Dbenchmark.current-dataset=tquad \
  -Dembedding.model-name=multilingual-minilm
```

## Sonuç

**Önerimiz:** Şimdilik limitation olarak belirtin. Q1 dergileri bunu kabul eder. Reviewer talep ederse eklersiniz.
