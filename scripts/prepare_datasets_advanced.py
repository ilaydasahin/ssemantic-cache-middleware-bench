#!/usr/bin/env python3
"""
ADVANCED Dataset Preparation with High-Quality Paraphrases

Q1 Publication Requirements:
1. Real semantic paraphrases (not simple pattern matching)
2. Back-translation for diversity
3. Quality validation (SBERT similarity check)
4. Larger dataset support (100K queries)

Methods:
- Back-translation (English → German → English)
- T5-based paraphrasing
- SBERT validation (0.7 < similarity < 0.95)

Usage: python3 prepare_datasets_advanced.py --output-dir ../data --sample-size 100000
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    MarianMTModel,
    MarianTokenizer
)
from sentence_transformers import SentenceTransformer, util

print("Loading models for high-quality paraphrase generation...")
print("This may take a few minutes on first run...")

# T5 for paraphrasing
t5_model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_paraphraser")
t5_tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")

# MarianMT for back-translation (English → German → English)
en2de_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
en2de_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
de2en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
de2en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# SBERT for quality validation
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ All models loaded successfully!")


def generate_t5_paraphrase(text: str, num_return_sequences: int = 3) -> List[str]:
    """Generate paraphrases using T5 model."""
    input_text = f"paraphrase: {text} </s>"
    encoding = t5_tokenizer.encode_plus(
        input_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = t5_model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=256,
            num_beams=5,
            num_return_sequences=num_return_sequences,
            temperature=1.5,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            early_stopping=True
        )
    
    paraphrases = []
    for output in outputs:
        paraphrase = t5_tokenizer.decode(output, skip_special_tokens=True)
        if paraphrase and paraphrase != text:
            paraphrases.append(paraphrase)
    
    return paraphrases


def generate_backtranslation_paraphrase(text: str) -> str:
    """Generate paraphrase via back-translation (EN → DE → EN)."""
    # English to German
    en_inputs = en2de_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        de_outputs = en2de_model.generate(**en_inputs)
    german_text = en2de_tokenizer.decode(de_outputs[0], skip_special_tokens=True)
    
    # German back to English
    de_inputs = de2en_tokenizer(german_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        en_outputs = de2en_model.generate(**de_inputs)
    paraphrase = de2en_tokenizer.decode(en_outputs[0], skip_special_tokens=True)
    
    return paraphrase


def validate_paraphrase_quality(original: str, paraphrase: str) -> Dict[str, float]:
    """
    Validate paraphrase quality using SBERT similarity.
    
    Q1 Requirements:
    - Similarity should be 0.70 < sim < 0.95
    - Too low: Not semantically equivalent
    - Too high: Too similar (not a real paraphrase)
    """
    with torch.no_grad():
        emb1 = sbert_model.encode(original, convert_to_tensor=True)
        emb2 = sbert_model.encode(paraphrase, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
    
    # Lexical overlap (Jaccard similarity)
    words1 = set(original.lower().split())
    words2 = set(paraphrase.lower().split())
    jaccard = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
    
    return {
        "semantic_similarity": similarity,
        "lexical_overlap": jaccard,
        "is_valid": 0.70 < similarity < 0.95 and jaccard < 0.8
    }


def generate_high_quality_paraphrase(text: str, seed: int) -> Dict[str, Any]:
    """
    Generate and validate high-quality paraphrase.
    
    Strategy:
    1. Try T5 paraphrasing (3 candidates)
    2. Try back-translation
    3. Select best candidate based on quality metrics
    4. Fallback to simple reformulation if all fail
    """
    random.seed(seed)
    
    candidates = []
    
    # Method 1: T5 paraphrasing
    try:
        t5_paraphrases = generate_t5_paraphrase(text, num_return_sequences=3)
        for para in t5_paraphrases:
            quality = validate_paraphrase_quality(text, para)
            if quality["is_valid"]:
                candidates.append({
                    "text": para,
                    "method": "t5",
                    "quality": quality
                })
    except Exception as e:
        print(f"⚠️  T5 paraphrasing failed: {e}")
    
    # Method 2: Back-translation
    try:
        bt_paraphrase = generate_backtranslation_paraphrase(text)
        quality = validate_paraphrase_quality(text, bt_paraphrase)
        if quality["is_valid"]:
            candidates.append({
                "text": bt_paraphrase,
                "method": "backtranslation",
                "quality": quality
            })
    except Exception as e:
        print(f"⚠️  Back-translation failed: {e}")
    
    # Select best candidate (highest semantic similarity within valid range)
    if candidates:
        best = max(candidates, key=lambda x: x["quality"]["semantic_similarity"])
        return {
            "paraphrase": best["text"],
            "method": best["method"],
            "semantic_similarity": best["quality"]["semantic_similarity"],
            "lexical_overlap": best["quality"]["lexical_overlap"]
        }
    
    # Fallback: Simple reformulation
    print(f"⚠️  All methods failed for: {text[:50]}... Using fallback")
    fallback_prefixes = [
        "Can you explain ",
        "Please provide information about ",
        "I would like to know about ",
        "Tell me regarding "
    ]
    fallback = random.choice(fallback_prefixes) + text.lower()
    
    return {
        "paraphrase": fallback,
        "method": "fallback",
        "semantic_similarity": 0.0,
        "lexical_overlap": 0.0
    }


def prepare_msmarco(output_dir: str, sample_size: int, seed: int) -> int:
    """Download and prepare MS MARCO dataset."""
    from datasets import load_dataset
    
    print(f"\n--- MS MARCO (sampling {sample_size} pairs) ---")
    
    try:
        ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    except Exception:
        ds = load_dataset("ms_marco", "v2.1", split="train", trust_remote_code=True)
    
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size * 2, len(ds)))
    
    output_path = os.path.join(output_dir, "msmarco_sample.jsonl")
    records: List[Dict[str, Any]] = []
    
    for idx in tqdm(indices, desc="Processing MS MARCO"):
        if len(records) >= sample_size:
            break
        item = ds[idx]
        query = item.get("query", "")
        answers = item.get("answers", [])
        answer = answers[0] if answers else ""
        
        if query and answer and answer != "No Answer Present.":
            records.append({"query": query, "answer": answer, "dataset": "msmarco"})
    
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    print(f"  ✅ Saved {len(records)} pairs to {output_path}")
    return len(records)


def prepare_natural_questions(output_dir: str, sample_size: int, seed: int) -> int:
    """Download and prepare Natural Questions dataset."""
    from datasets import load_dataset
    
    print(f"\n--- Natural Questions (sampling {sample_size} pairs) ---")
    
    try:
        ds = load_dataset("google-research-datasets/natural_questions", split="train")
    except Exception:
        try:
            ds = load_dataset("nq_open", split="train", trust_remote_code=True)
        except:
            print("⚠️  Could not load Natural Questions, skipping...")
            return 0
    
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size, len(ds)))
    
    output_path = os.path.join(output_dir, "nq_sample.jsonl")
    records: List[Dict[str, Any]] = []
    
    for idx in tqdm(indices, desc="Processing NQ"):
        item = ds[idx]
        query = item.get("question", "")
        answers = item.get("answer", [])
        answer = answers[0] if answers else ""
        
        if query and answer:
            records.append({"query": query, "answer": answer, "dataset": "natural-questions"})
    
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    print(f"  ✅ Saved {len(records)} pairs to {output_path}")
    return len(records)


def generate_paraphrases_advanced(input_path: str, output_path: str, seed: int):
    """Generate high-quality paraphrases with validation."""
    print(f"\n--- Generating ADVANCED paraphrases for {input_path} ---")
    print("This will take longer but produces Q1-quality paraphrases")
    
    with open(input_path, "r") as fin:
        records = [json.loads(line) for line in fin]
    
    paraphrased_records = []
    quality_stats = {
        "t5": 0,
        "backtranslation": 0,
        "fallback": 0
    }
    
    for i, record in enumerate(tqdm(records, desc="Paraphrasing")):
        query = record["query"]
        
        # Generate high-quality paraphrase
        result = generate_high_quality_paraphrase(query, seed + i)
        
        # Update record
        record["paraphrase"] = result["paraphrase"]
        record["paraphrase_method"] = result["method"]
        record["paraphrase_semantic_similarity"] = result["semantic_similarity"]
        record["paraphrase_lexical_overlap"] = result["lexical_overlap"]
        record["paraphrase_seed"] = seed + i
        
        paraphrased_records.append(record)
        quality_stats[result["method"]] += 1
    
    # Save results
    with open(output_path, "w") as fout:
        for record in paraphrased_records:
            fout.write(json.dumps(record) + "\n")
    
    print(f"  ✅ Saved {len(paraphrased_records)} paraphrased entries")
    print(f"\n  Quality breakdown:")
    print(f"    T5 paraphrasing: {quality_stats['t5']} ({quality_stats['t5']/len(records)*100:.1f}%)")
    print(f"    Back-translation: {quality_stats['backtranslation']} ({quality_stats['backtranslation']/len(records)*100:.1f}%)")
    print(f"    Fallback: {quality_stats['fallback']} ({quality_stats['fallback']/len(records)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Advanced dataset preparation for Q1 publication")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--sample-size", type=int, default=100000, 
                       help="Samples per dataset (default: 100K for Q1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip dataset download (use existing files)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("ADVANCED DATASET PREPARATION FOR Q1 PUBLICATION")
    print("=" * 70)
    print()
    print(f"Target sample size: {args.sample_size:,} per dataset")
    print(f"Paraphrase methods: T5, Back-translation, Fallback")
    print(f"Quality validation: SBERT similarity (0.70 < sim < 0.95)")
    print()
    
    if not args.skip_download:
        # Prepare datasets
        prepare_msmarco(args.output_dir, args.sample_size, args.seed)
        prepare_natural_questions(args.output_dir, args.sample_size, args.seed)
    
    # Generate high-quality paraphrases
    for name in ["msmarco_sample", "nq_sample"]:
        input_path = os.path.join(args.output_dir, f"{name}.jsonl")
        output_path = os.path.join(args.output_dir, f"{name}_with_paraphrases.jsonl")
        
        if os.path.exists(input_path):
            generate_paraphrases_advanced(input_path, output_path, args.seed)
        else:
            print(f"⚠️  Skipping {name} - file not found")
    
    print()
    print("=" * 70)
    print("✅ ADVANCED DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Verify paraphrase quality in output files")
    print("2. Run benchmark with new datasets")
    print("3. Report paraphrase methods in paper")


if __name__ == "__main__":
    main()
