"""
Dataset Preparation Script for Semantic Cache Benchmark.

Downloads and prepares 3 datasets in JSONL format:
  1. MS MARCO — 10K sampled Q&A pairs + paraphrases
  2. Natural Questions — 10K sampled Q&A pairs + paraphrases
  3. Quora Question Pairs — 10K labeled pairs

Output format (JSONL):
  {"query": "...", "answer": "...", "paraphrase": "...", "dataset": "..."}

Usage: python3 prepare_datasets.py --output-dir ../data --sample-size 10000

Dependencies: datasets, tqdm, nltk
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    from datasets import load_dataset  # type: ignore
    from tqdm import tqdm  # type: ignore
except ImportError:
    # Fallback for environments where these aren't installed yet
    pass


def prepare_msmarco(output_dir: str, sample_size: int, seed: int) -> int:
    """Download and prepare MS MARCO dataset."""

    print(f"\n--- MS MARCO (sampling {sample_size} pairs) ---")

    # Load MS MARCO Q&A
    try:
        ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    except Exception:
        ds = load_dataset("ms_marco", "v2.1", split="train", trust_remote_code=True)

    # Sample and prepare
    rng = random.Random(seed)
    # Filter for items that actually have content
    indices = rng.sample(range(len(ds)), min(sample_size * 2, len(ds)))

    output_path = os.path.join(output_dir, "msmarco_sample.jsonl")
    records: List[Dict[str, Any]] = []

    for idx in indices:
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

    print(f"\n--- Natural Questions (sampling {sample_size} pairs) ---")

    try:
        ds = load_dataset("google-natural-questions/nq_open", split="train")
    except Exception:
        ds = load_dataset("nq_open", split="train", trust_remote_code=True)

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size, len(ds)))

    output_path = os.path.join(output_dir, "nq_sample.jsonl")
    records: List[Dict[str, Any]] = []

    for idx in indices:
        item = ds[idx]
        query = item.get("question", "")
        answers = item.get("answer", [])
        answer = answers[0] if answers else ""

        if query and answer:
            records.append(
                {"query": query, "answer": answer, "dataset": "natural-questions"}
            )

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"  ✅ Saved {len(records)} pairs to {output_path}")
    return len(records)


def prepare_quora_pairs(output_dir: str, sample_size: int, seed: int) -> int:
    """Download and prepare Quora Question Pairs dataset."""

    print(f"\n--- Quora Question Pairs (sampling {sample_size} pairs) ---")

    try:
        ds = load_dataset(
            "sentence-transformers/quora-duplicates", "pair", split="train"
        )
    except Exception as e:
        print(f"  ⚠️ Skipping Quora dataset: {e}")
        return 0

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size, len(ds)))

    output_path = os.path.join(output_dir, "qqp_sample.jsonl")
    records: List[Dict[str, Any]] = []

    for idx in indices:
        item = ds[idx]
        text_list = item.get("questions", {}).get("text", [])
        q1 = item.get("anchor") or (text_list[0] if len(text_list) > 0 else "")
        q2 = item.get("positive") or (text_list[1] if len(text_list) > 1 else "")
        is_duplicate = item.get("is_duplicate", False)

        if q1 and q2:
            records.append(
                {
                    "query": q1,
                    "answer": q2,
                    "is_duplicate": is_duplicate,
                    "dataset": "quora-pairs",
                }
            )

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"  ✅ Saved {len(records)} pairs to {output_path}")
    return len(records)


def generate_paraphrases_neural(input_path: str, output_path: str, seed: int):
    """
    Generate high-quality paraphrases using T5 and back-translation.
    Validates semantic similarity using SBERT.
    """
    print(f"\n--- Generating neural paraphrases for {input_path} ---")
    
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
        from sentence_transformers import SentenceTransformer, util
        import torch
    except ImportError:
        print("⚠️  Neural paraphrasing requires: transformers, sentence-transformers, torch")
        print("   Falling back to pattern-based paraphrasing...")
        return generate_paraphrases_fallback(input_path, output_path, seed)
    
    # Load models
    print("  Loading T5-base for paraphrasing...")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    print("  Loading MarianMT for back-translation...")
    en_de_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    en_de_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    de_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    de_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    
    print("  Loading SBERT for validation...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    torch.manual_seed(seed)
    
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(tqdm(fin, desc="Paraphrasing")):
            record = json.loads(line)
            query = record["query"]
            
            # Method 1: T5 paraphrasing
            input_text = f"paraphrase: {query}"
            inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
            outputs = t5_model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=5,
                num_return_sequences=3,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            t5_paraphrases = [t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
            
            # Method 2: Back-translation (en->de->en)
            de_tokens = en_de_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            de_output = en_de_model.generate(**de_tokens)
            german = en_de_tokenizer.decode(de_output[0], skip_special_tokens=True)
            
            en_tokens = de_en_tokenizer(german, return_tensors="pt", padding=True, truncation=True)
            en_output = de_en_model.generate(**en_tokens)
            back_translated = de_en_tokenizer.decode(en_output[0], skip_special_tokens=True)
            
            # Combine candidates
            candidates = t5_paraphrases + [back_translated]
            
            # SBERT validation: select paraphrase with similarity 0.75-0.95
            query_emb = sbert.encode(query, convert_to_tensor=True)
            best_paraphrase = query
            best_score = 0.0
            
            for candidate in candidates:
                if candidate.strip() and candidate != query:
                    cand_emb = sbert.encode(candidate, convert_to_tensor=True)
                    sim = util.cos_sim(query_emb, cand_emb).item()
                    
                    # Target: high similarity but not identical
                    if 0.75 <= sim <= 0.95 and abs(sim - 0.85) < abs(best_score - 0.85):
                        best_paraphrase = candidate
                        best_score = sim
            
            record["paraphrase"] = best_paraphrase
            record["paraphrase_method"] = "neural"
            record["paraphrase_similarity"] = float(best_score)
            record["paraphrase_seed"] = seed + i
            fout.write(json.dumps(record) + "\n")
    
    print(f"  ✅ Neural paraphrases saved to {output_path}")


def generate_paraphrases_fallback(input_path: str, output_path: str, seed: int):
    """Fallback pattern-based paraphrasing if neural models unavailable."""
    print(f"\n--- Pattern-based paraphrasing (fallback) for {input_path} ---")
    
    patterns = [
        ("How do I", "Give me steps to"),
        ("How can I", "What is the best method to"),
        ("What is", "Can you provide details on"),
        ("Why does", "Explain the reason that"),
        ("Where can I", "Show me the place to"),
    ]
    
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            record = json.loads(line)
            query = record["query"]
            
            rng = random.Random(seed + i)
            paraphrased = str(query)
            pattern_applied = False
            
            for old, new in patterns:
                insensitive_old = re.compile(re.escape(old), re.IGNORECASE)
                if insensitive_old.search(paraphrased):
                    paraphrased = insensitive_old.sub(new, paraphrased, count=1)
                    pattern_applied = True
                    break
            
            if not pattern_applied or len(paraphrased) <= len(query) + 2:
                prefixes = [
                    "Inquire about ",
                    "Briefly explain ",
                    "Tell me more about ",
                    "Information regarding ",
                ]
                paraphrased = rng.choice(prefixes) + query.lower()
            
            if len(paraphrased) < len(query) * 1.2:
                paraphrased = "Provide information on: " + query.lower()
            
            record["paraphrase"] = paraphrased
            record["paraphrase_method"] = "pattern"
            record["paraphrase_seed"] = seed + i
            fout.write(json.dumps(record) + "\n")
    
    print(f"  ✅ Pattern-based paraphrases saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument(
        "--sample-size", type=int, default=100000, help="Samples per dataset (default: 100K for Q1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Dataset Preparation ===")

    # Prepare each dataset
    prepare_msmarco(args.output_dir, args.sample_size, args.seed)
    prepare_natural_questions(args.output_dir, args.sample_size, args.seed)
    prepare_quora_pairs(args.output_dir, args.sample_size, args.seed)

    # Generate paraphrases (neural by default, fallback to pattern-based)
    for name in ["msmarco_sample", "nq_sample", "qqp_sample"]:
        input_path = os.path.join(args.output_dir, f"{name}.jsonl")
        output_path = os.path.join(args.output_dir, f"{name}_with_paraphrases.jsonl")
        if os.path.exists(input_path):
            generate_paraphrases_neural(input_path, output_path, args.seed)

    print(f"\n=== Done: Dataset generation complete ===")


if __name__ == "__main__":
    main()
