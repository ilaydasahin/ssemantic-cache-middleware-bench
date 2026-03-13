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


def generate_paraphrases(input_path: str, output_path: str, seed: int):
    """Generate simple paraphrases using question reformulation patterns."""
    print(f"\n--- Generating paraphrases for {input_path} ---")

    patterns = [
        ("How do I", "Give me steps to"),
        ("How can I", "What is the best method to"),
        ("What is", "Can you provide details on"),
        ("Why does", "Explain the reason that"),
        ("Where can I", "Show me the place to"),
    ]

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        # Use enumerate to avoid explicit counter variable subject to type inference issues
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

            record["paraphrase"] = paraphrased
            fout.write(json.dumps(record) + "\n")

    print(f"  ✅ Added paraphrases to entries in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument(
        "--sample-size", type=int, default=10000, help="Samples per dataset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Dataset Preparation ===")

    # Prepare each dataset
    prepare_msmarco(args.output_dir, args.sample_size, args.seed)
    prepare_natural_questions(args.output_dir, args.sample_size, args.seed)
    prepare_quora_pairs(args.output_dir, args.sample_size, args.seed)

    # Generate paraphrases
    for name in ["msmarco_sample", "nq_sample", "qqp_sample"]:
        input_path = os.path.join(args.output_dir, f"{name}.jsonl")
        output_path = os.path.join(args.output_dir, f"{name}_with_paraphrases.jsonl")
        if os.path.exists(input_path):
            generate_paraphrases(input_path, output_path, args.seed)

    print(f"\n=== Done: Dataset generation complete ===")


if __name__ == "__main__":
    main()
