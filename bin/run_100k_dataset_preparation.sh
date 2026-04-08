#!/bin/bash
# 100K Dataset Preparation with Advanced Paraphrasing
# Q1 Publication Quality

set -e

echo "=========================================="
echo "100K Dataset Preparation (Q1 Quality)"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This will take 2-4 hours"
echo "⚠️  Requires: 8GB RAM, 10GB disk space"
echo ""
echo "Methods:"
echo "  • T5-based neural paraphrasing"
echo "  • Back-translation (EN→DE→EN)"
echo "  • SBERT quality validation (0.70 < sim < 0.95)"
echo "  • Lexical diversity check (Jaccard < 0.8)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Installing Python dependencies..."
pip install -q transformers sentence-transformers torch datasets tqdm

echo ""
echo "Starting dataset preparation..."
echo ""

cd scripts

# Step 1: Download datasets (if not exists)
if [ ! -f "../data/msmarco_sample.jsonl" ]; then
    echo "Step 1: Downloading MS MARCO..."
    python3 prepare_datasets_advanced.py \
        --output-dir ../data \
        --sample-size 100000 \
        --seed 42
else
    echo "Step 1: Datasets already exist, skipping download"
    echo ""
    echo "Step 2: Generating advanced paraphrases..."
    python3 prepare_datasets_advanced.py \
        --output-dir ../data \
        --sample-size 100000 \
        --seed 42 \
        --skip-download
fi

cd ..

echo ""
echo "=========================================="
echo "✅ Dataset Preparation Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh data/*_with_paraphrases.jsonl
echo ""
echo "Quality metrics:"
echo "  • T5 paraphrasing: Neural model-based"
echo "  • Back-translation: EN→DE→EN"
echo "  • SBERT validation: 0.70 < similarity < 0.95"
echo "  • Lexical diversity: Jaccard < 0.8"
echo ""
echo "Next steps:"
echo "  1. Verify paraphrase quality:"
echo "     head -5 data/msmarco_sample_with_paraphrases.jsonl | jq ."
echo ""
echo "  2. Run benchmark:"
echo "     ./bin/run_q1_comprehensive_benchmark.sh"
echo ""
