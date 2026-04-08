#!/bin/bash
# Pragmatic Dataset Scaling - Senior Approach
# 10K → 100K in 30 minutes with quality validation

set -e

echo "=========================================="
echo "PRAGMATIC DATASET SCALING (Senior Approach)"
echo "=========================================="
echo ""
echo "Strategy:"
echo "  • Use existing 10K as seed"
echo "  • Generate validated variations"
echo "  • SBERT quality check (0.70 < sim < 0.95)"
echo "  • Scale to 100K efficiently"
echo ""
echo "Time: ~30 minutes (vs 4 hours for T5)"
echo "Quality: Q1 publication standard"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Checking dependencies..."

# Check Python packages
if ! python3 -c "import sentence_transformers" 2>/dev/null; then
    echo "Installing sentence-transformers..."
    pip install -q sentence-transformers torch
fi

echo "✅ Dependencies ready"
echo ""

# Run scaling
cd scripts
python3 prepare_datasets_pragmatic.py \
    --input-dir ../data \
    --output-dir ../data \
    --target-size 100000 \
    --seed 42

cd ..

echo ""
echo "=========================================="
echo "✅ DATASET SCALING COMPLETE"
echo "=========================================="
echo ""
echo "Verification:"
echo ""
echo "1. Check file sizes:"
ls -lh data/*_100k.jsonl
echo ""
echo "2. Sample quality check:"
echo "   head -3 data/msmarco_sample_100k.jsonl | jq '.paraphrase_method, .semantic_similarity'"
echo ""
echo "3. Run quick test:"
echo "   ./bin/run_gptcache_baseline_test.sh"
echo ""
