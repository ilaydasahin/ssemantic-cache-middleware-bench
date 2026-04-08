#!/bin/bash
#
# Prepare 100K Production-Scale Datasets
#
# Generates high-quality datasets with neural paraphrasing:
# - MS MARCO: 100K Q&A pairs
# - Natural Questions: 100K Q&A pairs  
# - Quora Question Pairs: 100K pairs
#
# Methods:
# - T5-based neural paraphrasing
# - Back-translation (EN → DE → EN)
# - SBERT quality validation (0.70 < similarity < 0.95)
# - Lexical diversity check (Jaccard < 0.8)
#
# Requirements:
# - Python 3.8+
# - 16 GB RAM minimum
# - 50 GB disk space
# - GPU recommended (but not required)
#
# Estimated time:
# - CPU only: 4-6 hours
# - With GPU: 2-3 hours
#
# Usage: ./bin/prepare_100k_datasets.sh
#

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         100K PRODUCTION-SCALE DATASET PREPARATION             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will generate 100K queries per dataset with high-quality"
echo "neural paraphrases suitable for Q1 publication."
echo ""
echo "Estimated time: 2-6 hours (depending on hardware)"
echo "Disk space required: ~50 GB"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
cd scripts

if ! python3 -c "import transformers" 2>/dev/null; then
    echo "⚠️  transformers not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies OK"
echo ""

# Check disk space
AVAILABLE_GB=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G.*//')
if [ "$AVAILABLE_GB" -lt 50 ]; then
    echo "⚠️  WARNING: Low disk space (${AVAILABLE_GB}GB available, 50GB recommended)"
    echo "   Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 1: DOWNLOAD AND SAMPLE DATASETS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 prepare_datasets_advanced.py \
    --output-dir ../data \
    --sample-size 100000 \
    --seed 42 \
    --skip-paraphrasing

echo ""
echo "✅ Phase 1 complete: Base datasets downloaded"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 2: GENERATE NEURAL PARAPHRASES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This phase uses T5 and back-translation for high-quality paraphrases."
echo "Progress will be shown with ETA estimates."
echo ""

# Process all datasets
python3 prepare_datasets_advanced.py \
    --output-dir ../data \
    --sample-size 100000 \
    --seed 42 \
    --paraphrase-only \
    --batch-size 32

echo ""
echo "✅ Paraphrase generation complete"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 3: QUALITY VALIDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 validate_paraphrase_quality.py \
    --data-dir ../data \
    --sample-size 1000

echo ""
echo "✅ Quality validation complete"
echo ""

cd ..

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "100K DATASET PREPARATION COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Generated datasets:"
echo "  • data/msmarco_100k_with_paraphrases.jsonl"
echo "  • data/nq_100k_with_paraphrases.jsonl"
echo "  • data/qqp_100k_with_paraphrases.jsonl"
echo ""
echo "Dataset statistics:"
ls -lh data/*_100k_with_paraphrases.jsonl
echo ""
echo "Quality metrics:"
echo "  • Paraphrase method: T5 + back-translation"
echo "  • SBERT similarity: 0.70 < sim < 0.95"
echo "  • Lexical diversity: Jaccard < 0.8"
echo "  • Total queries: 300K (100K × 3 datasets)"
echo ""
echo "Next steps:"
echo "  1. Run convergence analysis with 100K:"
echo "     ./bin/run_convergence_analysis_100k.sh"
echo ""
echo "  2. Run Q1 benchmark with 100K:"
echo "     ./bin/run_q1_comprehensive_benchmark_100k.sh"
echo ""
echo "  3. Update paper to mention 100K scale:"
echo "     'We validate our approach on 100K queries per dataset,"
echo "     totaling 300K queries across three diverse domains.'"
echo ""
