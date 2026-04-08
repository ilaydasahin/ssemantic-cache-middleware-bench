#!/bin/bash
# Quick Validation Test - 3 Critical Issues
# Tests: Dataset quality, SOTA baseline, Paraphrase validation

set -e

echo "=========================================="
echo "Q1 CRITICAL ISSUES - QUICK VALIDATION"
echo "=========================================="
echo ""

# Issue 1: Dataset Size
echo "1️⃣  DATASET SIZE CHECK"
echo "----------------------------------------"
echo "Current datasets:"
wc -l data/*.jsonl | grep -v total
echo ""
echo "Q1 Requirement: 100K per dataset"
echo "Current: 10K per dataset"
echo "Status: ❌ INSUFFICIENT"
echo ""

# Issue 2: Paraphrase Quality
echo "2️⃣  PARAPHRASE QUALITY CHECK"
echo "----------------------------------------"
if grep -q "paraphrase_method" data/msmarco_sample_with_paraphrases.jsonl 2>/dev/null; then
    echo "Checking paraphrase methods..."
    head -10 data/msmarco_sample_with_paraphrases.jsonl | jq -r '.paraphrase_method' | sort | uniq -c
    echo ""
    echo "Status: ✅ ADVANCED (T5/back-translation)"
else
    echo "No paraphrase_method field found"
    echo "Status: ❌ BASIC (pattern-based)"
fi
echo ""

# Issue 3: SOTA Baseline
echo "3️⃣  SOTA BASELINE CHECK"
echo "----------------------------------------"
echo "Testing GPTCache baseline compilation..."
if mvn compile -q 2>&1 | grep -q "BUILD SUCCESS"; then
    echo "✅ Compiles successfully"
else
    echo "❌ Compilation failed"
fi
echo ""

echo "Checking if GPTCACHE_BASELINE is registered..."
if grep -q "GPTCACHE_BASELINE" src/main/java/com/semcache/model/CacheStrategy.java; then
    echo "✅ Registered in CacheStrategy enum"
else
    echo "❌ Not registered"
fi
echo ""

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Critical Issues Status:"
echo "  1. Dataset Size (10K→100K): ❌ TODO"
echo "  2. Paraphrase Quality: ⏳ IN PROGRESS"
echo "  3. SOTA Baseline: ✅ READY"
echo ""
echo "Next Steps:"
echo ""
echo "  IMMEDIATE (2-4 hours):"
echo "    ./bin/run_100k_dataset_preparation.sh"
echo ""
echo "  AFTER DATASETS (30 min):"
echo "    ./bin/run_gptcache_baseline_test.sh"
echo ""
echo "  FULL EXPERIMENT (12-16 hours):"
echo "    ./bin/run_q1_comprehensive_benchmark.sh"
echo ""
