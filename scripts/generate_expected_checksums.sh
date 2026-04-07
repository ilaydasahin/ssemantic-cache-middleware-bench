#!/bin/bash
# Generate SHA-256 checksums for datasets
# This ensures data integrity and reproducibility

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
OUTPUT_FILE="${SCRIPT_DIR}/expected_checksums.txt"

echo "=== Generating Dataset Checksums ==="
echo ""

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "   Run: python3 prepare_datasets.py"
    exit 1
fi

# Check if datasets exist
DATASETS=(
    "msmarco_sample.jsonl"
    "msmarco_sample_with_paraphrases.jsonl"
    "nq_sample.jsonl"
    "nq_sample_with_paraphrases.jsonl"
    "qqp_sample.jsonl"
    "qqp_sample_with_paraphrases.jsonl"
)

MISSING=0
for dataset in "${DATASETS[@]}"; do
    if [ ! -f "$DATA_DIR/$dataset" ]; then
        echo "⚠️  Missing: $dataset"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "❌ Error: $MISSING dataset(s) missing"
    echo "   Run: python3 prepare_datasets.py"
    exit 1
fi

# Generate checksums
echo "Calculating SHA-256 checksums..."
echo ""

cd "$DATA_DIR"

if command -v sha256sum &> /dev/null; then
    # Linux
    sha256sum *.jsonl > "$OUTPUT_FILE"
elif command -v shasum &> /dev/null; then
    # macOS
    shasum -a 256 *.jsonl > "$OUTPUT_FILE"
else
    echo "❌ Error: Neither sha256sum nor shasum found"
    exit 1
fi

cd - > /dev/null

echo "✅ Checksums saved to: $OUTPUT_FILE"
echo ""
echo "Contents:"
cat "$OUTPUT_FILE"
echo ""
echo "=== Done ==="
