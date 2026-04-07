#!/bin/bash
# Verify dataset integrity using SHA-256 checksums
# Ensures reproducibility and detects data corruption

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
EXPECTED_FILE="${SCRIPT_DIR}/expected_checksums.txt"

echo "=== Verifying Dataset Integrity ==="
echo ""

if [ ! -f "$EXPECTED_FILE" ]; then
    echo "❌ Error: Expected checksums file not found: $EXPECTED_FILE"
    echo "   Run: bash generate_expected_checksums.sh"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    exit 1
fi

cd "$DATA_DIR"

# Verify checksums
if command -v sha256sum &> /dev/null; then
    # Linux
    sha256sum -c "$EXPECTED_FILE"
elif command -v shasum &> /dev/null; then
    # macOS
    shasum -a 256 -c "$EXPECTED_FILE"
else
    echo "❌ Error: Neither sha256sum nor shasum found"
    exit 1
fi

RESULT=$?

cd - > /dev/null

echo ""
if [ $RESULT -eq 0 ]; then
    echo "✅ All checksums verified successfully"
    echo "   Datasets are intact and reproducible"
else
    echo "❌ Checksum verification failed"
    echo "   Datasets may be corrupted or modified"
    echo "   Re-run: python3 prepare_datasets.py"
    exit 1
fi

echo ""
echo "=== Done ==="
