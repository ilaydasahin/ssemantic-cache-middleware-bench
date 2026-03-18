#!/bin/bash
# Test script for multi-key setup
# Tests with a small sample to verify keys are working

set -e

echo "🔍 Testing Multi-Key Setup..."
echo ""

# Check if keys are set
if [ -z "$GEMINI_API_KEYS" ]; then
    echo "❌ ERROR: GEMINI_API_KEYS not set"
    echo ""
    echo "Please set your API keys:"
    echo "  export GEMINI_API_KEYS=\"key1,key2,key3,...\""
    echo ""
    echo "Or see MULTI_KEY_SETUP.md for detailed instructions"
    exit 1
fi

# Count keys
KEY_COUNT=$(echo "$GEMINI_API_KEYS" | tr ',' '\n' | wc -l | tr -d ' ')
echo "✅ Found $KEY_COUNT API keys"
echo ""

# Calculate capacity
RPM=$((KEY_COUNT * 12))
RPD=$((KEY_COUNT * 1450))
echo "📊 Estimated capacity:"
echo "   - Rate: ~${RPM} requests/minute"
echo "   - Daily: ~${RPD} requests/day"
echo ""

# Run small test
echo "🧪 Running test with 50 queries..."
echo ""

mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42 \
  -Dbenchmark.sample-size=50 \
  -Dbenchmark.output-file=results/multi_key_test.json \
  -q

echo ""
echo "✅ Test completed!"
echo ""
echo "Check results/multi_key_test.json for output"
echo ""
echo "To run full benchmark:"
echo "  bash run_full_benchmark_suite.sh"
