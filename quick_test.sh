#!/bin/bash
set -e

echo "🔍 Quick Test - 10 queries with 20 keys"
echo ""

# Load keys
source .env

# Count keys
KEY_COUNT=$(echo "$GEMINI_API_KEYS" | tr ',' '\n' | wc -l | tr -d ' ')
echo "✅ Loaded $KEY_COUNT API keys"
echo ""

# Calculate capacity
RPM=$((KEY_COUNT * 12))
RPD=$((KEY_COUNT * 1450))
echo "📊 Total capacity: ~${RPM} RPM, ~${RPD} RPD"
echo ""

# Run quick test
echo "🧪 Testing with 10 queries (should take ~30 seconds)..."
echo ""

# Set JVM options
export MAVEN_OPTS="-Xmx2g -Xms1g -XX:+UseG1GC"

mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark \
  -Dspring-boot.run.arguments="--llm.api-keys=${GEMINI_API_KEYS}" \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42 \
  -Dbenchmark.sample-size=10 \
  -Dbenchmark.output-file=results/quick_test.json 2>&1 | \
  grep -E "Multi-key|Progress|Metrics computed|ERROR" || true

echo ""
echo "✅ Test completed!"
echo ""
echo "Check results/quick_test.json for output"
