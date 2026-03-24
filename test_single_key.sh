#!/bin/bash

# Test with single key (16GB RAM optimized)
export GEMINI_API_KEYS="AIzaSyCAcLnwrC3gqK1pf8lt6IrLR0MoqzGw2NM"

echo "🧪 Testing with single key (RAM optimized)..."
echo "Key: AIzaSyCAcLnwrC3gqK1pf8lt6IrLR0MoqzGw2NM"

# Build
mvn clean package -DskipTests

# Run small test (10 samples) with limited heap and benchmark profile
java -Xmx2G -Xms1G -jar target/semantic-cache-benchmark-1.0.0.jar \
  --spring.profiles.active=benchmark \
  --benchmark.current-dataset=msmarco \
  --benchmark.sample-size=10

echo ""
echo "✅ Test complete. Check logs/benchmark-current.log for details"
