#!/bin/bash
# Full Test - 3 hours, 10000 queries

echo "🚀 Full Benchmark (3 hours, 10000 queries)"
echo "==========================================="

mvn spring-boot:run \
  -Dspring-boot.run.profiles=full \
  -q

echo ""
echo "✅ Full benchmark completed!"
