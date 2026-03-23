#!/bin/bash
# Medium Test - 1 hour, 1000 queries

echo "🚀 Medium Test (1 hour, 1000 queries)"
echo "========================================"

mvn spring-boot:run \
  -Dspring-boot.run.profiles=medium \
  -q

echo ""
echo "✅ Medium test completed!"
