#!/bin/bash
# Quick Test - 5 minutes, 100 queries

echo "🚀 Quick Test (5 minutes, 100 queries)"
echo "========================================"

mvn spring-boot:run \
  -Dspring-boot.run.profiles=quick \
  -q

echo ""
echo "✅ Quick test completed!"
