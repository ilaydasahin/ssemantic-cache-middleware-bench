#!/bin/bash
# Production Load Test - 1000+ RPS Validation
# Tests throughput claims for Q1 publication

set -e

echo "====================================================="
echo " PRODUCTION LOAD TEST (1000+ RPS)"
echo "====================================================="

# Check if k6 is installed
if ! command -v k6 &> /dev/null; then
    echo "❌ k6 not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install k6
    else
        echo "Please install k6: https://k6.io/docs/getting-started/installation/"
        exit 1
    fi
fi

# Start the application in background
echo "Starting application..."
mvn spring-boot:run -Dspring-boot.run.profiles=benchmark &
APP_PID=$!

# Wait for application to be ready
echo "Waiting for application to start..."
sleep 30

# Check if app is running
if ! curl -s http://localhost:8080/actuator/health > /dev/null; then
    echo "❌ Application failed to start"
    kill $APP_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Application ready"

# Run load tests with increasing load
RESULTS_DIR="results/load_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "--- Test 1: Baseline (100 VUs, 2min) ---"
k6 run --vus 100 --duration 2m \
    --out json="$RESULTS_DIR/baseline_100vu.json" \
    scripts/load-test.js

echo ""
echo "--- Test 2: Medium Load (500 VUs, 5min) ---"
k6 run --vus 500 --duration 5m \
    --out json="$RESULTS_DIR/medium_500vu.json" \
    scripts/load-test.js

echo ""
echo "--- Test 3: High Load (1000 VUs, 5min) ---"
k6 run --vus 1000 --duration 5m \
    --out json="$RESULTS_DIR/high_1000vu.json" \
    scripts/load-test.js

echo ""
echo "--- Test 4: Stress Test (2000 VUs, 3min) ---"
k6 run --vus 2000 --duration 3m \
    --out json="$RESULTS_DIR/stress_2000vu.json" \
    scripts/load-test.js

# Stop application
echo ""
echo "Stopping application..."
kill $APP_PID 2>/dev/null || true
wait $APP_PID 2>/dev/null || true

echo ""
echo "====================================================="
echo " ✅ LOAD TEST COMPLETED"
echo " Results saved to: $RESULTS_DIR"
echo "====================================================="
echo ""
echo "Next steps:"
echo "  1. Analyze results: python3 scripts/analyze_load_test.py --input-dir $RESULTS_DIR"
echo "  2. Generate report: python3 scripts/generate_publication_figures.py --load-test-dir $RESULTS_DIR"
