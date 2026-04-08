#!/bin/bash
# Production Load Test - Q1 Publication Requirement
# Demonstrates real-world scalability and performance

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Production Load Test - Q1 Publication             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if K6 is installed
if ! command -v k6 &> /dev/null; then
    echo "❌ K6 not found"
    echo ""
    echo "📥 Install K6:"
    echo "   macOS:   brew install k6"
    echo "   Linux:   sudo apt-get install k6"
    echo "   Windows: choco install k6"
    echo ""
    echo "Or download from: https://k6.io/docs/getting-started/installation/"
    exit 1
fi

echo "✅ K6 found: $(k6 version)"
echo ""

# Check if application is running
echo "🔍 Checking if application is running..."
if ! curl -s http://localhost:8080/actuator/health > /dev/null 2>&1; then
    echo "❌ Application not running on port 8080"
    echo ""
    echo "🚀 Start application first:"
    echo "   mvn spring-boot:run -Dspring-boot.run.profiles=production"
    echo ""
    echo "Or in Docker:"
    echo "   docker-compose up -d"
    echo ""
    exit 1
fi

echo "✅ Application is running"
echo ""

# Create results directory
RESULTS_DIR="results/load_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "📋 Load Test Configuration:"
echo "   Target: http://localhost:8080"
echo "   Duration: ~23 minutes"
echo "   Max VUs: 1000"
echo "   Results: $RESULTS_DIR"
echo ""

echo "🚀 Starting load test..."
echo ""
echo "Test stages:"
echo "  1. Warm up:     50 VUs for 1 min"
echo "  2. Ramp up:    100 VUs for 2 min"
echo "  3. Sustain:    100 VUs for 5 min"
echo "  4. Ramp up:    500 VUs for 2 min"
echo "  5. Sustain:    500 VUs for 5 min"
echo "  6. Ramp up:   1000 VUs for 2 min"
echo "  7. Sustain:   1000 VUs for 5 min"
echo "  8. Ramp down:    0 VUs for 2 min"
echo ""

# Run K6 load test
k6 run \
    --out json="$RESULTS_DIR/load-test-raw.json" \
    --summary-export="$RESULTS_DIR/load-test-summary.json" \
    scripts/load-test.js

echo ""
echo "✅ Load test completed!"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""

# Parse and display key metrics
if [ -f "$RESULTS_DIR/load-test-summary.json" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "KEY METRICS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python3 - <<EOF
import json
import sys

try:
    with open("$RESULTS_DIR/load-test-summary.json") as f:
        data = json.load(f)
    
    metrics = data.get("metrics", {})
    
    # HTTP metrics
    http_reqs = metrics.get("http_reqs", {}).get("values", {})
    http_req_failed = metrics.get("http_req_failed", {}).get("values", {})
    http_req_duration = metrics.get("http_req_duration", {}).get("values", {})
    
    print(f"\nHTTP Requests:")
    print(f"  Total:        {http_reqs.get('count', 0):,}")
    print(f"  Failed:       {http_req_failed.get('rate', 0)*100:.2f}%")
    print(f"  Rate:         {http_reqs.get('rate', 0):.2f} req/s")
    
    print(f"\nLatency:")
    print(f"  P50:          {http_req_duration.get('p(50)', 0):.2f} ms")
    print(f"  P95:          {http_req_duration.get('p(95)', 0):.2f} ms")
    print(f"  P99:          {http_req_duration.get('p(99)', 0):.2f} ms")
    print(f"  Max:          {http_req_duration.get('max', 0):.2f} ms")
    
    # Cache metrics (if available)
    cache_hits = metrics.get("cache_hits", {}).get("values", {})
    if cache_hits:
        print(f"\nCache:")
        print(f"  Hit Rate:     {cache_hits.get('rate', 0)*100:.2f}%")
    
    # Thresholds
    print(f"\nThresholds:")
    for metric_name, metric_data in metrics.items():
        thresholds = metric_data.get("thresholds", {})
        for threshold_name, threshold_data in thresholds.items():
            status = "✅ PASS" if threshold_data.get("ok") else "❌ FAIL"
            print(f"  {status}: {metric_name} {threshold_name}")
    
    print("")
    
except Exception as e:
    print(f"⚠️  Failed to parse summary: {e}", file=sys.stderr)
EOF
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

echo ""
echo "📈 For Q1 publication, include:"
echo "  • Throughput: X,XXX requests/second"
echo "  • P99 Latency: < 1000ms under 1000 concurrent users"
echo "  • Error Rate: < 1%"
echo "  • Cache Hit Rate: > 80%"
echo ""
echo "📊 Visualize results:"
echo "   k6 cloud upload $RESULTS_DIR/load-test-raw.json"
echo ""
echo "Or use Grafana dashboard with Prometheus metrics"
echo ""
