#!/bin/bash
# Background benchmark runner with logging and monitoring

set -e

# Configuration
LOG_DIR="logs"
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/benchmark_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/benchmark.pid"

# Create directories
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Load API keys
if [ -f .env ]; then
    source .env
else
    echo "❌ ERROR: .env file not found"
    echo "Please create .env with your API keys"
    exit 1
fi

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Benchmark already running (PID: $OLD_PID)"
        echo "   Log: $LOG_FILE"
        echo ""
        echo "To stop: kill $OLD_PID"
        echo "To monitor: tail -f $LOG_FILE"
        exit 1
    else
        rm "$PID_FILE"
    fi
fi

# Function to cleanup on exit
cleanup() {
    rm -f "$PID_FILE"
    echo ""
    echo "✅ Benchmark stopped"
}
trap cleanup EXIT

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        SEMANTIC CACHE BENCHMARK - BACKGROUND MODE          ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  📊 Configuration:                                         ║"
echo "║     • 20 API keys loaded                                   ║"
echo "║     • Capacity: ~240 RPM, ~29,000 RPD                     ║"
echo "║     • Log: $LOG_FILE"
echo "║                                                            ║"
echo "║  🎯 Running full benchmark suite...                        ║"
echo "║     • MS MARCO (10K queries)                              ║"
echo "║     • Natural Questions (10K queries)                     ║"
echo "║     • Quora Pairs (10K queries)                           ║"
echo "║                                                            ║"
echo "║  ⏱️  Estimated time: ~3-4 hours                            ║"
echo "║  💰 Cost: $0.00 (free tier)                               ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting in background..."
echo ""

# Run in background
(
    echo "=== Benchmark started at $(date) ===" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Export keys for Maven
    export GEMINI_API_KEYS
    
    # Run benchmark suite
    bash run_full_benchmark_suite.sh >> "$LOG_FILE" 2>&1
    
    echo "" >> "$LOG_FILE"
    echo "=== Benchmark completed at $(date) ===" >> "$LOG_FILE"
) &

# Save PID
BENCHMARK_PID=$!
echo $BENCHMARK_PID > "$PID_FILE"

echo "✅ Benchmark started in background"
echo ""
echo "   PID: $BENCHMARK_PID"
echo "   Log: $LOG_FILE"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f $LOG_FILE | grep -E 'Progress|Metrics|ERROR'"
echo ""
echo "🛑 Stop benchmark:"
echo "   kill $BENCHMARK_PID"
echo ""
echo "📈 Check results:"
echo "   ls -lh $RESULTS_DIR/"
echo ""
