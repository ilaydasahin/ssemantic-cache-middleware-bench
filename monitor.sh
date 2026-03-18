#!/bin/bash
# Real-time monitoring for background benchmark

LOG_DIR="logs"
PID_FILE="${LOG_DIR}/benchmark.pid"

# Check if benchmark is running
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No benchmark running"
    echo ""
    echo "Start with: bash run_background.sh"
    exit 1
fi

PID=$(cat "$PID_FILE")
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "❌ Benchmark process not found (PID: $PID)"
    rm -f "$PID_FILE"
    exit 1
fi

# Find latest log file
LATEST_LOG=$(ls -t ${LOG_DIR}/benchmark_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log file found"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              BENCHMARK MONITORING                          ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  PID: $PID"
echo "║  Log: $LATEST_LOG"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Live progress (Ctrl+C to exit monitoring):"
echo ""

# Monitor with colored output
tail -f "$LATEST_LOG" | grep --line-buffered -E "Multi-key|Progress|Metrics computed|ERROR|WARN|completed|exhausted" | while read line; do
    if echo "$line" | grep -q "ERROR"; then
        echo "🔴 $line"
    elif echo "$line" | grep -q "WARN"; then
        echo "🟡 $line"
    elif echo "$line" | grep -q "Progress"; then
        echo "📈 $line"
    elif echo "$line" | grep -q "Metrics computed"; then
        echo "✅ $line"
    elif echo "$line" | grep -q "Multi-key"; then
        echo "🔑 $line"
    else
        echo "ℹ️  $line"
    fi
done
