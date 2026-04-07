#!/bin/bash
# Ollama Watchdog - Restarts Ollama if it crashes during long experiments

set -e

OLLAMA_URL="http://localhost:11434/api/tags"
CHECK_INTERVAL=60  # Check every 60 seconds
MAX_RETRIES=3
LOG_FILE="logs/ollama_watchdog.log"

mkdir -p logs

echo "🐕 Ollama Watchdog started at $(date)" | tee -a "$LOG_FILE"
echo "   Monitoring: $OLLAMA_URL" | tee -a "$LOG_FILE"
echo "   Check interval: ${CHECK_INTERVAL}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    if curl -s --max-time 5 "$OLLAMA_URL" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Ollama is healthy" >> "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Ollama is DOWN!" | tee -a "$LOG_FILE"
        
        # Try to restart
        for i in $(seq 1 $MAX_RETRIES); do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt $i/$MAX_RETRIES: Restarting Ollama..." | tee -a "$LOG_FILE"
            
            # Kill existing Ollama processes
            pkill -9 ollama || true
            sleep 2
            
            # Start Ollama
            ollama serve > /dev/null 2>&1 &
            sleep 5
            
            # Check if it's up
            if curl -s --max-time 5 "$OLLAMA_URL" > /dev/null 2>&1; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Ollama restarted successfully" | tee -a "$LOG_FILE"
                break
            fi
            
            if [ $i -eq $MAX_RETRIES ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed to restart Ollama after $MAX_RETRIES attempts" | tee -a "$LOG_FILE"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚨 MANUAL INTERVENTION REQUIRED" | tee -a "$LOG_FILE"
                # Send notification (optional)
                # osascript -e 'display notification "Ollama watchdog failed" with title "Benchmark Alert"'
            fi
        done
    fi
    
    sleep $CHECK_INTERVAL
done
