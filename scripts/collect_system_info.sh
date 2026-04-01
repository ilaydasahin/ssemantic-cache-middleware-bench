#!/bin/bash
# Collects system information for reproducibility documentation
# Output: system_info.json

OUTPUT_FILE="system_info.json"

echo "Collecting system information for reproducibility..."

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    CPU_MODEL=$(sysctl -n machdep.cpu.brand_string)
    TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')
    KERNEL=$(uname -r)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    CPU_MODEL=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
    TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
    KERNEL=$(uname -r)
else
    OS="Unknown"
    CPU_MODEL="Unknown"
    TOTAL_RAM="Unknown"
    KERNEL="Unknown"
fi

# Java version
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)

# Maven version
MAVEN_VERSION=$(mvn -version 2>&1 | head -n 1 | awk '{print $3}')

# Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

# Redis version (if available)
if command -v redis-server &> /dev/null; then
    REDIS_VERSION=$(redis-server --version | awk '{print $3}' | cut -d'=' -f2)
else
    REDIS_VERSION="Not installed"
fi

# Ollama version (if available)
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | awk '{print $NF}')
else
    OLLAMA_VERSION="Not installed"
fi

# Git commit hash (if in git repo)
if git rev-parse --git-dir > /dev/null 2>&1; then
    GIT_COMMIT=$(git rev-parse --short HEAD)
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
else
    GIT_COMMIT="N/A"
    GIT_BRANCH="N/A"
fi

# Timestamp
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Generate JSON
cat > "$OUTPUT_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "hardware": {
    "os": "$OS",
    "kernel": "$KERNEL",
    "cpu": "$CPU_MODEL",
    "ram": "$TOTAL_RAM"
  },
  "software": {
    "java": "$JAVA_VERSION",
    "maven": "$MAVEN_VERSION",
    "python": "$PYTHON_VERSION",
    "redis": "$REDIS_VERSION",
    "ollama": "$OLLAMA_VERSION"
  },
  "repository": {
    "commit": "$GIT_COMMIT",
    "branch": "$GIT_BRANCH"
  }
}
EOF

echo "✅ System information saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
