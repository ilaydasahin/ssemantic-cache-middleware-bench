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

# Java compile target (from pom.xml)
JAVA_COMPILE_TARGET="25"
if [ -f "pom.xml" ]; then
    COMPILE_TARGET=$(grep -m1 '<maven.compiler.target>' pom.xml 2>/dev/null | sed 's/.*<maven.compiler.target>\(.*\)<\/maven.compiler.target>.*/\1/')
    [ -n "$COMPILE_TARGET" ] && JAVA_COMPILE_TARGET="$COMPILE_TARGET"
fi

# Spring Boot version (from pom.xml)
SPRING_BOOT_VERSION="unknown"
if [ -f "pom.xml" ]; then
    SB_VERSION=$(grep -A1 'spring-boot-starter-parent' pom.xml 2>/dev/null | grep '<version>' | sed 's/.*<version>\(.*\)<\/version>.*/\1/')
    [ -n "$SB_VERSION" ] && SPRING_BOOT_VERSION="$SB_VERSION"
fi

# Maven version
MAVEN_VERSION=$(mvn -version 2>&1 | head -n 1 | awk '{print $3}')

# Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

# Python packages
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "Not installed")
PANDAS_VERSION=$(python3 -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "Not installed")
STATSMODELS_VERSION=$(python3 -c "import statsmodels; print(statsmodels.__version__)" 2>/dev/null || echo "Not installed")

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
    "java_runtime": "$JAVA_VERSION",
    "java_compile_target": "$JAVA_COMPILE_TARGET",
    "spring_boot": "$SPRING_BOOT_VERSION",
    "maven": "$MAVEN_VERSION",
    "python": "$PYTHON_VERSION",
    "numpy": "$NUMPY_VERSION",
    "pandas": "$PANDAS_VERSION",
    "statsmodels": "$STATSMODELS_VERSION",
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
