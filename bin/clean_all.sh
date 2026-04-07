#!/bin/bash
# Clean all generated files and results

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Clean All Generated Files                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "⚠️  This will delete:"
echo "   • All results in results/"
echo "   • All logs in logs/"
echo "   • All checkpoints in checkpoints/"
echo "   • Maven build artifacts in target/"
echo "   • Python cache files"
echo ""

read -p "Are you sure? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🧹 Cleaning..."

# Clean results
if [ -d "results" ]; then
    echo "  Removing results..."
    find results -type f -name "*.json" -delete
    find results -type f -name "*.jsonl" -delete
    find results -type f -name "*.log" -delete
    find results -type f -name "*.txt" -delete
    find results -type f -name "*.csv" -delete
    find results -type f -name "*.png" -delete
    find results -type d -empty -delete
    echo "  ✅ Results cleaned"
fi

# Clean logs
if [ -d "logs" ]; then
    echo "  Removing logs..."
    find logs -type f -name "*.log" -delete
    echo "  ✅ Logs cleaned"
fi

# Clean checkpoints
if [ -d "checkpoints" ]; then
    echo "  Removing checkpoints..."
    find checkpoints -type f -delete
    echo "  ✅ Checkpoints cleaned"
fi

# Clean Maven build
if [ -d "target" ]; then
    echo "  Removing Maven build artifacts..."
    rm -rf target
    echo "  ✅ Maven artifacts cleaned"
fi

# Clean Python cache
echo "  Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✅ Python cache cleaned"

# Clean IDE files
if [ -d ".idea" ]; then
    echo "  Removing IntelliJ IDEA files..."
    rm -rf .idea
    echo "  ✅ IDE files cleaned"
fi

# Clean macOS files
echo "  Removing macOS files..."
find . -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✅ macOS files cleaned"

echo ""
echo "✅ All generated files cleaned!"
echo ""
echo "📝 Note: The following are preserved:"
echo "   • Source code (src/)"
echo "   • Configuration files"
echo "   • Datasets (data/)"
echo "   • Models (models/)"
echo "   • Scripts (scripts/)"
echo ""
