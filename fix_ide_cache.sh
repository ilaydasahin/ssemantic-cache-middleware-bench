#!/bin/bash

# ============================================
# IDE Cache Temizleme Scripti
# ============================================
# IDE'de görünen eski problemleri temizler

echo "🧹 IDE Cache temizleniyor..."
echo ""

# Maven temizlik
echo "1️⃣ Maven cache temizleme..."
mvn clean -q
echo "✅ Maven cache temizlendi"

# Target dizini
echo "2️⃣ Target dizini temizleme..."
rm -rf target/
echo "✅ Target temizlendi"

# IDE cache dosyaları
echo "3️⃣ IDE cache dosyaları temizleme..."
rm -rf .vscode/.cache 2>/dev/null
rm -rf .idea/ 2>/dev/null
find . -name "*.iml" -delete 2>/dev/null
rm -rf .classpath .project .settings/ .factorypath 2>/dev/null
echo "✅ IDE cache temizlendi"

# Yeniden compile
echo "4️⃣ Yeniden compile..."
mvn compile -q
echo "✅ Compile tamamlandı"

# VS Code settings oluştur
echo "5️⃣ VS Code ayarları güncelleniyor..."
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "java.configuration.updateBuildConfiguration": "automatic",
  "java.compile.nullAnalysis.mode": "automatic",
  "java.saveActions.organizeImports": true
}
EOF
echo "✅ VS Code ayarları güncellendi"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 IDE CACHE TEMİZLENDİ!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Şimdi yapmanız gerekenler:"
echo "1. Kiro'yu yeniden yükleyin (Cmd+R veya Ctrl+R)"
echo "2. Veya tarayıcı sekmesini kapatıp yeniden açın"
echo ""
echo "Problemler kaybolmalı! ✅"

