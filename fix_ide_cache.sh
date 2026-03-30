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
rm -rf *.iml 2>/dev/null
echo "✅ IDE cache temizlendi"

# Yeniden compile
echo "4️⃣ Yeniden compile..."
mvn compile -q
echo "✅ Compile tamamlandı"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 IDE CACHE TEMİZLENDİ!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Şimdi yapmanız gerekenler:"
echo "1. IDE'yi kapatın (VS Code / IntelliJ)"
echo "2. IDE'yi yeniden açın"
echo "3. Workspace'i reload edin"
echo ""
echo "Problemler kaybolmalı! ✅"
