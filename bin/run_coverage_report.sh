#!/bin/bash
# Test Coverage Report Generator
# Generates comprehensive test coverage report with analysis

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║            Test Coverage Report Generator                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
TARGET_COVERAGE=80
REPORT_DIR="target/site/jacoco"

echo "🧪 Running tests with coverage..."
mvn clean test jacoco:report -q

if [ $? -ne 0 ]; then
    echo "❌ Tests failed!"
    exit 1
fi

echo "✅ Tests completed"
echo ""

# ============================================================================
# PARSE COVERAGE REPORT
# ============================================================================
echo "📊 Analyzing coverage..."
echo ""

if [ ! -f "$REPORT_DIR/index.html" ]; then
    echo "❌ Coverage report not found: $REPORT_DIR/index.html"
    exit 1
fi

# Extract coverage from HTML (more reliable than XML)
COVERAGE=$(grep -A 1 "Total" "$REPORT_DIR/index.html" | grep "ctr2" | head -1 | sed 's/.*%\([0-9]*\).*/\1/' || echo "0")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "COVERAGE SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Current Coverage: ${COVERAGE}%"
echo "  Target Coverage:  ${TARGET_COVERAGE}%"
echo ""

if [ "$COVERAGE" -ge "$TARGET_COVERAGE" ]; then
    echo "  Status: ✅ PASS (meets Q1 requirement)"
else
    GAP=$((TARGET_COVERAGE - COVERAGE))
    echo "  Status: ⚠️  NEEDS IMPROVEMENT"
    echo "  Gap:    ${GAP}%"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# PACKAGE-LEVEL BREAKDOWN
# ============================================================================
echo "📦 Package-level coverage:"
echo ""

# Parse package coverage from HTML
grep -A 10 "tbody" "$REPORT_DIR/index.html" | grep "el_package" | while read line; do
    PACKAGE=$(echo "$line" | sed 's/.*>\(com\.semcache[^<]*\)<.*/\1/')
    NEXT_LINE=$(grep -A 15 "$PACKAGE" "$REPORT_DIR/index.html" | grep "ctr2" | head -1)
    PKG_COVERAGE=$(echo "$NEXT_LINE" | sed 's/.*%\([0-9]*\).*/\1/' || echo "0")
    
    if [ "$PKG_COVERAGE" -ge "$TARGET_COVERAGE" ]; then
        STATUS="✅"
    else
        STATUS="⚠️ "
    fi
    
    printf "  %s %-40s %3s%%\n" "$STATUS" "$PACKAGE" "$PKG_COVERAGE"
done

echo ""

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
if [ "$COVERAGE" -lt "$TARGET_COVERAGE" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "RECOMMENDATIONS TO REACH ${TARGET_COVERAGE}%"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Priority areas (low coverage packages):"
    echo ""
    
    # Find packages with <60% coverage
    grep -A 10 "tbody" "$REPORT_DIR/index.html" | grep "el_package" | while read line; do
        PACKAGE=$(echo "$line" | sed 's/.*>\(com\.semcache[^<]*\)<.*/\1/')
        NEXT_LINE=$(grep -A 15 "$PACKAGE" "$REPORT_DIR/index.html" | grep "ctr2" | head -1)
        PKG_COVERAGE=$(echo "$NEXT_LINE" | sed 's/.*%\([0-9]*\).*/\1/' || echo "0")
        
        if [ "$PKG_COVERAGE" -lt 60 ]; then
            echo "  📝 $PACKAGE (${PKG_COVERAGE}%)"
            
            # Suggest test types
            case "$PACKAGE" in
                *controller*)
                    echo "     → Add integration tests (MockMvc)"
                    ;;
                *service*)
                    echo "     → Add unit tests with mocks"
                    ;;
                *benchmark*)
                    echo "     → Add benchmark runner tests"
                    ;;
                *notification*)
                    echo "     → Add notification service tests"
                    ;;
                *)
                    echo "     → Add unit tests"
                    ;;
            esac
            echo ""
        fi
    done
    
    echo "Quick wins:"
    echo "  • Add tests for edge cases (null inputs, empty lists)"
    echo "  • Test error handling paths"
    echo "  • Test configuration validation"
    echo "  • Test utility methods"
    echo ""
fi

# ============================================================================
# REPORT LOCATION
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DETAILED REPORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "HTML Report: $REPORT_DIR/index.html"
echo ""
echo "Open in browser:"
echo "  macOS:   open $REPORT_DIR/index.html"
echo "  Linux:   xdg-open $REPORT_DIR/index.html"
echo "  Windows: start $REPORT_DIR/index.html"
echo ""

# ============================================================================
# Q1 PUBLICATION CHECKLIST
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Q1 PUBLICATION CHECKLIST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$COVERAGE" -ge "$TARGET_COVERAGE" ]; then
    echo "  ✅ Test coverage ≥80%"
else
    echo "  ⚠️  Test coverage <80% (current: ${COVERAGE}%)"
fi

# Check if tests pass
if mvn test -q > /dev/null 2>&1; then
    echo "  ✅ All tests pass"
else
    echo "  ❌ Some tests fail"
fi

# Check test count
TEST_COUNT=$(find src/test -name "*Test.java" | wc -l | xargs)
echo "  ℹ️  Test files: $TEST_COUNT"

echo ""
echo "For Q1 publication, include in paper:"
echo "  • Test coverage: ${COVERAGE}%"
echo "  • Test count: $TEST_COUNT test classes"
echo "  • Testing framework: JUnit 5 + Mockito + AssertJ"
echo "  • Coverage tool: JaCoCo"
echo ""

exit 0
