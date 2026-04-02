#!/bin/bash
# Q1 Publication-Ready Comprehensive Benchmark
# This script runs experiments with 26 seeds for statistical power (d=0.8)
# Estimated time: 12-16 hours

set -e

# Configuration
SEEDS=(42 123 456 789 101112 131415 161718 192021 222324 252627 282930 313233 343536 373839 404142 434445 464748 495051 525354 555657 585960 616263 646566 676869 707172 737475)
DATASETS=("msmarco" "natural-questions" "quora-pairs")
STRATEGIES=("EXACT_MATCH" "SEMANTIC" "HYBRID")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/q1_comprehensive_${TIMESTAMP}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Q1 COMPREHENSIVE BENCHMARK - PUBLICATION READY           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  • Seeds: ${#SEEDS[@]} (for statistical power d=0.8)"
echo "  • Datasets: ${#DATASETS[@]} (MS MARCO, Natural Questions, Quora Pairs)"
echo "  • Strategies: ${#STRATEGIES[@]} (EXACT_MATCH, SEMANTIC, HYBRID)"
echo "  • Total experiments: $((${#SEEDS[@]} * ${#DATASETS[@]} * ${#STRATEGIES[@]}))"
echo "  • Results directory: ${RESULTS_DIR}"
echo "  • Estimated time: 12-16 hours"
echo ""

# Pre-flight validation
echo -e "${YELLOW}[1/6] Pre-flight validation...${NC}"
python3 scripts/validate_experiment.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Validation failed. Please fix issues before continuing.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Validation passed${NC}"
echo ""

# Collect system info
echo -e "${YELLOW}[2/6] Collecting system information...${NC}"
bash scripts/collect_system_info.sh
cp system_info.json "${RESULTS_DIR}_system_info.json"
echo -e "${GREEN}✅ System info saved${NC}"
echo ""

# Check services
echo -e "${YELLOW}[3/6] Checking required services...${NC}"
if ! redis-cli ping > /dev/null 2>&1; then
    echo -e "${RED}❌ Redis is not running. Start with: redis-server${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Redis is running${NC}"

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}❌ Ollama is not running. Start with: ollama serve${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Ollama is running${NC}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"
echo "Experiment started at: $(date)" > "${RESULTS_DIR}/experiment_log.txt"
echo "Seeds: ${SEEDS[*]}" >> "${RESULTS_DIR}/experiment_log.txt"
echo "Datasets: ${DATASETS[*]}" >> "${RESULTS_DIR}/experiment_log.txt"
echo "Strategies: ${STRATEGIES[*]}" >> "${RESULTS_DIR}/experiment_log.txt"
echo ""

# Run experiments
echo -e "${YELLOW}[4/6] Running comprehensive benchmark...${NC}"
TOTAL_EXPERIMENTS=$((${#SEEDS[@]} * ${#DATASETS[@]} * ${#STRATEGIES[@]}))
CURRENT=0
FAILED=0

START_TIME=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for STRATEGY in "${STRATEGIES[@]}"; do
            CURRENT=$((CURRENT + 1))
            PROGRESS=$((CURRENT * 100 / TOTAL_EXPERIMENTS))
            
            echo -e "${BLUE}[${CURRENT}/${TOTAL_EXPERIMENTS}] (${PROGRESS}%) Seed=${SEED}, Dataset=${DATASET}, Strategy=${STRATEGY}${NC}"
            
            LOG_FILE="${RESULTS_DIR}/${DATASET}_${SEED}_${STRATEGY}.log"
            
            # Run benchmark
            mvn spring-boot:run \
                -Dspring-boot.run.profiles=benchmark,ollama \
                -Dspring-boot.run.arguments="--mode=throughput --dataset=${DATASET} --seed=${SEED} --strategy=${STRATEGY} --concurrent-users=50" \
                > "${LOG_FILE}" 2>&1
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}  ✅ Success${NC}"
            else
                echo -e "${RED}  ❌ Failed (see ${LOG_FILE})${NC}"
                FAILED=$((FAILED + 1))
            fi
            
            # Brief pause to let system stabilize
            sleep 2
        done
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  BENCHMARK COMPLETE                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  • Total experiments: ${TOTAL_EXPERIMENTS}"
echo "  • Successful: $((TOTAL_EXPERIMENTS - FAILED))"
echo "  • Failed: ${FAILED}"
echo "  • Duration: ${HOURS}h ${MINUTES}m"
echo "  • Results: ${RESULTS_DIR}"
echo ""

if [ ${FAILED} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some experiments failed. Check log files for details.${NC}"
fi

# Statistical analysis
echo -e "${YELLOW}[5/6] Running statistical analysis...${NC}"
python3 scripts/analyze_results.py "${RESULTS_DIR}" > "${RESULTS_DIR}/statistical_analysis.txt" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Statistical analysis complete${NC}"
    echo "  Report: ${RESULTS_DIR}/statistical_analysis.txt"
else
    echo -e "${RED}❌ Statistical analysis failed${NC}"
fi
echo ""

# Bias analysis
echo -e "${YELLOW}[6/6] Running bias analysis...${NC}"
python3 scripts/bias_analysis.py --results-dir "${RESULTS_DIR}" > "${RESULTS_DIR}/bias_analysis.txt" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Bias analysis complete${NC}"
    echo "  Report: ${RESULTS_DIR}/bias_analysis.txt"
else
    echo -e "${RED}❌ Bias analysis failed${NC}"
fi
echo ""

# Generate summary
echo -e "${YELLOW}Generating experiment summary...${NC}"
cat > "${RESULTS_DIR}/README.md" << EOF
# Q1 Comprehensive Benchmark Results

## Experiment Details
- **Date**: $(date)
- **Duration**: ${HOURS}h ${MINUTES}m
- **Seeds**: ${#SEEDS[@]} (${SEEDS[*]})
- **Datasets**: ${#DATASETS[@]} (${DATASETS[*]})
- **Strategies**: ${#STRATEGIES[@]} (${STRATEGIES[*]})
- **Total Experiments**: ${TOTAL_EXPERIMENTS}
- **Success Rate**: $((100 * (TOTAL_EXPERIMENTS - FAILED) / TOTAL_EXPERIMENTS))%

## System Information
See: \`q1_comprehensive_${TIMESTAMP}_system_info.json\`

## Results Files
- Individual logs: \`{dataset}_{seed}_{strategy}.log\`
- Statistical analysis: \`statistical_analysis.txt\`
- Bias analysis: \`bias_analysis.txt\`
- Experiment log: \`experiment_log.txt\`

## Next Steps
1. Review statistical analysis for p-values and effect sizes
2. Check bias analysis for fairness concerns
3. Generate figures and tables for paper
4. Archive results to Zenodo for DOI

## Reproducibility
All experiments are reproducible using:
\`\`\`bash
./run_q1_comprehensive_benchmark.sh
\`\`\`

See \`REPRODUCIBILITY.md\` for full details.
EOF

echo -e "${GREEN}✅ Summary generated: ${RESULTS_DIR}/README.md${NC}"
echo ""

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  ALL TASKS COMPLETE                                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Your Q1-ready comprehensive benchmark is complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results in: ${RESULTS_DIR}"
echo "  2. Check statistical_analysis.txt for significance tests"
echo "  3. Check bias_analysis.txt for fairness metrics"
echo "  4. Generate publication figures"
echo "  5. Archive to Zenodo for DOI"
echo ""
echo -e "${YELLOW}For questions, see REPRODUCIBILITY.md${NC}"
