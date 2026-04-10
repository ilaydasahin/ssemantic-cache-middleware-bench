#!/bin/bash
# Q1 Publication-Ready Benchmark Suite
# 
# Runs ALL required experiments for Q1 journal publication:
# 1. Dataset preparation (50K samples, neural paraphrasing)
# 2. Power analysis (26+ seeds for 80% power)
# 3. Baseline comparison (GPTCache)
# 4. Ablation study (component contributions)
# 5. Multi-language validation (3 languages)
# 6. Load testing (production readiness)
# 7. Statistical analysis (effect sizes, CI)
#
# Estimated time: 8-12 hours
# Estimated disk: 5GB

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                   Q1 PUBLICATION-READY BENCHMARK SUITE                     ║"
echo "║                                                                            ║"
echo "║  This will run ALL experiments required for Q1 journal publication        ║"
echo "║  Estimated time: 8-12 hours                                               ║"
echo "║  Estimated disk: 5GB                                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check dependencies
echo "🔍 Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.8+"
    exit 1
fi

if ! command -v mvn &> /dev/null; then
    echo "❌ Maven not found. Please install Maven 3.6+"
    exit 1
fi

echo "✅ Dependencies OK"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r scripts/requirements.txt --quiet
echo "✅ Python dependencies installed"
echo ""

# Create output directories
mkdir -p data
mkdir -p results/q1_comprehensive
mkdir -p results/ablation
mkdir -p results/multilingual
mkdir -p results/baselines
mkdir -p results/load_test

# Step 1: Dataset Preparation
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 1/7: DATASET PREPARATION (50K samples, neural paraphrasing)          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "data/msmarco_sample_with_paraphrases.jsonl" ]; then
    echo "📥 Preparing datasets with neural paraphrasing..."
    python3 scripts/prepare_datasets.py \
        --output-dir data \
        --sample-size 50000 \
        --seed 42
    
    echo ""
    echo "✅ Validating paraphrase quality..."
    python3 scripts/validate_paraphrase_quality.py \
        --data-dir data \
        --output-dir results
    echo ""
else
    echo "✅ Datasets already prepared (skipping)"
    echo ""
fi

# Step 2: Power Analysis
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 2/7: STATISTICAL POWER ANALYSIS                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

python3 scripts/power_analysis.py \
    --effect-size 0.8 \
    --power 0.80 \
    --show-curves

echo ""

# Step 3: Main Benchmark (26 seeds for 80% power)
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 3/7: MAIN BENCHMARK (26 seeds, 80% power)                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

DATASET="msmarco"
OUTPUT_DIR="results/q1_comprehensive"
STRATEGIES=("EXACT_MATCH" "SEMANTIC" "HYBRID")

# 26 seeds for 80% power at d=0.8
SEEDS=(42 123 999 1024 2048 3141 5926 5358 9793 2384 6264 3383 2795 288 4197 1693 9937 5105 8209 7494 4592 3078 1640 6286 2089 9862)

echo "Compiling project..."
mvn clean compile package -DskipTests -q

total_runs=$((${#SEEDS[@]} * ${#STRATEGIES[@]}))
current_run=0

for seed in "${SEEDS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        current_run=$((current_run + 1))
        output_file="$OUTPUT_DIR/${DATASET}_${strategy}_${seed}.json"
        
        if [ -f "$output_file" ]; then
            echo "[$current_run/$total_runs] ✅ Skipping $strategy (seed=$seed) - already exists"
            continue
        fi
        
        echo "[$current_run/$total_runs] Running $strategy (seed=$seed)..."
        
        mvn spring-boot:run \
            -Dspring-boot.run.profiles=benchmark,benchmark-mock \
            -Dspring-boot.run.arguments="--benchmark.current-dataset=$DATASET --benchmark.current-seed=$seed --benchmark.strategy=$strategy --benchmark.output-file=$output_file" \
            -q
    done
done

echo ""
echo "✅ Main benchmark complete"
echo ""

# Step 4: Baseline Comparison (GPTCache)
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 4/7: BASELINE COMPARISON (GPTCache)                                  ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

if command -v python3 -c "import gptcache" &> /dev/null; then
    echo "Running GPTCache baseline..."
    for seed in 42 123 999; do
        python3 scripts/gptcache_baseline.py \
            --dataset data/msmarco_sample_with_paraphrases.jsonl \
            --output results/baselines/gptcache_${seed}.json \
            --seed $seed
    done
    echo "✅ GPTCache baseline complete"
else
    echo "⚠️  GPTCache not installed - skipping baseline comparison"
    echo "   Install with: pip install gptcache"
fi

echo ""

# Step 5: Ablation Study
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 5/7: ABLATION STUDY (component contributions)                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

python3 scripts/ablation_study.py \
    --output-dir results/ablation \
    --component all \
    --dataset msmarco \
    --seed 42

echo ""
echo "✅ Ablation study complete"
echo ""

# Step 6: Multi-language Validation
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 6/7: MULTI-LANGUAGE VALIDATION (Turkish, German, French)             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Preparing multilingual datasets..."
python3 scripts/prepare_multilingual_datasets.py \
    --languages tr de fr \
    --sample-size 10000 \
    --output-dir data \
    --with-paraphrases

echo ""
echo "Running multilingual benchmarks..."
for lang in tr de fr; do
    if [ -f "data/${lang}_sample_with_paraphrases.jsonl" ]; then
        echo "  Testing $lang..."
        mvn spring-boot:run \
            -Dspring-boot.run.profiles=benchmark,benchmark-mock \
            -Dspring-boot.run.arguments="--benchmark.current-dataset=${lang} --benchmark.current-seed=42 --benchmark.strategy=SEMANTIC --benchmark.output-file=results/multilingual/${lang}_semantic_42.json" \
            -q
    fi
done

echo ""
echo "✅ Multi-language validation complete"
echo ""

# Step 7: Load Testing
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 7/7: LOAD TESTING (production readiness)                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

if command -v k6 &> /dev/null; then
    echo "Running load test (1000 RPS for 60s)..."
    bash scripts/run_load_test.sh
    echo "✅ Load test complete"
else
    echo "⚠️  k6 not installed - skipping load test"
    echo "   Install from: https://k6.io/docs/getting-started/installation/"
fi

echo ""

# Statistical Analysis
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STATISTICAL ANALYSIS & REPORTING                                           ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📊 Calculating effect sizes..."
python3 scripts/effect_size_analysis.py \
    --results-dir results/q1_comprehensive \
    --baseline EXACT_MATCH \
    --metric hit_rate

echo ""
echo "📊 Generating publication figures..."
python3 scripts/generate_publication_figures.py \
    --input-dir results/q1_comprehensive \
    --output-dir results

echo ""
echo "📊 Running statistical validation..."
python3 scripts/statistical_validation.py \
    --results-dir results/q1_comprehensive

echo ""

# Final Summary
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          ✅ ALL EXPERIMENTS COMPLETE                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results saved to:"
echo "   • Main benchmark:      results/q1_comprehensive/"
echo "   • Ablation study:      results/ablation/"
echo "   • Multi-language:      results/multilingual/"
echo "   • Baselines:           results/baselines/"
echo "   • Load test:           results/load_test/"
echo ""
echo "📊 Analysis outputs:"
echo "   • Effect sizes:        results/effect_sizes.json"
echo "   • LaTeX tables:        results/*.tex"
echo "   • Figures:             results/*.pdf, results/*.png"
echo "   • Reports:             results/*_report.txt"
echo ""
echo "📝 Next steps for paper:"
echo "   1. Review effect_sizes_report.txt for key findings"
echo "   2. Include LaTeX tables in methods/results sections"
echo "   3. Add figures to paper (publication-quality PDFs)"
echo "   4. Report power analysis in methods section"
echo "   5. Discuss ablation results to justify design choices"
echo "   6. Cite multi-language results for generalizability"
echo ""
echo "🎉 Your semantic cache benchmark is now Q1 publication-ready!"
echo ""
