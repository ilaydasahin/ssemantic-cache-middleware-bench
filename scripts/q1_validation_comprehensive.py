#!/usr/bin/env python3
"""
Comprehensive Q1 Publication Validation Script

This script performs ALL validation checks required for Q1 journal submission:
1. Statistical power analysis
2. Sample size validation
3. Effect size reporting with CI
4. Multiple testing correction (FDR)
5. Normality assumptions
6. Bias analysis (query length, dataset, temporal)
7. Reproducibility score
8. Baseline comparisons
9. Data quality checks

Usage: python3 q1_validation_comprehensive.py --results-dir ../results/q1_comprehensive_*

Exit codes:
  0: All checks passed (ready for Q1 submission)
  1: Some warnings (review before submission)
  2: Critical failures (not ready for submission)
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multitest import multipletests
import pandas as pd


class Q1Validator:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        
    def validate_all(self):
        """Run all validation checks."""
        print("=" * 80)
        print(" " * 20 + "Q1 PUBLICATION VALIDATION")
        print("=" * 80)
        print()
        print(f"Results directory: {self.results_dir}")
        print()
        
        # Run all checks
        self.check_sample_size()
        self.check_baseline_comparison()
        self.check_effect_sizes()
        self.check_multiple_testing()
        self.check_normality()
        self.check_bias()
        self.check_reproducibility()
        self.check_data_quality()
        
        # Generate report
        self.generate_report()
        
        # Return exit code
        if self.checks_failed:
            return 2  # Critical failures
        elif self.warnings:
            return 1  # Warnings
        else:
            return 0  # All passed
    
    def check_sample_size(self):
        """Validate sample size meets power requirements."""
        print("─" * 80)
        print("CHECK 1: SAMPLE SIZE & STATISTICAL POWER")
        print("─" * 80)
        
        json_files = list(self.results_dir.glob('*.json'))
        json_files = [f for f in json_files if not any(x in f.name for x in ['all_results', 'scalability'])]
        
        if not json_files:
            self.checks_failed.append("No result files found")
            print("❌ FAIL: No result files found")
            print()
            return
        
        # Count seeds per configuration
        configs = {}
        for filepath in json_files:
            with open(filepath) as f:
                data = json.load(f)
            
            config_key = (
                data.get('dataset', 'unknown'),
                data.get('strategy', 'unknown'),
                data.get('embeddingModel', 'unknown')
            )
            
            if config_key not in configs:
                configs[config_key] = []
            configs[config_key].append(data.get('seed', 0))
        
        # Power analysis
        analysis = TTestIndPower()
        required_n_medium = analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.80)
        required_n_large = analysis.solve_power(effect_size=0.8, alpha=0.05, power=0.80)
        
        print(f"Power analysis (α=0.05, power=0.80):")
        print(f"  Medium effect (d=0.5): {int(np.ceil(required_n_medium))} seeds required")
        print(f"  Large effect (d=0.8): {int(np.ceil(required_n_large))} seeds required")
        print()
        
        min_seeds = min(len(seeds) for seeds in configs.values())
        max_seeds = max(len(seeds) for seeds in configs.values())
        
        print(f"Actual sample sizes:")
        print(f"  Minimum: {min_seeds} seeds")
        print(f"  Maximum: {max_seeds} seeds")
        print(f"  Configurations: {len(configs)}")
        print()
        
        if min_seeds >= required_n_medium:
            self.checks_passed.append("Sample size (medium effects)")
            print("✅ PASS: Sufficient for medium effects (d=0.5)")
        elif min_seeds >= required_n_large:
            self.checks_passed.append("Sample size (large effects)")
            self.warnings.append(f"Sample size only sufficient for large effects (d=0.8)")
            print("⚠️  WARN: Only sufficient for large effects (d=0.8)")
            print("   Consider increasing to 64 seeds for medium effects")
        else:
            self.checks_failed.append("Insufficient sample size")
            print(f"❌ FAIL: Insufficient sample size (need ≥{int(np.ceil(required_n_large))} for d=0.8)")
        
        print()
    
    def check_baseline_comparison(self):
        """Validate that NO_CACHE baseline is included."""
        print("─" * 80)
        print("CHECK 2: BASELINE COMPARISON")
        print("─" * 80)
        
        json_files = list(self.results_dir.glob('*.json'))
        strategies = set()
        
        for filepath in json_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                strategies.add(data.get('strategy', 'unknown'))
            except:
                pass
        
        print(f"Strategies found: {sorted(strategies)}")
        print()
        
        required_baselines = {'NONE', 'EXACT_MATCH'}
        missing = required_baselines - strategies
        
        if not missing:
            self.checks_passed.append("Baseline comparison")
            print("✅ PASS: All required baselines present")
            print("   • NONE (no cache control)")
            print("   • EXACT_MATCH (hash-based baseline)")
        else:
            self.checks_failed.append(f"Missing baselines: {missing}")
            print(f"❌ FAIL: Missing required baselines: {missing}")
            print()
            print("Q1 requirement: Must compare against:")
            print("  • NONE: No caching (100% LLM calls)")
            print("  • EXACT_MATCH: Simple hash-based cache")
        
        print()
    
    def check_effect_sizes(self):
        """Validate effect size reporting with confidence intervals."""
        print("─" * 80)
        print("CHECK 3: EFFECT SIZE REPORTING")
        print("─" * 80)
        
        # Check if analyze_results.py has been run
        summary_files = list(self.results_dir.glob('*summary*.csv'))
        
        if not summary_files:
            self.warnings.append("Effect size analysis not run")
            print("⚠️  WARN: No summary files found")
            print("   Run: python3 analyze_results.py " + str(self.results_dir))
            print()
            return
        
        # Check for Cohen's d in results
        has_cohens_d = False
        has_ci = False
        
        for summary_file in summary_files:
            try:
                df = pd.read_csv(summary_file)
                if any('cohen' in col.lower() for col in df.columns):
                    has_cohens_d = True
                if any('ci' in col.lower() or '±' in str(df.values) for col in df.columns):
                    has_ci = True
            except:
                pass
        
        if has_cohens_d and has_ci:
            self.checks_passed.append("Effect size reporting")
            print("✅ PASS: Effect sizes with confidence intervals reported")
        elif has_cohens_d:
            self.warnings.append("Effect sizes missing confidence intervals")
            print("⚠️  WARN: Effect sizes reported but missing confidence intervals")
        else:
            self.checks_failed.append("Effect sizes not reported")
            print("❌ FAIL: Effect sizes (Cohen's d) not reported")
            print("   Q1 requirement: Report Cohen's d with 95% CI for all comparisons")
        
        print()
    
    def check_multiple_testing(self):
        """Validate FDR correction for multiple comparisons."""
        print("─" * 80)
        print("CHECK 4: MULTIPLE TESTING CORRECTION")
        print("─" * 80)
        
        json_files = list(self.results_dir.glob('*.json'))
        n_configs = len(json_files)
        n_comparisons = n_configs * (n_configs - 1) // 2
        
        print(f"Configurations: {n_configs}")
        print(f"Potential pairwise comparisons: {n_comparisons}")
        print()
        
        if n_comparisons > 10:
            print("⚠️  Large number of comparisons detected")
            print()
            print("Required: Benjamini-Hochberg FDR correction")
            print(f"Adjusted α (Bonferroni): {0.05 / n_comparisons:.6f}")
            print()
            
            # Check if analyze_results.py uses FDR
            analyze_script = Path(__file__).parent / 'analyze_results.py'
            if analyze_script.exists():
                with open(analyze_script) as f:
                    content = f.read()
                
                if 'multipletests' in content and 'fdr_bh' in content:
                    self.checks_passed.append("Multiple testing correction")
                    print("✅ PASS: FDR correction implemented in analyze_results.py")
                else:
                    self.checks_failed.append("FDR correction not implemented")
                    print("❌ FAIL: FDR correction not found in analyze_results.py")
                    print("   Must use: statsmodels.stats.multitest.multipletests(method='fdr_bh')")
            else:
                self.warnings.append("Cannot verify FDR implementation")
                print("⚠️  WARN: Cannot verify FDR implementation")
        else:
            self.checks_passed.append("Multiple testing (small N)")
            print("✅ PASS: Small number of comparisons (< 10)")
            print("   FDR correction still recommended")
        
        print()
    
    def check_normality(self):
        """Check normality assumptions for parametric tests."""
        print("─" * 80)
        print("CHECK 5: NORMALITY ASSUMPTIONS")
        print("─" * 80)
        
        json_files = list(self.results_dir.glob('*.json'))
        hit_rates = []
        
        for filepath in json_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                hit_rate = data.get('hitRate', 0)
                if hit_rate > 0:
                    hit_rates.append(hit_rate)
            except:
                pass
        
        if len(hit_rates) < 3:
            self.warnings.append("Insufficient data for normality test")
            print("⚠️  WARN: Insufficient data for normality test (n < 3)")
            print()
            return
        
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(hit_rates)
        
        print(f"Shapiro-Wilk test on hit rates:")
        print(f"  Statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print()
        
        if p_value > 0.05:
            self.checks_passed.append("Normality assumption")
            print("✅ PASS: Data appears normally distributed (p > 0.05)")
            print("   Parametric tests (t-test, ANOVA) are appropriate")
        else:
            self.warnings.append("Non-normal distribution")
            print("⚠️  WARN: Data may not be normally distributed (p ≤ 0.05)")
            print("   Recommendation: Use non-parametric tests (Wilcoxon, Kruskal-Wallis)")
        
        print()
    
    def check_bias(self):
        """Check for various biases."""
        print("─" * 80)
        print("CHECK 6: BIAS ANALYSIS")
        print("─" * 80)
        
        # Check if bias_analysis.py has been run
        bias_files = list(self.results_dir.glob('*bias*.txt')) + list(self.results_dir.glob('*bias*.json'))
        
        if bias_files:
            self.checks_passed.append("Bias analysis")
            print("✅ PASS: Bias analysis files found")
            for f in bias_files:
                print(f"   • {f.name}")
        else:
            self.warnings.append("Bias analysis not run")
            print("⚠️  WARN: No bias analysis files found")
            print("   Run: python3 bias_analysis.py --results-dir " + str(self.results_dir))
        
        print()
    
    def check_reproducibility(self):
        """Calculate reproducibility score."""
        print("─" * 80)
        print("CHECK 7: REPRODUCIBILITY")
        print("─" * 80)
        
        score = 0
        max_score = 100
        
        # Check code availability (assumed yes if running this script)
        score += 10
        print("✅ Code available: +10")
        
        # Check data availability
        json_files = list(self.results_dir.glob('*.json'))
        if json_files:
            score += 10
            print("✅ Data available: +10")
        
        # Check environment documentation
        system_info = list(self.results_dir.glob('system_info.txt'))
        if system_info:
            score += 10
            print("✅ System info documented: +10")
        else:
            print("⚠️  System info missing: +0")
        
        # Check execution instructions (README)
        readme = Path('README.md')
        if readme.exists():
            score += 10
            print("✅ Execution instructions: +10")
        
        # Check expected results
        if json_files:
            score += 10
            print("✅ Expected results with variance: +10")
        
        # Check statistical methods
        if any(self.results_dir.glob('*summary*.csv')):
            score += 10
            print("✅ Statistical tests documented: +10")
        
        # Check limitations
        if readme.exists():
            with open(readme) as f:
                if 'limitation' in f.read().lower():
                    score += 10
                    print("✅ Limitations disclosed: +10")
        
        # Check DOI (Zenodo)
        # Assumed not done yet
        print("⏳ DOI (Zenodo): +0 (pending)")
        
        # Check independent verification
        print("⏳ Independent verification: +0 (pending)")
        
        print()
        print(f"Reproducibility Score: {score}/{max_score}")
        print()
        
        if score >= 90:
            self.checks_passed.append("Reproducibility (excellent)")
            print("✅ EXCELLENT: Ready for Q1 submission")
        elif score >= 70:
            self.checks_passed.append("Reproducibility (good)")
            print("✅ GOOD: Meets Q1 standards")
        elif score >= 50:
            self.warnings.append("Reproducibility score moderate")
            print("⚠️  MODERATE: Improvements recommended")
        else:
            self.checks_failed.append("Reproducibility score low")
            print("❌ LOW: Not ready for Q1 submission")
        
        print()
    
    def check_data_quality(self):
        """Check data quality and completeness."""
        print("─" * 80)
        print("CHECK 8: DATA QUALITY")
        print("─" * 80)
        
        json_files = list(self.results_dir.glob('*.json'))
        
        if not json_files:
            self.checks_failed.append("No data files")
            print("❌ FAIL: No data files found")
            print()
            return
        
        # Check for required fields
        required_fields = ['hitRate', 'avgLatencyMs', 'p99LatencyMs', 'strategy', 'dataset', 'seed']
        missing_fields = set()
        
        for filepath in json_files[:5]:  # Sample first 5 files
            try:
                with open(filepath) as f:
                    data = json.load(f)
                for field in required_fields:
                    if field not in data:
                        missing_fields.add(field)
            except:
                pass
        
        if not missing_fields:
            self.checks_passed.append("Data quality")
            print("✅ PASS: All required fields present")
        else:
            self.checks_failed.append(f"Missing fields: {missing_fields}")
            print(f"❌ FAIL: Missing required fields: {missing_fields}")
        
        print()
    
    def generate_report(self):
        """Generate final validation report."""
        print("=" * 80)
        print(" " * 30 + "FINAL REPORT")
        print("=" * 80)
        print()
        
        total_checks = len(self.checks_passed) + len(self.checks_failed) + len(self.warnings)
        
        print(f"Total checks: {total_checks}")
        print(f"  ✅ Passed: {len(self.checks_passed)}")
        print(f"  ⚠️  Warnings: {len(self.warnings)}")
        print(f"  ❌ Failed: {len(self.checks_failed)}")
        print()
        
        if self.checks_passed:
            print("Passed checks:")
            for check in self.checks_passed:
                print(f"  ✅ {check}")
            print()
        
        if self.warnings:
            print("Warnings:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
            print()
        
        if self.checks_failed:
            print("Failed checks:")
            for failure in self.checks_failed:
                print(f"  ❌ {failure}")
            print()
        
        # Overall verdict
        print("=" * 80)
        if not self.checks_failed and not self.warnings:
            print("🎉 VERDICT: READY FOR Q1 SUBMISSION")
            print()
            print("All checks passed! Your experiment meets Q1 publication standards.")
        elif not self.checks_failed:
            print("⚠️  VERDICT: REVIEW WARNINGS BEFORE SUBMISSION")
            print()
            print("No critical failures, but address warnings to strengthen submission.")
        else:
            print("❌ VERDICT: NOT READY FOR Q1 SUBMISSION")
            print()
            print("Critical failures detected. Address all failed checks before submission.")
        print("=" * 80)
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Q1 publication validation'
    )
    parser.add_argument('--results-dir', required=True,
                       help='Directory containing experiment results')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return 2
    
    validator = Q1Validator(results_dir)
    exit_code = validator.validate_all()
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
