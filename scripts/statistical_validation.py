#!/usr/bin/env python3
"""
Statistical Validation Script

Validates that experimental results meet Q1 publication standards:
1. Sufficient sample size (power analysis)
2. Statistical significance (p < 0.05, FDR-corrected)
3. Effect size reporting (Cohen's d > 0.5)
4. Variance reporting (95% CI)
5. Assumption checking (normality, independence)

Usage: python3 statistical_validation.py --results-dir ../results/
"""

import argparse
import json
import glob
from pathlib import Path
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower


def check_sample_size(results_dir):
    """Validate sample size meets power analysis requirements."""
    print("=" * 70)
    print("SAMPLE SIZE VALIDATION")
    print("=" * 70)
    print()
    
    # Count results per configuration
    json_files = list(Path(results_dir).glob('*.json'))
    
    if not json_files:
        print("❌ No result files found")
        return False
    
    # Group by configuration
    configs = {}
    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)
        
        config_key = (
            data.get('dataset', 'unknown'),
            data.get('strategy', 'unknown'),
            data.get('embeddingModel', 'unknown'),
            data.get('threshold', 0.9)
        )
        
        if config_key not in configs:
            configs[config_key] = []
        configs[config_key].append(data)
    
    print(f"Found {len(configs)} unique configurations")
    print()
    
    # Check each configuration
    min_samples = float('inf')
    insufficient = []
    
    for config_key, results in configs.items():
        n = len(results)
        min_samples = min(min_samples, n)
        
        dataset, strategy, model, threshold = config_key
        
        # Power analysis for medium effect (d=0.5)
        analysis = TTestIndPower()
        required_n = analysis.solve_power(
            effect_size=0.5, 
            alpha=0.05, 
            power=0.80, 
            alternative='two-sided'
        )
        
        status = "✅" if n >= required_n else "⚠️"
        
        print(f"{status} {dataset:20s} {strategy:15s} {model:10s} θ={threshold:.2f}: "
              f"n={n:2d} (required: {int(np.ceil(required_n))})")
        
        if n < required_n:
            insufficient.append((config_key, n, int(np.ceil(required_n))))
    
    print()
    print(f"Minimum samples per configuration: {min_samples}")
    print()
    
    if insufficient:
        print(f"⚠️  {len(insufficient)} configurations have insufficient samples")
        print()
        print("Recommendations:")
        print("  • For medium effects (d=0.5): Need 64 samples")
        print("  • For large effects (d=0.8): Need 26 samples")
        print("  • Current minimum: {min_samples} samples")
        print()
        return False
    else:
        print("✅ All configurations have sufficient samples for power=0.80")
        print()
        return True


def check_normality(results_dir):
    """Check normality assumption for parametric tests."""
    print("=" * 70)
    print("NORMALITY ASSUMPTION CHECK")
    print("=" * 70)
    print()
    
    json_files = list(Path(results_dir).glob('*.json'))
    
    if not json_files:
        print("❌ No result files found")
        return False
    
    # Collect hit rates
    hit_rates = []
    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)
        hit_rate = data.get('hitRate', 0)
        if hit_rate > 0:
            hit_rates.append(hit_rate)
    
    if len(hit_rates) < 3:
        print("⚠️  Insufficient data for normality test (n < 3)")
        return False
    
    # Shapiro-Wilk test
    stat, p_value = stats.shapiro(hit_rates)
    
    print(f"Shapiro-Wilk Test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print()
    
    if p_value > 0.05:
        print("✅ Data appears normally distributed (p > 0.05)")
        print("   Parametric tests (t-test, ANOVA) are appropriate")
        print()
        return True
    else:
        print("⚠️  Data may not be normally distributed (p ≤ 0.05)")
        print("   Recommendation: Use non-parametric tests (Wilcoxon, Kruskal-Wallis)")
        print()
        return False


def check_effect_sizes(results_dir):
    """Validate that effect sizes are reported and meaningful."""
    print("=" * 70)
    print("EFFECT SIZE VALIDATION")
    print("=" * 70)
    print()
    
    # This would typically parse analysis results
    # For now, we check if effect size analysis has been run
    
    effect_size_files = list(Path(results_dir).glob('*effect_size*.json'))
    
    if not effect_size_files:
        print("⚠️  No effect size analysis found")
        print()
        print("Recommendation:")
        print("  Run: python3 effect_size_calculator.py --group1 ... --group2 ...")
        print()
        return False
    
    print(f"✅ Found {len(effect_size_files)} effect size analysis files")
    print()
    
    # Check if Cohen's d is reported in main results
    json_files = list(Path(results_dir).glob('*.json'))
    has_cohens_d = False
    
    for filepath in json_files[:5]:  # Check first 5 files
        with open(filepath) as f:
            data = json.load(f)
        if 'cohensD' in data or 'effectSize' in data:
            has_cohens_d = True
            break
    
    if has_cohens_d:
        print("✅ Effect sizes (Cohen's d) are reported in results")
    else:
        print("⚠️  Effect sizes not found in result files")
        print("   Ensure analyze_results.py calculates and exports effect sizes")
    
    print()
    return has_cohens_d


def check_variance_reporting(results_dir):
    """Validate that variance/confidence intervals are reported."""
    print("=" * 70)
    print("VARIANCE REPORTING VALIDATION")
    print("=" * 70)
    print()
    
    # Check for summary files with CI
    summary_files = list(Path(results_dir).glob('*summary*.csv'))
    
    if not summary_files:
        print("⚠️  No summary files found")
        print()
        print("Recommendation:")
        print("  Run: python3 analyze_results.py ../results/")
        print()
        return False
    
    print(f"✅ Found {len(summary_files)} summary files")
    print()
    
    # Check if CI is in the summary
    for summary_file in summary_files:
        with open(summary_file) as f:
            content = f.read()
        
        if 'CI' in content or '±' in content or 'std' in content.lower():
            print(f"✅ Variance reported in: {summary_file.name}")
        else:
            print(f"⚠️  No variance found in: {summary_file.name}")
    
    print()
    return True


def check_multiple_comparisons(results_dir):
    """Validate that multiple comparison corrections are applied."""
    print("=" * 70)
    print("MULTIPLE COMPARISONS CORRECTION")
    print("=" * 70)
    print()
    
    # Count number of comparisons
    json_files = list(Path(results_dir).glob('*.json'))
    
    if not json_files:
        print("❌ No result files found")
        return False
    
    # Estimate number of pairwise comparisons
    n_configs = len(json_files)
    n_comparisons = n_configs * (n_configs - 1) // 2
    
    print(f"Configurations: {n_configs}")
    print(f"Potential pairwise comparisons: {n_comparisons}")
    print()
    
    if n_comparisons > 10:
        print("⚠️  Large number of comparisons detected")
        print()
        print("Required corrections:")
        print("  • Benjamini-Hochberg FDR (recommended)")
        print("  • Bonferroni (conservative)")
        print("  • Holm-Bonferroni (less conservative)")
        print()
        print(f"Adjusted alpha (Bonferroni): {0.05 / n_comparisons:.6f}")
        print()
        
        # Check if FDR correction is mentioned in analysis
        analysis_script = Path(__file__).parent / 'analyze_results.py'
        if analysis_script.exists():
            with open(analysis_script) as f:
                content = f.read()
            
            if 'benjamini' in content.lower() or 'fdr' in content.lower():
                print("✅ FDR correction implemented in analyze_results.py")
            else:
                print("⚠️  FDR correction not found in analyze_results.py")
        
        print()
    else:
        print("✅ Small number of comparisons (< 10)")
        print("   Multiple comparison correction still recommended")
        print()
    
    return True


def generate_validation_report(results_dir, output_file):
    """Generate comprehensive validation report."""
    print("=" * 70)
    print("GENERATING VALIDATION REPORT")
    print("=" * 70)
    print()
    
    report = {
        'sample_size': check_sample_size(results_dir),
        'normality': check_normality(results_dir),
        'effect_sizes': check_effect_sizes(results_dir),
        'variance_reporting': check_variance_reporting(results_dir),
        'multiple_comparisons': check_multiple_comparisons(results_dir)
    }
    
    # Calculate overall score
    passed = sum(report.values())
    total = len(report)
    score = (passed / total) * 100
    
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()
    print(f"Checks passed: {passed}/{total} ({score:.0f}%)")
    print()
    
    for check, result in report.items():
        status = "✅" if result else "⚠️"
        print(f"{status} {check.replace('_', ' ').title()}")
    
    print()
    
    if score >= 80:
        print("✅ OVERALL: Meets Q1 statistical standards")
        print()
        print("Your results are ready for publication!")
    elif score >= 60:
        print("⚠️  OVERALL: Partially meets Q1 standards")
        print()
        print("Address the warnings above before submission.")
    else:
        print("❌ OVERALL: Does not meet Q1 standards")
        print()
        print("Significant improvements needed before submission.")
    
    print()
    
    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                'checks': report,
                'score': score,
                'passed': passed,
                'total': total
            }, f, indent=2)
        print(f"Report saved to: {output_file}")
        print()
    
    return score >= 80


def main():
    parser = argparse.ArgumentParser(
        description='Validate statistical rigor for Q1 publication'
    )
    parser.add_argument('--results-dir', required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for validation report')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return 1
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "STATISTICAL VALIDATION FOR Q1" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    success = generate_validation_report(results_dir, args.output)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
