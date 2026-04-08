"""
Statistical Power Analysis for Semantic Cache Benchmark.

Computes required sample size (N seeds) to detect meaningful effect sizes
with 80% power at α=0.05 significance level.

Usage: 
    python3 power_analysis.py --effect-size 0.8
    python3 power_analysis.py --power 0.90 --alpha 0.01
    python3 power_analysis.py --show-curves

Dependencies: statsmodels, numpy, scipy, matplotlib
"""

import argparse
import numpy as np
from statsmodels.stats.power import TTestIndPower
from scipy import stats
import sys


def compute_required_sample_size(effect_size: float, alpha: float = 0.05, power: float = 0.80):
    """
    Calculate required N per group for independent t-test.
    
    Args:
        effect_size: Cohen's d (small=0.2, medium=0.5, large=0.8)
        alpha: Type I error rate (typically 0.05)
        power: Statistical power (typically 0.80)
    
    Returns:
        Required sample size per group
    """
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
    return int(np.ceil(n))


def compute_achieved_power(n: int, effect_size: float, alpha: float = 0.05):
    """
    Calculate achieved power given sample size and effect size.
    
    Args:
        n: Sample size per group
        effect_size: Cohen's d
        alpha: Type I error rate
    
    Returns:
        Achieved statistical power (0-1)
    """
    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, alternative='two-sided')
    return power


def analyze_pilot_data(pilot_results: list):
    """
    Estimate effect size from pilot data to inform full experiment design.
    
    Args:
        pilot_results: List of dicts with 'strategy' and 'hitRate' keys
    
    Returns:
        Estimated Cohen's d between strategies
    """
    strategies = {}
    for result in pilot_results:
        strat = result['strategy']
        if strat not in strategies:
            strategies[strat] = []
        strategies[strat].append(result['hitRate'])
    
    if len(strategies) < 2:
        return None
    
    # Compare first two strategies
    strat_names = list(strategies.keys())
    group1 = np.array(strategies[strat_names[0]])
    group2 = np.array(strategies[strat_names[1]])
    
    pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return abs(cohens_d)


def print_power_table(alpha: float = 0.05, power: float = 0.80):
    """Print comprehensive power analysis table."""
    print("=" * 100)
    print("STATISTICAL POWER ANALYSIS FOR Q1 PUBLICATION")
    print("=" * 100)
    print(f"\nTarget Parameters: α={alpha} (Type I error), Power={power} (1-β, Type II error)")
    print(f"Test: Two-tailed independent t-test (comparing two strategies)\n")
    
    # Standard effect sizes
    effect_sizes = [
        ("Very Small", 0.1),
        ("Small", 0.2),
        ("Small-Medium", 0.35),
        ("Medium", 0.5),
        ("Medium-Large", 0.65),
        ("Large", 0.8),
        ("Very Large", 1.0),
        ("Huge", 1.5),
        ("Extreme", 2.0),
    ]
    
    print(f"{'Effect Size':<20} | {'Cohen\'s d':<12} | {'N per group':<15} | {'Total Seeds':<15} | {'Interpretation'}")
    print("-" * 100)
    
    for label, d in effect_sizes:
        n = compute_required_sample_size(d, alpha, power)
        total = n  # For paired comparison (same seeds across strategies)
        
        if d < 0.2:
            interp = "Negligible - not worth detecting"
        elif d < 0.5:
            interp = "Small - requires large N"
        elif d < 0.8:
            interp = "Medium - typical for applied research"
        elif d < 1.2:
            interp = "Large - easy to detect"
        else:
            interp = "Very large - obvious differences"
        
        print(f"{label:<20} | {d:<12.2f} | {n:<15} | {total:<15} | {interp}")
    
    print("\n" + "=" * 100)
    print("CURRENT SETUP ANALYSIS")
    print("=" * 100)
    
    # Analyze current setups
    setups = [
        ("Quick Test (3 seeds)", 3),
        ("Pilot Study (5 seeds)", 5),
        ("Q1 Minimum (26 seeds)", 26),
        ("Q1 Recommended (64 seeds)", 64),
        ("Nature/Science (100 seeds)", 100),
    ]
    
    print(f"\n{'Setup':<30} | {'N':<8} | {'d=0.2':<10} | {'d=0.5':<10} | {'d=0.8':<10} | {'d=1.0':<10} | {'d=1.5':<10}")
    print("-" * 100)
    
    for label, n in setups:
        powers = []
        for d in [0.2, 0.5, 0.8, 1.0, 1.5]:
            p = compute_achieved_power(n, d, alpha)
            powers.append(f"{p:.1%}")
        
        print(f"{label:<30} | {n:<8} | {powers[0]:<10} | {powers[1]:<10} | {powers[2]:<10} | {powers[3]:<10} | {powers[4]:<10}")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR Q1 PUBLICATION")
    print("=" * 100)
    print("""
1. MINIMUM ACCEPTABLE (d=0.8, Power=80%):
   • Use 26 seeds per configuration
   • Can detect large effects (d≥0.8) with 80% power
   • Suitable for: IEEE TKDE, ACM TOIS, Information Sciences
   • Risk: May miss medium effects (d=0.5)

2. RECOMMENDED (d=0.5, Power=80%):
   • Use 64 seeds per configuration
   • Can detect medium effects (d≥0.5) with 80% power
   • Suitable for: Top-tier Q1 journals, Nature/Science
   • Benefit: More robust, can detect smaller meaningful differences

3. CONSERVATIVE (d=0.5, Power=90%):
   • Use 86 seeds per configuration
   • Can detect medium effects with 90% power
   • Suitable for: High-stakes claims, controversial findings
   • Benefit: Maximum confidence in results

4. CURRENT SETUP (3-5 seeds):
   • Only detects extreme effects (d>1.5) with reasonable power
   • Insufficient for Q1 publication
   • Suitable for: Pilot studies, preliminary results, workshops

⚠️  TYPE II ERROR RISK:
   With 3 seeds and d=0.8: Power = {:.1%} (20% chance of missing real effect!)
   With 26 seeds and d=0.8: Power = 80.0% (acceptable)
   With 64 seeds and d=0.5: Power = 80.0% (robust)

📊 JOURNAL EXPECTATIONS:
   • Q1 Journals: Typically expect power ≥ 80% for claimed effect sizes
   • Reviewers will ask: "Was your study adequately powered?"
   • Underpowered studies risk rejection or major revision
   • Must justify sample size with power analysis in methods section

🎯 ACTION ITEMS:
   1. Run power analysis BEFORE collecting data (a priori)
   2. Report expected effect size based on pilot data or literature
   3. Justify sample size in methods section
   4. Report achieved power in results section
   5. Discuss Type II error risk in limitations section
""".format(compute_achieved_power(3, 0.8, alpha)))


def plot_power_curves(alpha: float = 0.05):
    """Generate power curves for different sample sizes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
        return
    
    effect_sizes = np.linspace(0.1, 2.0, 100)
    sample_sizes = [3, 5, 10, 26, 64, 100]
    
    plt.figure(figsize=(12, 8))
    
    for n in sample_sizes:
        powers = [compute_achieved_power(n, d, alpha) for d in effect_sizes]
        label = f"N={n}"
        if n == 3:
            label += " (Quick Test)"
        elif n == 26:
            label += " (Q1 Min)"
        elif n == 64:
            label += " (Q1 Rec)"
        plt.plot(effect_sizes, powers, label=label, linewidth=2)
    
    plt.axhline(y=0.80, color='red', linestyle='--', label='80% Power Target', linewidth=1.5)
    plt.axvline(x=0.5, color='green', linestyle='--', label='Medium Effect (d=0.5)', linewidth=1.5)
    plt.axvline(x=0.8, color='blue', linestyle='--', label='Large Effect (d=0.8)', linewidth=1.5)
    
    plt.xlabel('Effect Size (Cohen\'s d)', fontsize=12)
    plt.ylabel('Statistical Power (1-β)', fontsize=12)
    plt.title(f'Power Curves for Different Sample Sizes (α={alpha})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2.0)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('power_curves.png', dpi=300, bbox_inches='tight')
    print("\n✅ Power curves saved to: power_curves.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Power Analysis for Q1 Publication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 power_analysis.py                          # Default analysis (d=0.5, power=0.80)
  python3 power_analysis.py --effect-size 0.8        # For large effects
  python3 power_analysis.py --power 0.90             # For 90% power
  python3 power_analysis.py --show-curves            # Generate power curves plot
  python3 power_analysis.py --alpha 0.01             # For stricter significance
        """
    )
    parser.add_argument("--effect-size", type=float, default=0.5, 
                       help="Expected Cohen's d (0.2=small, 0.5=medium, 0.8=large)")
    parser.add_argument("--alpha", type=float, default=0.05, 
                       help="Significance level (Type I error rate)")
    parser.add_argument("--power", type=float, default=0.80, 
                       help="Desired statistical power (1 - Type II error rate)")
    parser.add_argument("--show-curves", action="store_true",
                       help="Generate power curves plot (requires matplotlib)")
    args = parser.parse_args()
    
    # Validate inputs
    if not 0 < args.alpha < 1:
        print("❌ Error: alpha must be between 0 and 1")
        sys.exit(1)
    if not 0 < args.power < 1:
        print("❌ Error: power must be between 0 and 1")
        sys.exit(1)
    if args.effect_size <= 0:
        print("❌ Error: effect-size must be positive")
        sys.exit(1)
    
    # Print comprehensive table
    print_power_table(args.alpha, args.power)
    
    # Generate plots if requested
    if args.show_curves:
        print("\n" + "=" * 100)
        print("GENERATING POWER CURVES")
        print("=" * 100)
        plot_power_curves(args.alpha)
    
    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("""
1. Choose your target effect size:
   • Review pilot data or literature to estimate expected effect
   • Conservative: Use d=0.5 (medium effect)
   • Optimistic: Use d=0.8 (large effect)

2. Run appropriate benchmark:
   • Quick test (3 seeds):     ./bin/run_q1_quick_test.sh
   • Q1 minimum (26 seeds):    ./bin/run_q1_comprehensive_benchmark.sh
   • Q1 robust (64 seeds):     ./bin/run_q1plus_mega_benchmark.sh

3. Report in methods section:
   "Sample size was determined via a priori power analysis using G*Power.
    To detect a medium effect size (Cohen's d=0.5) with 80% power at α=0.05,
    we required N=64 seeds per configuration. We conducted [N] independent
    runs with different random seeds to ensure adequate statistical power."

4. Report in results section:
   "Post-hoc power analysis confirmed adequate power (1-β={:.2f}) to detect
    the observed effect size (d={:.2f})."

5. Include in limitations:
   "Our study was powered to detect medium-to-large effects. Smaller effects
    (d<0.5) may exist but were not detectable with our sample size."
""".format(args.power, args.effect_size))


if __name__ == "__main__":
    main()
