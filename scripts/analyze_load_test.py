#!/usr/bin/env python3
"""
Load Test Results Analysis

Analyzes K6 load test results and generates publication-quality report.

Usage: python3 analyze_load_test.py results/stress_test_*/
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_k6_summary(summary_file):
    """Load K6 summary JSON."""
    with open(summary_file) as f:
        return json.load(f)


def analyze_sustained_load(results_dir):
    """Analyze sustained load test results."""
    summary_file = Path(results_dir) / 'sustained_load_summary.json'
    
    if not summary_file.exists():
        print(f"⚠️  Sustained load summary not found: {summary_file}")
        return None
    
    data = load_k6_summary(summary_file)
    metrics = data.get('metrics', {})
    
    # Extract key metrics
    http_reqs = metrics.get('http_reqs', {})
    http_req_duration = metrics.get('http_req_duration', {})
    http_req_failed = metrics.get('http_req_failed', {})
    
    total_requests = http_reqs.get('count', 0)
    duration_seconds = http_reqs.get('rate', 0)  # requests per second
    
    p50 = http_req_duration.get('values', {}).get('p(50)', 0)
    p95 = http_req_duration.get('values', {}).get('p(95)', 0)
    p99 = http_req_duration.get('values', {}).get('p(99)', 0)
    
    error_rate = http_req_failed.get('values', {}).get('rate', 0)
    
    return {
        'total_requests': total_requests,
        'rps': duration_seconds,
        'p50_ms': p50,
        'p95_ms': p95,
        'p99_ms': p99,
        'error_rate': error_rate * 100
    }


def analyze_spike_test(results_dir):
    """Analyze spike test results."""
    summary_file = Path(results_dir) / 'spike_test_summary.json'
    
    if not summary_file.exists():
        print(f"⚠️  Spike test summary not found: {summary_file}")
        return None
    
    data = load_k6_summary(summary_file)
    metrics = data.get('metrics', {})
    
    http_reqs = metrics.get('http_reqs', {})
    http_req_duration = metrics.get('http_req_duration', {})
    
    peak_rps = http_reqs.get('rate', 0)
    p99 = http_req_duration.get('values', {}).get('p(99)', 0)
    
    return {
        'peak_rps': peak_rps,
        'p99_ms': p99
    }


def generate_report(results_dir):
    """Generate comprehensive load test report."""
    print("=" * 70)
    print("PRODUCTION LOAD TEST ANALYSIS")
    print("=" * 70)
    print()
    
    # Analyze sustained load
    sustained = analyze_sustained_load(results_dir)
    
    if sustained:
        print("SUSTAINED LOAD TEST (30 minutes)")
        print("-" * 70)
        print(f"  Total requests: {sustained['total_requests']:,}")
        print(f"  Throughput: {sustained['rps']:,.0f} RPS")
        print(f"  P50 latency: {sustained['p50_ms']:.2f} ms")
        print(f"  P95 latency: {sustained['p95_ms']:.2f} ms")
        print(f"  P99 latency: {sustained['p99_ms']:.2f} ms")
        print(f"  Error rate: {sustained['error_rate']:.3f}%")
        print()
        
        # Validation
        if sustained['rps'] >= 10000:
            print("✅ Throughput target met (≥10K RPS)")
        else:
            print(f"⚠️  Throughput below target ({sustained['rps']:.0f} < 10,000 RPS)")
        
        if sustained['p99_ms'] <= 100:
            print("✅ P99 latency target met (≤100ms)")
        else:
            print(f"⚠️  P99 latency above target ({sustained['p99_ms']:.2f} > 100ms)")
        
        if sustained['error_rate'] <= 1.0:
            print("✅ Error rate target met (≤1%)")
        else:
            print(f"⚠️  Error rate above target ({sustained['error_rate']:.3f}% > 1%)")
        
        print()
    
    # Analyze spike test
    spike = analyze_spike_test(results_dir)
    
    if spike:
        print("SPIKE TEST")
        print("-" * 70)
        print(f"  Peak throughput: {spike['peak_rps']:,.0f} RPS")
        print(f"  P99 latency (peak): {spike['p99_ms']:.2f} ms")
        print()
        
        if spike['peak_rps'] >= 15000:
            print("✅ Peak throughput target met (≥15K RPS)")
        else:
            print(f"⚠️  Peak throughput below target ({spike['peak_rps']:.0f} < 15,000 RPS)")
        
        print()
    
    # Generate LaTeX table
    generate_latex_table(sustained, spike, results_dir)
    
    # Generate plots
    generate_plots(results_dir)
    
    print()
    print("=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print()


def generate_latex_table(sustained, spike, results_dir):
    """Generate LaTeX table for paper."""
    if not sustained:
        return
    
    latex = r"""\begin{table}[h]
\centering
\caption{Production Load Test Results}
\label{tab:load_test}
\begin{tabular}{lrr}
\toprule
Metric & Value & Target \\
\midrule
"""
    
    latex += f"Sustained RPS & {sustained['rps']:,.0f} & $\\geq$ 10,000 \\\\\n"
    
    if spike:
        latex += f"Peak RPS & {spike['peak_rps']:,.0f} & $\\geq$ 15,000 \\\\\n"
    
    latex += f"P50 Latency (ms) & {sustained['p50_ms']:.1f} & $<$ 50 \\\\\n"
    latex += f"P95 Latency (ms) & {sustained['p95_ms']:.1f} & $<$ 100 \\\\\n"
    latex += f"P99 Latency (ms) & {sustained['p99_ms']:.1f} & $<$ 100 \\\\\n"
    latex += f"Error Rate (\\%) & {sustained['error_rate']:.3f} & $<$ 1.0 \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    output_path = Path(results_dir) / 'load_test_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved: {output_path}")


def generate_plots(results_dir):
    """Generate publication-quality plots."""
    # This would parse the full JSON logs and create time-series plots
    # For now, just create a placeholder
    
    print("⚠️  Time-series plots require full K6 JSON logs")
    print("   Implement if needed for paper")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze K6 load test results'
    )
    parser.add_argument('results_dir',
                       help='Directory containing K6 results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Error: {results_dir} does not exist")
        return 1
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "LOAD TEST ANALYSIS FOR Q1" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    generate_report(results_dir)
    
    return 0


if __name__ == '__main__':
    exit(main())
