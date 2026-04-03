package com.semcache.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Compares experimental results against established baselines.
 * 
 * Q1 Publication Requirement: All novel systems must be compared against:
 * 1. No-cache baseline (100% LLM calls)
 * 2. Exact-match baseline (hash-based cache)
 * 3. State-of-the-art (if applicable)
 * 
 * This class implements the comparison methodology described in §5.2.
 */
@Component
public class BaselineComparator {
    
    private static final Logger log = LoggerFactory.getLogger(BaselineComparator.class);
    
    /**
     * Computes relative improvement over no-cache baseline.
     * 
     * Formula: improvement% = (baseline_metric - experimental_metric) / baseline_metric × 100
     * 
     * @param experimentalLatency p99 latency with semantic cache (ms)
     * @param baselineLatency p99 latency without cache (ms)
     * @return Percentage improvement (positive = better)
     */
    public double computeLatencyImprovement(double experimentalLatency, double baselineLatency) {
        if (baselineLatency == 0) {
            log.warn("Baseline latency is zero - cannot compute improvement");
            return 0.0;
        }
        return ((baselineLatency - experimentalLatency) / baselineLatency) * 100.0;
    }
    
    /**
     * Computes cost reduction vs. no-cache baseline.
     * 
     * @param hitRate Cache hit rate (0-100%)
     * @param avgLlmCostPerQuery Average LLM API cost per query ($)
     * @return Total cost savings ($)
     */
    public double computeCostSavings(double hitRate, double avgLlmCostPerQuery, int totalQueries) {
        double hitRateFraction = hitRate / 100.0;
        double queriesServedFromCache = totalQueries * hitRateFraction;
        return queriesServedFromCache * avgLlmCostPerQuery;
    }
    
    /**
     * Generates a baseline comparison report for publication.
     * 
     * @param experimentalMetrics Metrics from semantic cache run
     * @param exactMatchMetrics Metrics from exact-match baseline
     * @param noCacheMetrics Metrics from no-cache baseline
     * @return Formatted comparison table
     */
    public Map<String, Object> generateComparisonReport(
            MetricsCollector.AggregateMetrics experimentalMetrics,
            MetricsCollector.AggregateMetrics exactMatchMetrics,
            MetricsCollector.AggregateMetrics noCacheMetrics) {
        
        Map<String, Object> report = new LinkedHashMap<>();
        
        // Latency improvements
        double latencyVsNoCache = computeLatencyImprovement(
                experimentalMetrics.p99LatencyMs(), 
                noCacheMetrics.p99LatencyMs());
        double latencyVsExactMatch = computeLatencyImprovement(
                experimentalMetrics.p99LatencyMs(), 
                exactMatchMetrics.p99LatencyMs());
        
        report.put("latencyImprovementVsNoCache", round2(latencyVsNoCache));
        report.put("latencyImprovementVsExactMatch", round2(latencyVsExactMatch));
        
        // Hit rate improvements
        double hitRateGain = experimentalMetrics.hitRate() - exactMatchMetrics.hitRate();
        report.put("hitRateGainVsExactMatch", round2(hitRateGain));
        
        // Cost analysis
        report.put("costSavingsVsNoCache", round2(experimentalMetrics.costSavingsPercent()));
        
        // Statistical significance placeholder (computed by analyze_results.py)
        report.put("statisticallySignificant", null);
        report.put("pValue", null);
        report.put("cohensD", null);
        
        log.info("Baseline comparison: Latency improvement vs no-cache: {}%, vs exact-match: {}%",
                round2(latencyVsNoCache), round2(latencyVsExactMatch));
        
        return report;
    }
    
    private double round2(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
