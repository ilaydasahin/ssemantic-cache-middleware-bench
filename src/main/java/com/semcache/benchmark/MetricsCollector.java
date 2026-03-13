package com.semcache.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Computes and aggregates all quantitative metrics reported in the experimental
 * evaluation section of the paper.
 *
 * <p><b>Scientific Purpose:</b> This class implements the metric computations
 * described in §5 (Experimental Evaluation), ensuring they are deterministic,
 * independently testable, and free from data-leakage between train and test sets.
 *
 * <p><b>Metrics produced:</b>
 * <ul>
 *   <li><b>Hit Rate</b> — fraction of test queries served from cache (0–100 %)</li>
 *   <li><b>Latency Percentiles</b> — p50, p95, p99 of end-to-end query latency (ms)</li>
 *   <li><b>Embedding Latency</b> — mean ONNX inference time per query (ms)</li>
 *   <li><b>LLM Latency</b> — mean LLM API latency for cache-miss queries (ms)</li>
 *   <li><b>Cost Savings</b> — percentage reduction in LLM API cost vs no-cache baseline</li>
 *   <li><b>Memory Estimate</b> — JVM heap used at experiment end (MB)</li>
 * </ul>
 *
 * <p><b>Statistical validity:</b> All averages are computed over the full test
 * set.  Percentile calculations follow the nearest-rank method.  For cross-run
 * comparisons, use the Wilcoxon signed-rank test as implemented in
 * {@code analyze_results.py}.
 */
@Component
public class MetricsCollector {

    private static final Logger log = LoggerFactory.getLogger(MetricsCollector.class);

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Accumulates a per-query observation into the growing metric set.
     *
     * <p>Call this once per processed test query, immediately after determining
     * cache hit/miss and measuring latencies.
     *
     * @param hit               {@code true} if the cache returned a response
     * @param totalLatencyMs    End-to-end wall-clock latency for this query (ms)
     * @param embeddingLatencyMs Time spent on ONNX embedding inference (ms)
     * @param llmLatencyMs      Time spent waiting for the LLM API on miss (0 on hit)
     * @param similarityScore   Cosine similarity against the best cache candidate
     *                          (0.0 on exact-match hit or miss)
     * @param actualCost        Estimated LLM API cost for this query (0.0 on hit)
     * @param baselineCost      Estimated cost if no cache were in use
     */
    public void record(boolean hit,
                       long    totalLatencyMs,
                       long    embeddingLatencyMs,
                       long    llmLatencyMs,
                       double  similarityScore,
                       double  actualCost,
                       double  baselineCost) {

        observations.add(new Observation(
                hit, totalLatencyMs, embeddingLatencyMs, llmLatencyMs,
                similarityScore, actualCost, baselineCost));
    }

    /**
     * Computes the complete {@link AggregateMetrics} from all recorded observations.
     *
     * <p>This method is idempotent; it may be called multiple times without
     * changing the underlying observation list.
     *
     * @return Fully populated aggregate metrics record
     * @throws IllegalStateException if called before any observations are recorded
     */
    public AggregateMetrics compute() {
        if (observations.isEmpty()) {
            throw new IllegalStateException(
                    "MetricsCollector.compute() called with zero observations");
        }

        int    totalQueries  = observations.size();
        int    cacheHits     = (int) observations.stream().filter(Observation::hit).count();
        int    cacheMisses   = totalQueries - cacheHits;

        // Hit rate: fraction of queries answered from cache (expressed as 0–100 %)
        double hitRate = (double) cacheHits / totalQueries * 100.0;

        // Latency percentiles over ALL queries (hit + miss), sorted ascending
        List<Long> sortedLatencies = observations.stream()
                .map(Observation::totalLatencyMs)
                .sorted()
                .collect(Collectors.toList());

        double p50 = nearestRankPercentile(sortedLatencies, 50);
        double p95 = nearestRankPercentile(sortedLatencies, 95);
        double p99 = nearestRankPercentile(sortedLatencies, 99);

        // Mean embedding latency — averaged across all queries (hit + miss)
        double avgEmbeddingLatencyMs = observations.stream()
                .mapToLong(Observation::embeddingLatencyMs)
                .average()
                .orElse(0.0);

        // Mean LLM latency — averaged over MISS queries only
        // (hit queries incur zero LLM cost; including them would dilute the metric)
        OptionalDouble avgLlmOpt = observations.stream()
                .filter(o -> !o.hit())
                .mapToLong(Observation::llmLatencyMs)
                .average();
        double avgLlmLatencyMs = avgLlmOpt.isPresent() ? avgLlmOpt.getAsDouble() : 0.0;

        // Cost savings: percentage reduction in API spend vs. all-miss baseline
        // Formula: savings% = (baseline_total - actual_total) / baseline_total × 100
        double totalActualCost   = observations.stream().mapToDouble(Observation::actualCost).sum();
        double totalBaselineCost = observations.stream().mapToDouble(Observation::baselineCost).sum();
        double costSavingsPercent = totalBaselineCost > 0
                ? (totalBaselineCost - totalActualCost) / totalBaselineCost * 100.0
                : 0.0;

        // Runtime JVM heap usage at metric-collection time
        double memoryUsageMb = measureJvmHeapUsageMb();

        AggregateMetrics result = new AggregateMetrics(
                totalQueries, cacheHits, cacheMisses, hitRate,
                p50, p95, p99,
                avgEmbeddingLatencyMs, avgLlmLatencyMs,
                costSavingsPercent, memoryUsageMb);

        log.info("Metrics computed: hitRate={:.1f}%, p50={}ms, p99={}ms, " +
                 "costSavings={:.1f}%, memory={:.1f}MB",
                result.hitRate(), result.p50LatencyMs(), result.p99LatencyMs(),
                result.costSavingsPercent(), result.memoryUsageMb());

        return result;
    }

    /** Resets all recorded observations. Call before each independent experimental run. */
    public void reset() {
        observations.clear();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Nearest-rank percentile method.
     *
     * <p>Given a sorted list {@code L} of {@code n} values, the p-th percentile
     * (p ∈ [1, 100]) is {@code L[⌈p/100 × n⌉ − 1]}.  Edge cases (empty list,
     * index overflow) are guarded.
     *
     * @param sortedValues Ascending-sorted latency measurements
     * @param percentile   Target percentile in [1, 100]
     * @return The corresponding percentile value, or 0 if the list is empty
     */
    static double nearestRankPercentile(List<Long> sortedValues, int percentile) {
        if (sortedValues.isEmpty()) return 0.0;
        int n     = sortedValues.size();
        int index = (int) Math.ceil(percentile / 100.0 * n) - 1;
        int clamped = Math.max(0, Math.min(index, n - 1));
        return sortedValues.get(clamped);
    }

    /**
     * Measures the currently committed JVM heap usage.
     *
     * <p>{@code totalMemory()} returns the amount of memory the JVM has
     * committed (not necessarily all in use); {@code freeMemory()} is the
     * portion of committed memory that is currently unused.  Their difference
     * approximates the live memory footprint of the experiment.
     *
     * @return Approximate JVM heap usage in megabytes
     */
    static double measureJvmHeapUsageMb() {
        Runtime rt        = Runtime.getRuntime();
        long    usedBytes = rt.totalMemory() - rt.freeMemory();
        return usedBytes / (1024.0 * 1024.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal state
    // ─────────────────────────────────────────────────────────────────────────

    private final List<Observation> observations = new ArrayList<>();

    // ─────────────────────────────────────────────────────────────────────────
    // Value types
    // ─────────────────────────────────────────────────────────────────────────

    /** Raw per-query measurement captured during a single experimental trial. */
    record Observation(
            boolean hit,
            long    totalLatencyMs,
            long    embeddingLatencyMs,
            long    llmLatencyMs,
            double  similarityScore,
            double  actualCost,
            double  baselineCost) {}

    /**
     * Complete aggregate statistics for one experimental configuration.
     *
     * <p>All fields correspond directly to columns in Table 4 of the paper.
     *
     * @param totalQueries         Number of queries in the test set
     * @param cacheHits            Count of queries served from cache
     * @param cacheMisses          Count of queries requiring LLM fallback
     * @param hitRate              Cache hit rate (0–100 %)
     * @param p50LatencyMs         Median end-to-end latency (ms)
     * @param p95LatencyMs         95th-percentile end-to-end latency (ms)
     * @param p99LatencyMs         99th-percentile / tail latency (ms)
     * @param avgEmbeddingLatencyMs Mean ONNX embedding inference time (ms)
     * @param avgLlmLatencyMs      Mean LLM API call latency, miss queries only (ms)
     * @param costSavingsPercent   API cost reduction vs. no-cache baseline (%)
     * @param memoryUsageMb        JVM heap at experiment end (MB)
     */
    public record AggregateMetrics(
            int    totalQueries,
            int    cacheHits,
            int    cacheMisses,
            double hitRate,
            double p50LatencyMs,
            double p95LatencyMs,
            double p99LatencyMs,
            double avgEmbeddingLatencyMs,
            double avgLlmLatencyMs,
            double costSavingsPercent,
            double memoryUsageMb) {}
}
