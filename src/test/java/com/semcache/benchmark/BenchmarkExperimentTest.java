package com.semcache.benchmark;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

/**
 * Unit tests for the benchmark experiment modules.
 *
 * <p><b>Scientific Purpose:</b> These tests verify the correctness of metric
 * computations and data-processing logic independently of the live system
 * (Redis, ONNX models, LLM API). They serve as the "Methods" appendix test
 * that academic reviewers expect to be able to run in under 5 seconds on any
 * machine without external dependencies.
 *
 * <p>Test coverage targets (M.8 Level 2 compliance):
 * <ul>
 *   <li>{@link MetricsCollector} — percentile, hit rate, cost savings formulas</li>
 *   <li>{@link DatasetLoader} — sampling, splitting, edge case handling</li>
 *   <li>{@link ExperimentConfig} — serialization round-trip</li>
 * </ul>
 *
 * <p>Run with: {@code mvn test -Dtest=BenchmarkExperimentTest}
 */
@DisplayName("Benchmark Experiment Unit Tests")
class BenchmarkExperimentTest {

    // ─────────────────────────────────────────────────────────────────────────
    // Section 1: MetricsCollector — percentile calculations
    // ─────────────────────────────────────────────────────────────────────────

    @Nested
    @DisplayName("MetricsCollector — Percentile Calculations")
    class PercentileTests {

        /**
         * Verifies that {@code nearestRankPercentile} returns the correct value for
         * a known 10-element sorted sequence.
         *
         * <p>Reference: nearest-rank formula — percentile_p = sorted[⌈(p/100)×n⌉ − 1]
         */
        @Test
        @DisplayName("p50 of [1..10] should be 5")
        void testP50_standardSequence() {
            List<Long> sorted = Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L);
            assertThat(MetricsCollector.nearestRankPercentile(sorted, 50)).isEqualTo(5.0);
        }

        @Test
        @DisplayName("p99 of [1..10] should be 10 (tail latency)")
        void testP99_tailLatency() {
            List<Long> sorted = Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L);
            assertThat(MetricsCollector.nearestRankPercentile(sorted, 99)).isEqualTo(10.0);
        }

        @Test
        @DisplayName("Empty list returns 0 — no ArrayIndexOutOfBoundsException")
        void testPercentile_emptyList_returnsZero() {
            assertThat(MetricsCollector.nearestRankPercentile(Collections.emptyList(), 99))
                    .isEqualTo(0.0);
        }

        @Test
        @DisplayName("Single-element list — all percentiles return that element")
        void testPercentile_singleElement() {
            List<Long> single = List.of(42L);
            assertThat(MetricsCollector.nearestRankPercentile(single, 1)).isEqualTo(42.0);
            assertThat(MetricsCollector.nearestRankPercentile(single, 50)).isEqualTo(42.0);
            assertThat(MetricsCollector.nearestRankPercentile(single, 99)).isEqualTo(42.0);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 2: MetricsCollector — aggregate metric computation
    // ─────────────────────────────────────────────────────────────────────────

    @Nested
    @DisplayName("MetricsCollector — Aggregate Metrics")
    class AggregateMetricTests {

        private MetricsCollector collector;

        @BeforeEach
        void setUp() {
            collector = new MetricsCollector();
        }

        /**
         * Verifies the hit rate formula.
         *
         * <p>Given 3 hits and 2 misses, hit rate should be 3/5 × 100 = 60.0 %.
         */
        @Test
        @DisplayName("Hit rate: 3 hits + 2 misses = 60.0%")
        void testHitRate_threeHitsTwoMisses() {
            // Record 3 hits
            collector.record(true,  5L,  2L, 0L, 0.91, 0.0, 0.002);
            collector.record(true,  4L,  2L, 0L, 0.93, 0.0, 0.002);
            collector.record(true,  6L,  3L, 0L, 0.88, 0.0, 0.002);
            // Record 2 misses
            collector.record(false, 120L, 3L, 110L, 0.0, 0.002, 0.002);
            collector.record(false, 115L, 3L, 105L, 0.0, 0.002, 0.002);

            MetricsCollector.AggregateMetrics m = collector.compute();
            assertThat(m.hitRate()).isCloseTo(60.0, within(0.01));
            assertThat(m.cacheHits()).isEqualTo(3);
            assertThat(m.cacheMisses()).isEqualTo(2);
        }

        /**
         * Verifies that cost savings formula correctly reflects the fraction saved.
         *
         * <p>3 hits → $0.00 actual cost for those queries.
         * 2 misses → $0.002 each (actual = baseline for those).
         * Baseline for all 5 = 5 × $0.002 = $0.010.
         * Actual = 2 × $0.002 = $0.004 (hits saved $0.006).
         * Savings = 0.006 / 0.010 × 100 = 60.0%.
         */
        @Test
        @DisplayName("Cost savings reflect actual vs baseline LLM spend")
        void testCostSavings_correctFormula() {
            collector.record(true,   5L, 2L,   0L, 0.91, 0.000, 0.002);
            collector.record(true,   4L, 2L,   0L, 0.93, 0.000, 0.002);
            collector.record(true,   6L, 3L,   0L, 0.88, 0.000, 0.002);
            collector.record(false, 120L, 3L, 110L, 0.0,  0.002, 0.002);
            collector.record(false, 115L, 3L, 105L, 0.0,  0.002, 0.002);

            MetricsCollector.AggregateMetrics m = collector.compute();
            assertThat(m.costSavingsPercent()).isCloseTo(60.0, within(0.01));
        }

        @Test
        @DisplayName("compute() on empty collector throws IllegalStateException")
        void testCompute_emptyCollector_throws() {
            assertThatThrownBy(collector::compute)
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("zero observations");
        }

        @Test
        @DisplayName("reset() clears all observations")
        void testReset_clearsObservations() {
            collector.record(true, 10L, 2L, 0L, 0.9, 0.0, 0.001);
            collector.reset();
            assertThatThrownBy(collector::compute)
                    .isInstanceOf(IllegalStateException.class);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 3: DatasetLoader — sampling and splitting
    // ─────────────────────────────────────────────────────────────────────────

    @Nested
    @DisplayName("DatasetLoader — Sampling and Splitting")
    class DatasetLoaderTests {

        private DatasetLoader loader;

        @BeforeEach
        void setUp() {
            loader = new DatasetLoader(new com.fasterxml.jackson.databind.ObjectMapper());
        }

        /**
         * Sampling is deterministic: same seed always yields the same subset.
         */
        @Test
        @DisplayName("shuffleAndSample is deterministic for fixed seed")
        void testShuffleAndSample_deterministicWithFixedSeed() {
            List<DatasetLoader.DatasetRecord> fullDataset = buildFakeDataset(100);
            List<DatasetLoader.DatasetRecord> sample1 = loader.shuffleAndSample(fullDataset, 42L, 20);
            List<DatasetLoader.DatasetRecord> sample2 = loader.shuffleAndSample(fullDataset, 42L, 20);

            assertThat(sample1).hasSize(20);
            assertThat(sample1).containsExactlyElementsOf(sample2);
        }

        /**
         * Different seeds must (with overwhelming probability) produce different orderings.
         */
        @Test
        @DisplayName("Different seeds produce different orderings")
        void testShuffleAndSample_differentSeedsProduceDifferentOrder() {
            List<DatasetLoader.DatasetRecord> fullDataset = buildFakeDataset(100);
            List<DatasetLoader.DatasetRecord> s1 = loader.shuffleAndSample(fullDataset, 42L,  50);
            List<DatasetLoader.DatasetRecord> s2 = loader.shuffleAndSample(fullDataset, 999L, 50);
            assertThat(s1).isNotEqualTo(s2);
        }

        /**
         * Verifies that warmup and test sets are non-overlapping and sum to the full dataset.
         */
        @Test
        @DisplayName("split() produces non-overlapping, exhaustive partition")
        void testSplit_nonOverlapping() {
            List<DatasetLoader.DatasetRecord> dataset = buildFakeDataset(100);
            DatasetLoader.DatasetSplit split = loader.split(dataset, 0.30);

            assertThat(split.warmupSet().size() + split.testSet().size()).isEqualTo(100);
            assertThat(split.warmupSet()).doesNotContainAnyElementsOf(split.testSet());
        }

        @ParameterizedTest
        @ValueSource(doubles = {0.0, 1.0, -0.1, 1.1})
        @DisplayName("split() rejects invalid warmup ratios")
        void testSplit_invalidRatio_throws(double invalidRatio) {
            List<DatasetLoader.DatasetRecord> dataset = buildFakeDataset(10);
            assertThatThrownBy(() -> loader.split(dataset, invalidRatio))
                    .isInstanceOf(IllegalArgumentException.class);
        }

        private List<DatasetLoader.DatasetRecord> buildFakeDataset(int n) {
            List<DatasetLoader.DatasetRecord> ds = new java.util.ArrayList<>();
            for (int i = 0; i < n; i++) {
                ds.add(new DatasetLoader.DatasetRecord("query_" + i, "answer_" + i, "para_" + i));
            }
            return ds;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 4: ExperimentConfig — factory and self-description
    // ─────────────────────────────────────────────────────────────────────────

    @Nested
    @DisplayName("ExperimentConfig — Factory and Serialization")
    class ExperimentConfigTests {

        @Test
        @DisplayName("create() assigns a non-null UUID and ISO timestamp")
        void testCreate_populatesMetadata() {
            ExperimentConfig cfg = ExperimentConfig.createV2(
                    "msmarco", "data/msmarco_sample.jsonl",
                    "minilm", 0.85, "BIDIRECTIONAL", 0.30,
                    42L, 1000, true, "SEMANTIC", 5,
                    50000, 86400L, null, 0.0, 0.0, "results/test.json");

            assertThat(cfg.experimentId()).isNotBlank();
            assertThat(cfg.runTimestamp()).isNotBlank();
            assertThat(cfg.javaVersion()).isNotBlank();
        }

        @Test
        @DisplayName("toLogSummary() contains all key parameters")
        void testToLogSummary_containsAllParams() {
            ExperimentConfig cfg = ExperimentConfig.createV2(
                    "quora-pairs", "data/qqp.jsonl",
                    "mpnet", 0.90, "UNIDIRECTIONAL", 0.20,
                    123L, 500, false, "SEMANTIC", 3,
                    10000, 3600L, null, 0.0, 0.0, "results/qqp_result.json");

            String summary = cfg.toLogSummary();
            assertThat(summary).contains("quora-pairs");
            assertThat(summary).contains("mpnet");
            assertThat(summary).contains("0.90");
            assertThat(summary).contains("UNIDIRECTIONAL");
        }
    }
}
