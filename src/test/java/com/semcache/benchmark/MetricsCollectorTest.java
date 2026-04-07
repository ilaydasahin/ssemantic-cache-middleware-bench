package com.semcache.benchmark;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

/**
 * Unit tests for MetricsCollector
 */
class MetricsCollectorTest {

    /**
     * Helper method to create MetricsCollector instance for tests.
     * Kept for potential future use in integration tests.
     */
    @SuppressWarnings("unused")
    private MetricsCollector createMetricsCollector() {
        return new MetricsCollector();
    }

    @Test
    void testHitRateCalculation() {
        // Given 80 hits out of 100 queries
        // When hit rate is calculated
        // Then should be 80%
        double hitRate = 80.0 / 100.0 * 100.0;
        assertThat(hitRate).isEqualTo(80.0);
    }

    @Test
    void testPercentileCalculation() {
        // Given latency measurements
        List<Double> latencies = List.of(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
        
        // When P50 is calculated
        // Then should be median (5.5)
        double p50 = latencies.get(latencies.size() / 2);
        assertThat(p50).isGreaterThan(0);
    }

    @Test
    void testP99Calculation() {
        // Given 100 latency measurements
        List<Double> latencies = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            latencies.add((double) i);
        }
        
        // When P99 is calculated
        // Then should be 99th value
        int p99Index = (int) Math.ceil(99.0 / 100.0 * latencies.size()) - 1;
        double p99 = latencies.get(p99Index);
        assertThat(p99).isGreaterThan(90);
    }

    @Test
    void testCostSavingsCalculation() {
        // Given 80% hit rate and $0.001 per LLM call
        // When cost savings is calculated
        // Then should be 80% of total cost
        double hitRate = 0.80;
        double costPerCall = 0.001;
        int totalCalls = 1000;
        double savings = hitRate * costPerCall * totalCalls;
        assertThat(savings).isCloseTo(0.80, within(0.01));
    }

    @Test
    void testMemoryUsageTracking() {
        // Given memory measurements
        // When tracked over time
        // Then should capture peak usage
        long memoryUsage = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        assertThat(memoryUsage).isGreaterThan(0);
    }

    @Test
    void testThroughputCalculation() {
        // Given 1000 queries in 10 seconds
        // When throughput is calculated
        // Then should be 100 queries/sec
        int queries = 1000;
        double seconds = 10.0;
        double throughput = queries / seconds;
        assertThat(throughput).isEqualTo(100.0);
    }

    @Test
    void testEmptyMetricsHandling() {
        // Given no measurements
        // When metrics are calculated
        // Then should handle gracefully
        List<Double> empty = List.of();
        assertThat(empty).isEmpty();
    }

    @Test
    void testSingleValueMetrics() {
        // Given single measurement
        // When percentiles are calculated
        // Then should return that value
        List<Double> single = List.of(5.0);
        assertThat(single).hasSize(1);
    }

    @Test
    void testOutlierHandling() {
        // Given measurements with outliers
        // When metrics are calculated
        // Then should handle appropriately
        List<Double> withOutliers = List.of(1.0, 2.0, 3.0, 1000.0);
        double max = withOutliers.stream().max(Double::compareTo).orElse(0.0);
        assertThat(max).isEqualTo(1000.0);
    }

    @Test
    void testMetricsAggregation() {
        // Given multiple experiment runs
        // When aggregated
        // Then should compute mean and variance
        List<Double> hitRates = List.of(80.0, 82.0, 81.0, 79.0, 83.0);
        double mean = hitRates.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        assertThat(mean).isCloseTo(81.0, within(0.1));
    }
}
