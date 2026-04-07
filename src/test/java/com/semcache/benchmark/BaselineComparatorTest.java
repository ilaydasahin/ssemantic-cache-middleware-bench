package com.semcache.benchmark;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for BaselineComparator
 */
class BaselineComparatorTest {

    private BaselineComparator comparator;

    @BeforeEach
    void setUp() {
        comparator = new BaselineComparator();
    }

    @Test
    void testComputeLatencyImprovement() {
        double improvement = comparator.computeLatencyImprovement(100, 200);
        assertThat(improvement).isEqualTo(50.0); // 50% improvement
    }

    @Test
    void testComputeLatencyImprovementNoBaseline() {
        double improvement = comparator.computeLatencyImprovement(100, 0);
        assertThat(improvement).isEqualTo(0.0);
    }

    @Test
    void testComputeCostSavings() {
        double savings = comparator.computeCostSavings(80.0, 0.01, 1000);
        assertThat(savings).isEqualTo(8.0); // 80% hit rate * 1000 queries * $0.01
    }

    @Test
    void testGenerateSimplifiedComparison() {
        Map<String, Object> report = comparator.generateSimplifiedComparison(
                1000, 800, // throughput
                50, 100,   // latency
                80, 60     // hit rate
        );

        assertThat(report).containsKeys(
                "throughputImprovement", "latencyImprovement", "hitRateGain");
        assertThat(report.get("throughputImprovement")).isEqualTo(25.0);
        assertThat(report.get("latencyImprovement")).isEqualTo(50.0);
        assertThat(report.get("hitRateGain")).isEqualTo(20.0);
    }
}
