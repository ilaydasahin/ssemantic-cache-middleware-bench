package com.semcache.benchmark;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

class ZipfianDistributionTest {

    @Test
    @DisplayName("Should generate Zipfian distribution with correct skew")
    void testZipfianDistribution() {
        int numItems = 100;
        int numSamples = 10000;
        double skew = 1.0;
        
        List<Integer> indices = ZipfianDistribution.generateIndices(numItems, numSamples, skew, 42);
        
        assertThat(indices).hasSize(numSamples);
        
        // Count frequency of each index
        Map<Integer, Long> frequencies = indices.stream()
            .collect(Collectors.groupingBy(i -> i, Collectors.counting()));
        
        // Most frequent item should be index 0 (Zipfian property)
        long maxFreq = frequencies.values().stream().max(Long::compare).orElse(0L);
        assertThat(frequencies.get(0)).isEqualTo(maxFreq);
        
        // Verify power-law: frequency should decrease as rank increases
        long freq0 = frequencies.getOrDefault(0, 0L);
        long freq10 = frequencies.getOrDefault(10, 0L);
        long freq50 = frequencies.getOrDefault(50, 0L);
        
        assertThat(freq0).isGreaterThan(freq10);
        assertThat(freq10).isGreaterThan(freq50);
    }

    @Test
    @DisplayName("Should be deterministic with same seed")
    void testDeterministic() {
        List<Integer> indices1 = ZipfianDistribution.generateIndices(100, 1000, 1.0, 42);
        List<Integer> indices2 = ZipfianDistribution.generateIndices(100, 1000, 1.0, 42);
        
        assertThat(indices1).isEqualTo(indices2);
    }

    @Test
    @DisplayName("Should handle uniform distribution with zero skew")
    void testUniformDistribution() {
        List<Integer> indices = ZipfianDistribution.generateIndices(10, 1000, 0.01, 42);
        
        Map<Integer, Long> frequencies = indices.stream()
            .collect(Collectors.groupingBy(i -> i, Collectors.counting()));
        
        // With low skew, distribution should be more uniform
        long maxFreq = frequencies.values().stream().max(Long::compare).orElse(0L);
        long minFreq = frequencies.values().stream().min(Long::compare).orElse(0L);
        
        // Difference should be relatively small for uniform distribution
        assertThat(maxFreq - minFreq).isLessThan(200);
    }
}
