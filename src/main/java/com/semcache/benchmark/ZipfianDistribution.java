package com.semcache.benchmark;

import java.util.*;

/**
 * Generates query index sequences following a Zipfian power-law distribution.
 * Shared by BenchmarkRunner and ThroughputBenchmarkRunner for realistic load testing.
 */
public final class ZipfianDistribution {

    private ZipfianDistribution() {} // utility class

    public static List<Integer> generateIndices(int poolSize, int count, double skew, long seed) {
        if (skew < 0.01) {
            List<Integer> indices = new ArrayList<>(count);
            for (int i = 0; i < count; i++) indices.add(i % poolSize);
            return indices;
        }

        double[] weights = new double[poolSize];
        double total = 0.0;
        for (int i = 0; i < poolSize; i++) {
            weights[i] = 1.0 / Math.pow(i + 1, skew);
            total += weights[i];
        }

        double[] cdf = new double[poolSize];
        double sum = 0.0;
        for (int i = 0; i < poolSize; i++) {
            sum += weights[i] / total;
            cdf[i] = sum;
        }

        List<Integer> indices = new ArrayList<>(count);
        Random rand = new Random(seed);
        for (int i = 0; i < count; i++) {
            double p = rand.nextDouble();
            int idx = Arrays.binarySearch(cdf, p);
            if (idx < 0) idx = -(idx + 1);
            indices.add(Math.min(idx, poolSize - 1));
        }
        return indices;
    }
}
