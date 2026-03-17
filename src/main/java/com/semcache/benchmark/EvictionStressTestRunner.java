package com.semcache.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.semcache.service.EmbeddingService;
import com.semcache.service.SemanticCacheService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * M.6 Scalability Assessment: Heavy Churn Eviction Stress Test
 * Validates that the background batched eviction mechanism prevents latency
 * jitter
 * when the cache is flooded past its maximum capacity.
 */
@Component
public class EvictionStressTestRunner {

    private static final Logger log = LoggerFactory.getLogger(EvictionStressTestRunner.class);

    private final SemanticCacheService cacheService;
    private final EmbeddingService embeddingService;
    private final ObjectMapper objectMapper;

    public EvictionStressTestRunner(SemanticCacheService cacheService,
            EmbeddingService embeddingService,
            ObjectMapper objectMapper) {
        this.cacheService = cacheService;
        this.embeddingService = embeddingService;
        this.objectMapper = objectMapper;
    }

    public void runHeavyChurnTest(String datasetPath, DatasetLoader datasetLoader, String outputFile) throws Exception {
        log.info("=== Eviction Stress Test (Heavy Churn) ===");
        log.info("Goal: Measure p99 latency jitter when cache capacity is exceeded.");

        // Force a very small capacity for the test if possible, or assume it's set via
        // properties
        // The properties should have set maxCacheEntries to e.g., 1000

        List<DatasetLoader.DatasetRecord> dataset = datasetLoader.load(datasetPath);
        int testCount = Math.min(10000, dataset.size());

        log.info("Loaded {} records. Encoding vectors...", testCount);
        List<float[]> vectors = new ArrayList<>(testCount);
        for (int i = 0; i < testCount; i++) {
            vectors.add(embeddingService.encode(dataset.get(i).query()));
        }

        cacheService.clearCache();
        log.info("Commencing strict sequential injection of {} records to trigger eviction...", testCount);

        List<Long> insertLatenciesNs = new ArrayList<>(testCount);
        long sumNs    = 0L;
        long startTime = System.nanoTime();

        for (int i = 0; i < testCount; i++) {
            String q = dataset.get(i).query();
            float[] v = vectors.get(i);
            String a = dataset.get(i).answer();

            long startOp = System.nanoTime();
            cacheService.store(q, v, a);
            long elapsed = System.nanoTime() - startOp;

            insertLatenciesNs.add(elapsed);
            sumNs += elapsed;

            // Brief pause to allow background threads to interleave naturally
            if (i % 100 == 0) {
                Thread.sleep(1);
            }
        }
        long totalTimeMs = (System.nanoTime() - startTime) / 1_000_000;

        Collections.sort(insertLatenciesNs);
        double avgNs = (double) sumNs / testCount;
        long p50Ns  = (long) MetricsCollector.nearestRankPercentile(insertLatenciesNs, 50.0);
        long p99Ns  = (long) MetricsCollector.nearestRankPercentile(insertLatenciesNs, 99.0);
        long p999Ns = (long) MetricsCollector.nearestRankPercentile(insertLatenciesNs, 99.9);

        log.info("Eviction Stress Test Complete in {}ms", totalTimeMs);
        log.info("Insertion Latency (ms): Avg={}, p50={}, p99={}, p99.9={}",
                String.format(java.util.Locale.US, "%.3f", avgNs / 1_000_000.0),
                String.format(java.util.Locale.US, "%.3f", p50Ns / 1_000_000.0),
                String.format(java.util.Locale.US, "%.3f", p99Ns / 1_000_000.0),
                String.format(java.util.Locale.US, "%.3f", p999Ns / 1_000_000.0));

        StressResult result = new StressResult(
                testCount,
                totalTimeMs,
                avgNs / 1_000_000.0,
                p50Ns / 1_000_000.0,
                p99Ns / 1_000_000.0,
                p999Ns / 1_000_000.0);

        if (outputFile != null) {
            File f = new File(outputFile);
            File parent = f.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                log.warn("Could not create output directory: {}", parent);
            }
            objectMapper.writeValue(f, result);
            log.info("Saved stress test results to {}", outputFile);
        }
    }

    public record StressResult(
            int recordsInserted,
            long totalDurationMs,
            double avgLatencyMs,
            double p50LatencyMs,
            double p99LatencyMs,
            double p999LatencyMs) {
    }
}
