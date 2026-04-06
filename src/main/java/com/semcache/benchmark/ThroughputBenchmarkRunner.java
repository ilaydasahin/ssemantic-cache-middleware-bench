package com.semcache.benchmark;

import com.semcache.service.EmbeddingService;
import com.semcache.service.SemanticCacheService;
import com.semcache.config.BenchmarkProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Throughput Benchmark Runner — Measures RPS and Latency vs Concurrent Users.
 *
 * NOT a CommandLineRunner itself — invoked by BenchmarkCommandLineRunner when
 * --benchmark.concurrent-users is set and --benchmark.current-dataset is
 * absent.
 */
@Component
@Profile("benchmark")
public class ThroughputBenchmarkRunner {

    private static final Logger log = LoggerFactory.getLogger(ThroughputBenchmarkRunner.class);

    private final SemanticCacheService cacheService;
    private final EmbeddingService embeddingService;
    private final ObjectMapper objectMapper;
    private final BenchmarkProperties properties;

    public ThroughputBenchmarkRunner(SemanticCacheService cacheService,
            EmbeddingService embeddingService,
            ObjectMapper objectMapper,
            BenchmarkProperties properties) {
        this.cacheService = cacheService;
        this.embeddingService = embeddingService;
        this.objectMapper = objectMapper;
        this.properties = properties;
    }

    /**
     * Run a throughput test for a specific concurrency level using real dataset
     * queries.
     * Called by BenchmarkCommandLineRunner in throughput mode.
     */
    public void runForUsers(int concurrentUsers,
            java.util.List<com.semcache.benchmark.DatasetLoader.DatasetRecord> dataset, String outputFile,
            long experimentSeed, String datasetName, String strategy)
            throws Exception {
        log.info("=== Throughput Test: {} concurrent users ===", concurrentUsers);

        // 1. Prepare fixed cache state (pool of entries) for stable measurement
        cacheService.clearCache();
        int poolSize = Math.min(5000, dataset.size());

        log.info("Warming up cache with {} encoded REAL records...", poolSize);
        long startWarmup = System.nanoTime();
        for (int i = 0; i < poolSize; i++) {
            String q = dataset.get(i).query();
            float[] vec = embeddingService.encode(q);
            cacheService.store(q, vec, dataset.get(i).answer());
        }
        log.info("Warmup complete in {}ms", (System.nanoTime() - startWarmup) / 1_000_000);

        // 2. Generate test queries using Zipfian distribution (M.6 External Validity)
        // Real-world systems exhibit skew: a few queries are extremely popular.
        int totalRequests = properties.getThroughputTotalRequests() != null 
                             ? properties.getThroughputTotalRequests() : 2000;
        double zipfExponent = properties.getThroughputZipfExponent() != null 
                             ? properties.getThroughputZipfExponent() : 1.1;

        log.info("Generating {} test requests following Zipfian distribution (s={}) over {} pool...",
                totalRequests, zipfExponent, poolSize);
        List<String> testQueries = generateZipfianTestQueries(dataset.subList(0, poolSize), totalRequests, zipfExponent, experimentSeed);

        // 3. Run load test
        ThroughputResult result = runLoadTest(concurrentUsers, testQueries);

        log.info("Throughput: users={}, rps={}, avgLatency={}ms, p99={}ms",
                concurrentUsers,
                String.format(java.util.Locale.US, "%.2f", result.rps()),
                String.format(java.util.Locale.US, "%.2f", result.avgLatencyMs()),
                result.p99Ms());

        // 4. Save result to JSON with enhanced metadata for Q1 analysis
        if (outputFile != null) {
            File f = new File(outputFile);
            File parent = f.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                log.warn("Could not create output directory: {}", parent);
            }
            
            // Create enhanced result with metadata for statistical analysis
            Map<String, Object> enhancedResult = new LinkedHashMap<>();
            enhancedResult.put("dataset", datasetName != null ? datasetName : "unknown");
            enhancedResult.put("seed", experimentSeed);
            enhancedResult.put("strategy", strategy != null ? strategy : properties.getStrategy());
            enhancedResult.put("embeddingModel", properties.getEmbeddingModel() != null ? properties.getEmbeddingModel() : "minilm");
            enhancedResult.put("concurrentUsers", result.concurrentUsers());
            enhancedResult.put("totalRequests", result.totalRequests());
            enhancedResult.put("throughput", result.rps());
            enhancedResult.put("avgLatencyMs", result.avgLatencyMs());
            enhancedResult.put("p99LatencyMs", result.p99Ms());
            enhancedResult.put("timestamp", System.currentTimeMillis());
            enhancedResult.put("threshold", properties.getSimilarityThreshold() != null ? properties.getSimilarityThreshold() : 0.90);
            
            // Estimate hit rate from cache statistics (if available)
            // This is a placeholder - actual hit rate should come from cache service
            enhancedResult.put("hitRate", 0.0); // Will be updated by cache service
            
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(f, enhancedResult);
            log.info("Throughput result saved to: {}", outputFile);
        }
    }

    private ThroughputResult runLoadTest(int concurrency, List<String> queries) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(concurrency);
        int totalRequests = queries.size();

        // Use primitive long array instead of ConcurrentLinkedQueue to avoid boxing
        long[] latenciesNs = new long[totalRequests];
        AtomicInteger nextQueryIndex = new AtomicInteger(0);

        long start = System.nanoTime();

        List<CompletableFuture<Void>> futures = new ArrayList<>();
        for (int i = 0; i < concurrency; i++) {
            futures.add(CompletableFuture.runAsync(() -> {
                while (true) {
                    int reqIndex = nextQueryIndex.getAndIncrement();
                    if (reqIndex >= totalRequests)
                        break;

                    String query = queries.get(reqIndex);

                    long actualStart = System.nanoTime();
                    cacheService.lookup(query);
                    long qEnd = System.nanoTime();

                    latenciesNs[reqIndex] = qEnd - actualStart;
                }
            }, executor));
        }

        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        long totalTimeNs = System.nanoTime() - start;
        executor.shutdown();

        double totalTimeSec = totalTimeNs / 1_000_000_000.0;
        double rps = totalRequests / totalTimeSec;

        // Sort for percentile calculation
        Arrays.sort(latenciesNs);

        long sumNs = 0L;
        for (long v : latenciesNs) sumNs += v;
        double avgLatencyNs = latenciesNs.length > 0 ? (double) sumNs / latenciesNs.length : 0.0;
        
        // Convert to List<Double> for percentile calculation
        List<Double> latenciesList = new ArrayList<>(latenciesNs.length);
        for (long v : latenciesNs) latenciesList.add((double) v);
        long p99Ns = (long) MetricsCollector.nearestRankPercentile(latenciesList, 99.0);

        return new ThroughputResult(concurrency, totalRequests, rps, avgLatencyNs / 1_000_000.0, p99Ns / 1_000_000);
    }

    private List<String> generateZipfianTestQueries(
            List<com.semcache.benchmark.DatasetLoader.DatasetRecord> pool, int numRequests, double s, long seed) {
        
        List<Integer> indices = ZipfianDistribution.generateIndices(pool.size(), numRequests, s, seed);
        
        List<String> queries = new ArrayList<>(numRequests);
        for (int idx : indices) {
            queries.add(pool.get(idx).query());
        }
        return queries;
    }

    public record ThroughputResult(

            int concurrentUsers,
            int totalRequests,
            double rps,
            double avgLatencyMs,
            long p99Ms) {
    }
}
