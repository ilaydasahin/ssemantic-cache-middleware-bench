package com.semcache.benchmark;

import com.semcache.service.EmbeddingService;
import com.semcache.service.SemanticCacheService;
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

    public ThroughputBenchmarkRunner(SemanticCacheService cacheService,
            EmbeddingService embeddingService,
            ObjectMapper objectMapper) {
        this.cacheService = cacheService;
        this.embeddingService = embeddingService;
        this.objectMapper = objectMapper;
    }

    /**
     * Run a throughput test for a specific concurrency level using real dataset
     * queries.
     * Called by BenchmarkCommandLineRunner in throughput mode.
     */
    public void runForUsers(int concurrentUsers,
            java.util.List<com.semcache.benchmark.DatasetLoader.DatasetRecord> dataset, String outputFile)
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
        int totalRequests = 2000;
        double zipfExponent = 1.1; // Typical value for web/search workloads

        log.info("Generating {} test requests following Zipfian distribution (s={}) over {} pool...",
                totalRequests, zipfExponent, poolSize);
        List<String> testQueries = generateZipfianTestQueries(dataset.subList(0, poolSize), totalRequests, zipfExponent);

        // 3. Run load test
        ThroughputResult result = runLoadTest(concurrentUsers, testQueries);

        log.info("Throughput: users={}, rps={}, avgLatency={}ms, p99={}ms",
                concurrentUsers,
                String.format(java.util.Locale.US, "%.2f", result.rps()),
                String.format(java.util.Locale.US, "%.2f", result.avgLatencyMs()),
                result.p99Ms());

        // 4. Save result to JSON if output file specified
        if (outputFile != null) {
            File f = new File(outputFile);
            File parent = f.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                log.warn("Could not create output directory: {}", parent);
            }
            objectMapper.writeValue(f, result);
            log.info("Throughput result saved to: {}", outputFile);
        }
    }

    private ThroughputResult runLoadTest(int concurrency, List<String> queries) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(concurrency);
        int totalRequests = queries.size();

        Queue<Long> latenciesNs = new ConcurrentLinkedQueue<>();
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

                    latenciesNs.add(qEnd - actualStart);
                }
            }, executor));
        }

        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        long totalTimeNs = System.nanoTime() - start;
        executor.shutdown();

        double totalTimeSec = totalTimeNs / 1_000_000_000.0;
        double rps = totalRequests / totalTimeSec;

        List<Long> latenciesSorted = new ArrayList<>(latenciesNs);
        Collections.sort(latenciesSorted);

        long sumNs = 0L;
        for (long v : latenciesSorted) sumNs += v;
        double avgLatencyNs = latenciesSorted.isEmpty() ? 0.0 : (double) sumNs / latenciesSorted.size();
        long p99Ns = (long) MetricsCollector.nearestRankPercentile(latenciesSorted, 99);

        return new ThroughputResult(concurrency, totalRequests, rps, avgLatencyNs / 1_000_000.0, p99Ns / 1_000_000);
    }

    /**
     * Generates a list of queries sampled via Zipfian distribution from the actual
     * pool of queries.
     * Prevents uniform-random caching artifacts by modeling realistic power-law
     * traffic.
     */
    private List<String> generateZipfianTestQueries(
            List<com.semcache.benchmark.DatasetLoader.DatasetRecord> pool, int numRequests, double s) {
        int poolSize = pool.size();

        // Compute power weights once; reuse for both normalization and CDF construction
        double[] weights = new double[poolSize];
        double total = 0;
        for (int i = 0; i < poolSize; i++) {
            weights[i] = 1.0 / Math.pow(i + 1, s);
            total += weights[i];
        }

        double[] cdf = new double[poolSize];
        double sum = 0;
        for (int i = 0; i < poolSize; i++) {
            sum += weights[i] / total;
            cdf[i] = sum;
        }

        List<String> queries = new ArrayList<>(numRequests);
        Random random = new Random(42); // deterministic
        for (int i = 0; i < numRequests; i++) {
            double p = random.nextDouble();
            int index = Arrays.binarySearch(cdf, p);
            if (index < 0)
                index = -(index + 1);
            index = Math.min(index, poolSize - 1);
            queries.add(pool.get(index).query());
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
