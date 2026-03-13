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
        java.util.List<String> warmupQueries = new java.util.ArrayList<>(poolSize);
        java.util.List<String> warmupAnswers = new java.util.ArrayList<>(poolSize);
        for (int i = 0; i < poolSize; i++) {
            warmupQueries.add(dataset.get(i).query());
            warmupAnswers.add(dataset.get(i).answer());
        }

        log.info("Warming up cache with {} encoded REAL records...", poolSize);
        long startWarmup = System.nanoTime();
        for (int i = 0; i < poolSize; i++) {
            String q = warmupQueries.get(i);
            float[] vec = embeddingService.encode(q);
            cacheService.store(q, vec, warmupAnswers.get(i));
        }
        log.info("Warmup complete in {}ms", (System.nanoTime() - startWarmup) / 1_000_000);

        // 2. Generate test queries using Zipfian distribution (M.6 External Validity)
        // Real-world systems exhibit skew: a few queries are extremely popular.
        int totalRequests = 2000;
        double zipfExponent = 1.1; // Typical value for web/search workloads

        log.info("Generating {} test requests following Zipfian distribution (s={}) over {} pool...",
                totalRequests, zipfExponent, poolSize);
        List<String> testQueries = generateZipfianTestQueries(warmupQueries, totalRequests, zipfExponent);

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
            if (f.getParentFile() != null)
                f.getParentFile().mkdirs();
            objectMapper.writeValue(f, result);
            log.info("Throughput result saved to: {}", outputFile);
        }
    }

    private ThroughputResult runLoadTest(int concurrency, List<String> queries) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(concurrency);
        int totalRequests = queries.size();

        Queue<Long> latenciesNs = new ConcurrentLinkedQueue<>();
        AtomicInteger completed = new AtomicInteger(0);

        long start = System.nanoTime();

        List<CompletableFuture<Void>> futures = new ArrayList<>();
        for (int i = 0; i < concurrency; i++) {
            futures.add(CompletableFuture.runAsync(() -> {
                while (true) {
                    int reqIndex = completed.getAndIncrement();
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

        double avgLatencyNs = latenciesSorted.stream().mapToLong(l -> l).average().orElse(0.0);
        long p99Ns = latenciesSorted.isEmpty() ? 0
                : latenciesSorted.get(Math.max(0, (int) (latenciesSorted.size() * 0.99) - 1));

        return new ThroughputResult(concurrency, totalRequests, rps, avgLatencyNs / 1_000_000.0, p99Ns / 1_000_000);
    }

    /**
     * Generates a list of queries sampled via Zipfian distribution from the actual
     * pool of queries.
     * Prevents uniform-random caching artifacts by modeling realistic power-law
     * traffic.
     */
    private List<String> generateZipfianTestQueries(List<String> pool, int numRequests, double s) {
        int poolSize = pool.size();
        List<String> queries = new ArrayList<>(numRequests);
        double c = 0;
        for (int i = 1; i <= poolSize; i++) {
            c += (1.0 / Math.pow(i, s));
        }
        c = 1.0 / c;

        double[] cdf = new double[poolSize];
        double sum = 0;
        for (int i = 1; i <= poolSize; i++) {
            sum += c * (1.0 / Math.pow(i, s));
            cdf[i - 1] = sum;
        }

        Random random = new Random(42); // deterministic
        for (int i = 0; i < numRequests; i++) {
            double p = random.nextDouble();
            int index = Arrays.binarySearch(cdf, p);
            if (index < 0)
                index = -(index + 1);
            index = Math.min(index, poolSize - 1);
            queries.add(pool.get(index));
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
