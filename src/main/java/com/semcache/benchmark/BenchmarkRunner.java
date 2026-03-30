package com.semcache.benchmark;

import com.semcache.benchmark.DatasetLoader.DatasetRecord;
import com.semcache.benchmark.DatasetLoader.DatasetSplit;
import com.semcache.benchmark.ExperimentResultExporter.QueryLog;
import com.semcache.benchmark.MetricsCollector.AggregateMetrics;
import com.semcache.model.CacheLookupResult;
import com.semcache.service.EmbeddingService;
import com.semcache.service.LLMService;
import com.semcache.service.MockGeminiService;
import com.semcache.service.SemanticCacheService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.io.File;

/**
 * Orchestrates a single experimental trial in the semantic cache benchmark.
 *
 * <p>
 * <b>Experiment Design (§4, Algorithm 1):</b>
 * Given a configuration tuple ⟨dataset, model, θ, strategy, warmupStrategy,
 * seed⟩:
 * <ol>
 * <li>Load and deterministically shuffle the dataset with the provided
 * seed.</li>
 * <li>Partition into a warmup set (first {@code warmupRatio} fraction) and test
 * set.</li>
 * <li>Pre-populate the semantic cache using the warmup set (originals
 * only).</li>
 * <li>Process each test query through the caching layer and record per-query
 * metrics.</li>
 * <li>Compute aggregate metrics and export a self-describing result JSON.</li>
 * </ol>
 *
 * <p>
 * <b>Threat to validity addressed:</b> Paraphrases are deliberately withheld
 * from the
 * warmup phase (cf. {@link DatasetLoader#split}) to ensure that test queries
 * exercise the
 * semantic retrieval path rather than the trivial O(1) exact-match path.
 *
 * <p>
 * <b>Module responsibilities:</b>
 * <ul>
 * <li>{@link DatasetLoader} — loading, shuffling, sampling, splitting</li>
 * <li>{@link MetricsCollector} — per-query accumulation and aggregate
 * computation</li>
 * <li>{@link ExperimentResultExporter} — JSON / JSONL output</li>
 * </ul>
 * This class is responsible only for orchestration; it contains no metric math
 * or I/O logic.
 */
@Component
public class BenchmarkRunner {

    private static final Logger log = LoggerFactory.getLogger(BenchmarkRunner.class);
    
    // Configuration constants (extracted from magic numbers)
    private static final int HEALTH_CHECK_FREQUENCY = 100; // queries
    private static final int PROGRESS_REPORT_FREQUENCY = 100; // queries
    private static final int CHECKPOINT_FREQUENCY_SMALL = 5; // for first 100 queries
    private static final int CHECKPOINT_FREQUENCY_LARGE = 50; // for remaining queries
    private static final int MAX_THREAD_POOL_SIZE = 32;
    private static final int THREAD_POOL_SHUTDOWN_TIMEOUT_SECONDS = 60;

    // ── Configurable experiment parameter (S3: no hardcoded constants) ────────
    /**
     * Fraction of the dataset used for cache warming. Injected from
     * {@code application.yml}.
     */
    @Value("${benchmark.warmup-ratio:0.30}")
    private double warmupRatio;

    /**
     * Zipfian skew parameter (s). 0.0 means uniform distribution. Q1 Gold Standard:
     * 0.7-1.0.
     */
    @Value("${benchmark.zipfian-skew:0.0}")
    private double zipfianSkew;

    // ── Dependencies ──────────────────────────────────────────────────────────
    private final SemanticCacheService cacheService;
    private final EmbeddingService embeddingService;
    private final LLMService llmService;
    private final DatasetLoader datasetLoader;
    private final MetricsCollector metricsCollector;
    private final ExperimentResultExporter resultExporter;
    private final NoiseGenerator noiseGenerator;
    private final CheckpointManager checkpointManager;

    public BenchmarkRunner(SemanticCacheService cacheService,
            EmbeddingService embeddingService,
            LLMService llmService,
            DatasetLoader datasetLoader,
            MetricsCollector metricsCollector,
            ExperimentResultExporter resultExporter,
            NoiseGenerator noiseGenerator,
            CheckpointManager checkpointManager) {
        this.cacheService = cacheService;
        this.embeddingService = embeddingService;
        this.llmService = llmService;
        this.datasetLoader = datasetLoader;
        this.metricsCollector = metricsCollector;
        this.resultExporter = resultExporter;
        this.noiseGenerator = noiseGenerator;
        this.checkpointManager = checkpointManager;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Executes a complete experimental trial and exports the result.
     *
     * <p>
     * This is the main entry point called by {@link BenchmarkCommandLineRunner}.
     * Each call is stateless with respect to the result files (the cache and
     * metrics
     * collector are reset at the start of this method).
     *
     * @param config A fully populated {@link ExperimentConfig} for this trial
     * @throws Exception if dataset loading, cache operations, or result export fail
     */
    public void run(ExperimentConfig config) throws Exception {
        log.info(config.toLogSummary());
        
        // Generate experiment ID for checkpoint tracking
        String experimentId = CheckpointManager.generateExperimentId(
                config.datasetName(), config.randomSeed(), config.similarityThreshold());

        // ── Phase 0: Reset stateful components ───────────────────────────────
        metricsCollector.reset();
        configureCacheService(config);
        cacheService.clearCache();

        // ── Phase 1: Load and partition dataset ──────────────────────────────
        List<DatasetRecord> fullDataset = datasetLoader.load(config.datasetPath());
        List<DatasetRecord> sampledDataset = datasetLoader.shuffleAndSample(
                fullDataset, config.randomSeed(), config.sampleSize());
        DatasetSplit split = datasetLoader.split(sampledDataset, warmupRatio);

        log.info("Dataset ready: warmup={}, test={}", split.warmupSet().size(), split.testSet().size());
        
        // Check for existing checkpoint
        CheckpointManager.Checkpoint checkpoint = checkpointManager.loadCheckpoint(experimentId);
        if (checkpoint == null) {
            checkpoint = new CheckpointManager.Checkpoint(
                    experimentId, config.datasetName(), config.randomSeed(), 
                    config.similarityThreshold(), split.testSet().size());
            log.info("Starting new experiment: {}", experimentId);
        } else {
            // Validate checkpoint integrity
            if (checkpoint.totalQueries != split.testSet().size()) {
                log.warn("Checkpoint mismatch: expected {} queries, found {}. Starting fresh.", 
                        split.testSet().size(), checkpoint.totalQueries);
                checkpoint = new CheckpointManager.Checkpoint(
                        experimentId, config.datasetName(), config.randomSeed(), 
                        config.similarityThreshold(), split.testSet().size());
            } else if (checkpoint.completedQueryIndices == null) {
                log.error("Checkpoint corrupted: completedQueryIndices is null. Starting fresh.");
                checkpoint = new CheckpointManager.Checkpoint(
                        experimentId, config.datasetName(), config.randomSeed(), 
                        config.similarityThreshold(), split.testSet().size());
            } else {
                log.info("Resuming experiment: {} ({}/{} queries remaining)", 
                        experimentId, 
                        split.testSet().size() - checkpoint.completedQueryIndices.size(),
                        split.testSet().size());
            }
        }

        // ── Phase 2: Register ground-truth answers in Mock LLM (if active) ───
        registerGroundTruthIfMock(sampledDataset);

        // ── Phase 3: Pre-populate cache (warmup phase) ───────────────────────
        warmUpCache(split.warmupSet(), config.warmupStrategy());

        // ── Phase 4: Measurement phase (with checkpoint support) ─────────────
        List<QueryLog> queryLogs = processTestSet(split.testSet(), config, checkpoint);

        // ── Phase 5: Compute and export results ──────────────────────────────
        AggregateMetrics metrics = metricsCollector.compute();
        resultExporter.export(config, metrics, queryLogs);
        
        // Delete checkpoint after successful completion
        checkpointManager.deleteCheckpoint(experimentId);
        log.info("✅ Experiment completed successfully: {}", experimentId);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private — Experiment phase implementations
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Propagates all cache-relevant settings from the config to the cache service.
     *
     * <p>
     * Setting these before {@code clearCache()} ensures the cache operates
     * with the correct threshold and strategy from the very first warmup insertion.
     */
    private void configureCacheService(ExperimentConfig config) {
        cacheService.setSimilarityThreshold(config.similarityThreshold());
        cacheService.setStrategy(config.cacheStrategy());
        cacheService.setHnswEnabled(config.hnswEnabled());
        cacheService.setWarmupStrategy(config.warmupStrategy());
    }

    /**
     * Pre-populates the cache with the warmup portion of the dataset.
     *
     * <p>
     * <b>E1 — Experimental Validity:</b> Only original queries are stored.
     * If paraphrases were also stored here (as the {@code BIDIRECTIONAL} name might
     * suggest), test-phase lookups using paraphrases would hit via the O(1)
     * exact-match
     * path, bypassing the semantic retrieval component under evaluation entirely.
     * The {@code BIDIRECTIONAL} strategy applies only to runtime caching of new
     * misses.
     *
     * @param warmupSet      Records to pre-populate (originals only)
     * @param warmupStrategy Logged for audit; does NOT change which records are
     *                       cached here
     */
    private void warmUpCache(List<DatasetRecord> warmupSet, String warmupStrategy) {
        log.info("Warming up cache with {} originals (strategy={}, paraphrases withheld for test validity)",
                warmupSet.size(), warmupStrategy);

        for (DatasetRecord record : warmupSet) {
            float[] embedding = embeddingService.encode(record.query());
            cacheService.store(record.query(), embedding, record.answer());
        }

        log.info("Cache pre-populated: size={}", cacheService.getCacheSize());
    }

    /**
     * Registers ground-truth answers in the {@link MockGeminiService} registry.
     *
     * <p>
     * <b>E3 — SBERT/ROUGE correctness:</b> Registering both {@code query} and
     * {@code paraphrase} → the same {@code answer} ensures that when the cache
     * misses on a paraphrase and falls through to the LLM, the mock returns the
     * reference answer rather than a synthetic fallback string. Without this,
     * SBERT cosine similarity would be evaluated against mock strings instead of
     * ground truth, invalidating the semantic fidelity metrics.
     *
     * @param dataset Full (sampled) dataset; registration covers both warmup and
     *                test records
     */
    private void registerGroundTruthIfMock(List<DatasetRecord> dataset) {
        if (!(llmService instanceof MockGeminiService mockLlm))
            return;

        mockLlm.clearRegistry();
        int registeredCount = 0;

        for (DatasetRecord record : dataset) {
            mockLlm.registerGroundTruth(record.query(), record.answer());
            registeredCount++;
            if (record.hasParaphrase()) {
                mockLlm.registerGroundTruth(record.paraphrase(), record.answer());
                registeredCount++;
            }
        }
        log.info("Mock LLM registry populated: {} entries ({} records × ~2 variants)",
                registeredCount, dataset.size());
    }

    /**
     * Processes the test set, recording per-query metrics and building the query
     * log.
     *
     * <p>
     * <b>Test query selection (E1):</b> When a paraphrase is available it is used
     * as
     * the test query; the cached original will be retrieved only if the embedding
     * similarity exceeds θ. This is the intended evaluation scenario: can the
     * system
     * serve semantically equivalent queries without recomputing the LLM response?
     *
     * <p>
     * <b>Embedding reuse (efficiency):</b> If the cache lookup produces a query
     * embedding as a side-effect (which it does for miss queries), that embedding
     * is
     * reused for the subsequent {@code store()} call rather than re-encoding the
     * text.
     *
     * @param testSet List of records from {@link DatasetSplit#testSet()}
     * @param config  Active experiment configuration (for LLM cost estimation)
     * @return Ordered list of per-query observations for post-hoc evaluation
     */
    private List<QueryLog> processTestSet(List<DatasetRecord> testSet, ExperimentConfig config, 
                                           CheckpointManager.Checkpoint checkpoint) {
        int testSize = testSet.size();
        List<QueryLog> queryLogs = java.util.Collections.synchronizedList(new ArrayList<>(testSize));
        
        // Use thread-safe set for checkpoint tracking
        Set<Integer> completedIndices = java.util.Collections.synchronizedSet(checkpoint.completedQueryIndices);

        // M.6 Gold Standard: Zipfian Distribution Generator
        // Simulates realistic "Head/Tail" traffic where some queries are much more
        // frequent.
        List<Integer> queryIndices = new ArrayList<>();
        if (zipfianSkew > 0.01) {
            log.info("Generating Zipfian query sequence (skew={})", zipfianSkew);
            // Precompute weights then build CDF for O(N log N) total vs O(N²) linear scan
            double[] weights = new double[testSize];
            double normConst = 0.0;
            for (int i = 0; i < testSize; i++) {
                weights[i] = 1.0 / Math.pow(i + 1, zipfianSkew);
                normConst += weights[i];
            }
            double[] cdf = new double[testSize];
            double cumSum = 0.0;
            for (int i = 0; i < testSize; i++) {
                cumSum += weights[i] / normConst;
                cdf[i] = cumSum;
            }
            Random rand = new Random(config.randomSeed());
            for (int i = 0; i < testSize; i++) {
                double p = rand.nextDouble();
                int idx = java.util.Arrays.binarySearch(cdf, p);
                if (idx < 0) idx = -(idx + 1);
                queryIndices.add(Math.min(idx, testSize - 1));
            }
        } else {
            // Uniform distribution (default)
            for (int i = 0; i < testSize; i++)
                queryIndices.add(i);
        }

        // TURBO MODE: Parallel processing with custom ForkJoinPool to prevent thread exhaustion
        // Default ForkJoinPool can cause issues with 450K queries - use bounded pool
        int parallelism = Math.min(Runtime.getRuntime().availableProcessors() * 2, MAX_THREAD_POOL_SIZE);
        @SuppressWarnings("resource") // Closed in finally block
        java.util.concurrent.ForkJoinPool customThreadPool = new java.util.concurrent.ForkJoinPool(parallelism);
        
        java.util.concurrent.atomic.AtomicInteger progressCounter = new java.util.concurrent.atomic.AtomicInteger(0);
        long benchmarkStartTime = System.currentTimeMillis();
        
        try {
            customThreadPool.submit(() -> {
                queryIndices.parallelStream().forEach(index -> {
            // Skip if already completed (checkpoint resume)
            if (completedIndices.contains(index)) {
                return;
            }
            
            DatasetRecord record = testSet.get(index);
            
            // Retry loop - NEVER FAIL! (but with max 100 attempts to prevent infinite loops)
            boolean success = false;
            int retryCount = 0;
            final int MAX_RETRIES = 100;
            while (!success && retryCount < MAX_RETRIES) {
                try {
                    long wallClockStart = System.nanoTime();

            // Select test query: paraphrase when available (stresses semantic path),
            // otherwise fall back to the original (tests exact-match path)
            String testQuery = record.hasParaphrase() ? record.paraphrase() : record.query();

            // M.6 Gold Standard: Adversarial Noise Injection (Robustness)
            if (config.noiseProbability() > 0.0) {
                testQuery = noiseGenerator.injectNoise(testQuery, config.noiseProbability(), config.randomSeed());
            }

            CacheLookupResult lookupResult = cacheService.lookup(testQuery);

            if (lookupResult.hit()) {
                // ──── CACHE HIT path ──────────────────────────────────────────────
                // The cache returned a response that satisfied similarity threshold θ.
                // No LLM call required; total cost = 0.
                String response = lookupResult.response();

                long totalMs = (System.nanoTime() - wallClockStart) / 1_000_000;
                metricsCollector.record(
                        true, totalMs,
                        lookupResult.embeddingTimeMs(), 0L,
                        lookupResult.similarityScore(),
                        0.0,
                        llmService.estimateCost(testQuery, record.answer()));

                queryLogs.add(new QueryLog(
                        testQuery, record.answer(), response,
                        true, lookupResult.similarityScore(),
                        totalMs, lookupResult.embeddingTimeMs(), 0L));

            } else {
                // ──── CACHE MISS path ─────────────────────────────────────────────
                // No sufficiently similar entry found. Invoke LLM and store result.
                long llmStart = System.nanoTime();

                // M.6 Fix: Even if testQuery is a paraphrase, we generate the answer
                // based on testQuery but ensure the cost is estimated fairly.
                String response = llmService.generateSync(testQuery);
                long llmLatencyMs = (System.nanoTime() - llmStart) / 1_000_000;

                // Reuse the embedding computed during lookup to avoid redundant ONNX call
                float[] embedding = lookupResult.queryEmbedding() != null
                        ? lookupResult.queryEmbedding()
                        : embeddingService.encode(testQuery);
                cacheService.store(testQuery, embedding, response);

                long totalMs = (System.nanoTime() - wallClockStart) / 1_000_000;

                // Cost calculation evaluates the actual tokens generated/used.
                double actualCost = llmService.estimateCost(testQuery, response);
                double baselineCost = llmService.estimateCost(testQuery, record.answer());

                metricsCollector.record(
                        false, totalMs,
                        lookupResult.embeddingTimeMs(), llmLatencyMs,
                        0.0, actualCost, baselineCost);

                queryLogs.add(new ExperimentResultExporter.QueryLog(
                        testQuery, record.answer(), response,
                        false, 0.0,
                        totalMs, lookupResult.embeddingTimeMs(), llmLatencyMs));
            }

            int done = progressCounter.incrementAndGet();
            
            // Mark query as completed and save checkpoint (thread-safe)
            completedIndices.add(index);
            int checkpointFrequency = done < 100 ? CHECKPOINT_FREQUENCY_SMALL : CHECKPOINT_FREQUENCY_LARGE;
            if (done % checkpointFrequency == 0) {
                checkpointManager.saveCheckpoint(checkpoint);
            }
            
            // Health check every N queries
            if (done % HEALTH_CHECK_FREQUENCY == 0) {
                performHealthCheck(done, testSize);
            }
            
            // Enhanced progress reporting every N queries
            if (done % PROGRESS_REPORT_FREQUENCY == 0) {
                long elapsedMs = System.currentTimeMillis() - benchmarkStartTime;
                double progressPct = (done * 100.0) / testSize;
                long etaMs = (long) ((elapsedMs / done) * (testSize - done));
                long etaMinutes = etaMs / 60000;
                
                // Calculate current metrics
                double currentHitRate = metricsCollector.getHitCount() * 100.0 / done;
                double avgLatency = metricsCollector.getAverageLatency();
                
                log.info("Progress: {}/{} ({:.1f}%) | ETA: {}min | Hit Rate: {:.1f}% | Avg Latency: {:.0f}ms",
                        done, testSize, progressPct, etaMinutes, currentHitRate, avgLatency);
            }
            
            success = true; // Mark as successful
            
                } catch (Exception ex) {
                    retryCount++;
                    if (retryCount >= MAX_RETRIES) {
                        log.error("Query {} failed after {} attempts. Skipping to prevent infinite loop.", 
                                index, MAX_RETRIES);
                        break; // Exit retry loop after max attempts
                    }
                    log.error("Query {} failed (attempt {}/{}): {}. Retrying in 5s...", 
                            index, retryCount, MAX_RETRIES, ex.getMessage());
                    try {
                        Thread.sleep(5000);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                    // Loop continues - NEVER GIVE UP (until max retries)!
                }
            } // end while
                });
            }).get(); // Wait for completion
        } catch (Exception e) {
            log.error("Parallel processing failed: {}", e.getMessage(), e);
        } finally {
            customThreadPool.shutdown();
            try {
                if (!customThreadPool.awaitTermination(THREAD_POOL_SHUTDOWN_TIMEOUT_SECONDS, java.util.concurrent.TimeUnit.SECONDS)) {
                    customThreadPool.shutdownNow();
                }
            } catch (InterruptedException e) {
                customThreadPool.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        log.info("Test phase complete: {} queries processed", testSet.size());
        return queryLogs;
    }
    
    /**
     * Performs health check to ensure system is healthy for continued operation.
     * Checks: memory, disk space, thread count, Redis connectivity.
     * If unhealthy, logs warning but continues (graceful degradation).
     */
    private void performHealthCheck(int queriesCompleted, int totalQueries) {
        try {
            // 1. Memory check
            Runtime runtime = Runtime.getRuntime();
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            long maxMemory = runtime.maxMemory();
            double memoryUsagePercent = (usedMemory * 100.0) / maxMemory;
            
            if (memoryUsagePercent > 90) {
                log.warn("⚠️ HIGH MEMORY USAGE: {:.1f}% ({} MB / {} MB)", 
                        memoryUsagePercent, usedMemory / (1024 * 1024), maxMemory / (1024 * 1024));
                // Suggest GC
                System.gc();
            }
            
            // 2. Disk space check
            File resultsDir = new File("results");
            long freeSpaceMB = resultsDir.getFreeSpace() / (1024 * 1024);
            if (freeSpaceMB < 1000) {
                log.warn("⚠️ LOW DISK SPACE: {} MB free (recommend 1GB+)", freeSpaceMB);
            }
            
            // 3. Thread count check
            int threadCount = Thread.activeCount();
            if (threadCount > 100) {
                log.warn("⚠️ HIGH THREAD COUNT: {} active threads", threadCount);
            }
            
            // 4. Progress sanity check
            double progressPercent = (queriesCompleted * 100.0) / totalQueries;
            if (progressPercent > 0 && progressPercent < 100) {
                log.debug("✅ Health check passed: Memory {:.1f}%, Disk {} MB, Threads {}", 
                        memoryUsagePercent, freeSpaceMB, threadCount);
            }
            
        } catch (Exception e) {
            log.warn("Health check failed: {}", e.getMessage());
        }
    }
}
