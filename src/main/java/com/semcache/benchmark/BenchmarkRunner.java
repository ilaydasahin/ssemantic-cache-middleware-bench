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

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
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
    
    // ── Experiment constants (extracted from magic numbers) ────────────────
    private static final int HEALTH_CHECK_FREQUENCY      = 100; // queries
    private static final int PROGRESS_REPORT_FREQUENCY   = 100; // queries
    private static final int CHECKPOINT_FREQUENCY_SMALL  = 25;
    private static final int CHECKPOINT_FREQUENCY_LARGE  = 100;

    // ── Dependencies ──────────────────────────────────────────────────────────
    private final SemanticCacheService cacheService;
    private final EmbeddingService embeddingService;
    private final LLMService llmService;
    private final DatasetLoader datasetLoader;
    private final MetricsCollector metricsCollector;
    private final ExperimentResultExporter resultExporter;
    private final NoiseGenerator noiseGenerator;
    private final CheckpointManager checkpointManager;
    private final ExperimentValidator experimentValidator;

    public BenchmarkRunner(SemanticCacheService cacheService,
            EmbeddingService embeddingService,
            LLMService llmService,
            DatasetLoader datasetLoader,
            MetricsCollector metricsCollector,
            ExperimentResultExporter resultExporter,
            NoiseGenerator noiseGenerator,
            CheckpointManager checkpointManager,
            ExperimentValidator experimentValidator) {
        this.cacheService = cacheService;
        this.embeddingService = embeddingService;
        this.llmService = llmService;
        this.datasetLoader = datasetLoader;
        this.metricsCollector = metricsCollector;
        this.resultExporter = resultExporter;
        this.noiseGenerator = noiseGenerator;
        this.checkpointManager = checkpointManager;
        this.experimentValidator = experimentValidator;
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
        
        // ── Phase 0.1: Pre-flight validation ─────────────────────────────────
        List<String> validationErrors = experimentValidator.validate(config);
        if (!validationErrors.isEmpty()) {
            throw new IllegalArgumentException(
                "Experiment validation failed: " + String.join("; ", validationErrors));
        }
        
        // ── Phase 0.2: Capture metadata for reproducibility ──────────────────
        long startTime = System.currentTimeMillis();
        ExperimentMetadata metadata = ExperimentMetadata.create(config);
        log.info("System: {} {} ({})", 
            metadata.system().os(), 
            metadata.system().osVersion(), 
            metadata.system().arch());
        log.info("Java: {} ({})", 
            metadata.software().javaVersion(), 
            metadata.software().javaVendor());
        log.info("Memory: {} MB heap, {} MB total", 
            metadata.system().maxHeapMB(), 
            metadata.system().totalMemoryMB());
        
        if (metadata.git() != null) {
            log.info("Git: {} @ {} {}", 
                metadata.git().branch(), 
                metadata.git().commitHash().substring(0, Math.min(8, metadata.git().commitHash().length())),
                metadata.git().isDirty() ? "(dirty)" : "");
        }
        
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
        DatasetSplit split = datasetLoader.split(sampledDataset, config.warmupRatio());

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
        
        // Update metadata with final duration and dataset info
        long durationSeconds = (System.currentTimeMillis() - startTime) / 1000;
        metadata = metadata.withDuration(durationSeconds);
        
        // Calculate dataset fingerprint
        try {
            java.io.File datasetFile = new java.io.File(config.datasetPath());
            String sha256 = calculateSHA256(datasetFile);
            ExperimentMetadata.DatasetInfo datasetInfo = new ExperimentMetadata.DatasetInfo(
                config.datasetName(),
                config.datasetPath(),
                datasetFile.length(),
                split.testSet().size() + split.warmupSet().size(),
                sha256
            );
            metadata = metadata.withDatasetInfo(datasetInfo);
        } catch (Exception e) {
            log.warn("Could not calculate dataset fingerprint: {}", e.getMessage());
        }
        
        resultExporter.export(config, metrics, queryLogs, metadata);
        
        // Delete checkpoint after successful completion
        checkpointManager.deleteCheckpoint(experimentId);
        log.info("✅ Experiment completed successfully: {} (duration: {}s)", 
            experimentId, durationSeconds);
    }
    
    /**
     * Calculates SHA-256 fingerprint of a file for reproducibility tracking.
     */
    private String calculateSHA256(java.io.File file) throws Exception {
        java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
        try (java.io.FileInputStream fis = new java.io.FileInputStream(file)) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = fis.read(buffer)) != -1) {
                digest.update(buffer, 0, bytesRead);
            }
        }
        byte[] hashBytes = digest.digest();
        StringBuilder sb = new StringBuilder();
        for (byte b : hashBytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
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
        Set<Integer> completedIndices = checkpoint.completedQueryIndices;
        List<QueryLog> queryLogs = new ArrayList<>(testSize);
        
        // Ensure the list is pre-sized and filled with nulls to preserve order while adding out of order (if we were concurrent)
        // Since we are sequential now, we don't strictly need this, but it guarantees ordering if indices are scrambled.
        for (int i = 0; i < testSize; i++) queryLogs.add(null);

        List<Integer> queryIndices = buildQuerySequence(testSize, config);
        
        long benchmarkStartTime = System.currentTimeMillis();
        int done = completedIndices.size();

        for (int index : queryIndices) {
            if (completedIndices.contains(index)) {
                continue;
            }
            
            DatasetRecord record = testSet.get(index);
            QueryLog logEntry = processSingleQuery(record, config, index);
            queryLogs.set(index, logEntry);
            
            done++;
            completedIndices.add(index);
            
            if (shouldSaveCheckpoint(done)) {
                checkpointManager.saveCheckpoint(checkpoint);
            }
            
            if (done % PROGRESS_REPORT_FREQUENCY == 0) {
                reportProgress(done, testSize, benchmarkStartTime);
            }
            if (done % HEALTH_CHECK_FREQUENCY == 0) {
                performHealthCheck(done, testSize);
            }
        }
        
        log.info("Test phase complete: {} queries processed", testSize);
        
        // Remove nulls (if any queries were absolutely skipped, which shouldn't happen)
        queryLogs.removeIf(l -> l == null);
        
        log.info("Query logs collected: {} entries (expected: {})", queryLogs.size(), testSize);
        return queryLogs;
    }

    private List<Integer> buildQuerySequence(int testSize, ExperimentConfig config) {
        if (config.zipfianSkew() > 0.01) {
            log.info("Generating Zipfian query sequence (skew={})", config.zipfianSkew());
            return ZipfianDistribution.generateIndices(testSize, testSize, config.zipfianSkew(), config.randomSeed());
        } else {
            List<Integer> indices = new ArrayList<>(testSize);
            for (int i = 0; i < testSize; i++) indices.add(i);
            return indices;
        }
    }

    private QueryLog processSingleQuery(DatasetRecord record, ExperimentConfig config, int index) {
        boolean success = false;
        int retryCount = 0;
        int maxRetries = (llmService instanceof MockGeminiService) ? 0 : 5;

        while (!success && retryCount <= maxRetries) {
            try {
                long wallClockStart = System.nanoTime();

                String testQuery = record.hasParaphrase() ? record.paraphrase() : record.query();
                if (config.noiseProbability() > 0.0) {
                    testQuery = noiseGenerator.injectNoise(testQuery, config.noiseProbability(), config.randomSeed());
                }

                CacheLookupResult lookupResult = cacheService.lookup(testQuery);

                if (lookupResult.hit()) {
                    String response = lookupResult.response();
                    double totalMs = (System.nanoTime() - wallClockStart) / 1_000_000.0;
                    double embeddingMs = lookupResult.embeddingTimeMs() / 1_000_000.0;

                    metricsCollector.record(
                            true, totalMs, embeddingMs, 0.0,
                            lookupResult.similarityScore(),
                            0.0,
                            llmService.estimateCost(testQuery, record.answer()));

                    return new QueryLog(
                            testQuery, record.answer(), response,
                            true, lookupResult.similarityScore(),
                            totalMs, embeddingMs, 0.0);

                } else {
                    long llmStart = System.nanoTime();
                    String response = llmService.generateSync(testQuery);
                    double llmLatencyMs = (System.nanoTime() - llmStart) / 1_000_000.0;

                    float[] embedding = lookupResult.queryEmbedding() != null
                            ? lookupResult.queryEmbedding()
                            : embeddingService.encode(testQuery);
                    cacheService.store(testQuery, embedding, response);

                    double totalMs = (System.nanoTime() - wallClockStart) / 1_000_000.0;
                    double embeddingMs = lookupResult.embeddingTimeMs() / 1_000_000.0;
                    double actualCost = llmService.estimateCost(testQuery, response);
                    double baselineCost = llmService.estimateCost(testQuery, record.answer());

                    metricsCollector.record(
                            false, totalMs, embeddingMs, llmLatencyMs,
                            0.0, actualCost, baselineCost);

                    return new QueryLog(
                            testQuery, record.answer(), response,
                            false, 0.0,
                            totalMs, embeddingMs, llmLatencyMs);
                }
            } catch (Exception ex) {
                retryCount++;
                if (retryCount > maxRetries) {
                    log.error("Query {} failed after {} attempts.", index, maxRetries);
                    // Ö-4 fix: never return null — always return a structured ERROR entry
                    return new QueryLog(
                            record.query(), record.answer(), "ERROR: Max retries exceeded",
                            false, 0.0, 0.0, 0.0, 0.0);
                }
                long backoffMs = Math.min(5000L * (1L << (retryCount - 1)), 60000L);
                log.warn("Query {} failed (attempt {}/{}): {}. Retrying in {}s...",
                        index, retryCount, maxRetries, ex.getMessage(), backoffMs / 1000);
                try {
                    Thread.sleep(backoffMs);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    // Ö-4 fix: interrupted — return ERROR entry instead of null
                    return new QueryLog(
                            record.query(), record.answer(), "ERROR: Interrupted",
                            false, 0.0, 0.0, 0.0, 0.0);
                }
            }
        }
        // Should never reach here, but return ERROR instead of null as safety net
        return new QueryLog(
                record.query(), record.answer(), "ERROR: Unexpected exit",
                false, 0.0, 0.0, 0.0, 0.0);
    }

    private boolean shouldSaveCheckpoint(int done) {
        int checkpointFrequency = done < 100 ? CHECKPOINT_FREQUENCY_SMALL : CHECKPOINT_FREQUENCY_LARGE;
        return done % checkpointFrequency == 0;
    }

    private void reportProgress(int done, int testSize, long benchmarkStartTime) {
        long elapsedMs = System.currentTimeMillis() - benchmarkStartTime;
        double progressPct = (done * 100.0) / testSize;
        long etaMs = (long) ((elapsedMs / done) * (testSize - done));
        long etaMinutes = etaMs / 60000;
        
        double currentHitRate = metricsCollector.getHitCount() * 100.0 / done;
        double avgLatency = metricsCollector.getAverageLatency();
        
        log.info("Progress: {}/{} ({}%) | ETA: {}min | Hit Rate: {}% | Avg Latency: {}ms",
                done, testSize, 
                String.format(java.util.Locale.US, "%.1f", progressPct), 
                etaMinutes, 
                String.format(java.util.Locale.US, "%.1f", currentHitRate), 
                String.format(java.util.Locale.US, "%.0f", avgLatency));
    }

    private void performHealthCheck(int queriesCompleted, int totalQueries) {
        try {
            Runtime runtime = Runtime.getRuntime();
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            long maxMemory = runtime.maxMemory();
            double memoryUsagePercent = (usedMemory * 100.0) / maxMemory;
            
            if (memoryUsagePercent > 90) {
                log.warn("HIGH MEMORY USAGE detected: {}% ({} MB / {} MB). Consider increasing -Xmx.", 
                        String.format(java.util.Locale.US, "%.1f", memoryUsagePercent), 
                        usedMemory / (1024 * 1024), maxMemory / (1024 * 1024));
            }
            
            File resultsDir = new File("results");
            long freeSpaceMB = resultsDir.getFreeSpace() / (1024 * 1024);
            if (freeSpaceMB < 1000) {
                log.warn("LOW DISK SPACE: {} MB free (recommend 1GB+)", freeSpaceMB);
            }
            
            int threadCount = Thread.activeCount();
            if (threadCount > 100) {
                log.warn("HIGH THREAD COUNT: {} active threads", threadCount);
            }
        } catch (Exception e) {
            log.warn("Health check failed: {}", e.getMessage());
        }
    }

}