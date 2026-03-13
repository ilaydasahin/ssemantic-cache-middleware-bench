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

    public BenchmarkRunner(SemanticCacheService cacheService,
            EmbeddingService embeddingService,
            LLMService llmService,
            DatasetLoader datasetLoader,
            MetricsCollector metricsCollector,
            ExperimentResultExporter resultExporter,
            NoiseGenerator noiseGenerator) {
        this.cacheService = cacheService;
        this.embeddingService = embeddingService;
        this.llmService = llmService;
        this.datasetLoader = datasetLoader;
        this.metricsCollector = metricsCollector;
        this.resultExporter = resultExporter;
        this.noiseGenerator = noiseGenerator;
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

        // ── Phase 2: Register ground-truth answers in Mock LLM (if active) ───
        // Registering paraphrase → same answer prevents "Mock response for: ..."
        // fallbacks
        // that corrupt SBERT / ROUGE-L post-hoc evaluation (fix E3).
        registerGroundTruthIfMock(sampledDataset);

        // ── Phase 3: Pre-populate cache (warmup phase) ───────────────────────
        // Only originals are stored; paraphrases are withheld to preserve test validity
        // (E1).
        warmUpCache(split.warmupSet(), config.warmupStrategy());

        // ── Phase 4: Measurement phase ───────────────────────────────────────
        List<QueryLog> queryLogs = processTestSet(split.testSet(), config);

        // ── Phase 5: Compute and export results ──────────────────────────────
        AggregateMetrics metrics = metricsCollector.compute();
        resultExporter.export(config, metrics, queryLogs);
    }

    /**
     * Legacy wrapper retained for backward compatibility with scripts that pass
     * individual parameters rather than an {@link ExperimentConfig}.
     *
     * @deprecated Prefer {@link #run(ExperimentConfig)} with an explicit config
     *             object.
     */
    @Deprecated(since = "2.0", forRemoval = true)
    public void runSingleBenchmark(String datasetPath,
            String datasetName,
            double threshold,
            int seed,
            Integer sampleSize) throws Exception {
        ExperimentConfig cfg = ExperimentConfig.createV2(
                datasetName, datasetPath,
                embeddingService.getModelName(),
                threshold, "BIDIRECTIONAL", warmupRatio,
                seed, sampleSize,
                true, "SEMANTIC", 5, 50000, 86400L, null, 0.0, 0.0, "/tmp/debug.json");
        run(cfg);
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
    private List<QueryLog> processTestSet(List<DatasetRecord> testSet, ExperimentConfig config) {
        int testSize = testSet.size();
        List<QueryLog> queryLogs = new ArrayList<>(testSize);

        // M.6 Gold Standard: Zipfian Distribution Generator
        // Simulates realistic "Head/Tail" traffic where some queries are much more
        // frequent.
        List<Integer> queryIndices = new ArrayList<>();
        if (zipfianSkew > 0.01) {
            log.info("Generating Zipfian query sequence (skew={})", zipfianSkew);
            Random rand = new Random(config.randomSeed());
            for (int i = 0; i < testSize; i++) {
                queryIndices.add(generateZipfianIndex(testSize, zipfianSkew, rand));
            }
        } else {
            // Uniform distribution (default)
            for (int i = 0; i < testSize; i++)
                queryIndices.add(i);
        }

        for (int index : queryIndices) {
            DatasetRecord record = testSet.get(index);
            long wallClockStart = System.nanoTime();

            // Select test query: paraphrase when available (stresses semantic path),
            // otherwise fall back to the original (tests exact-match path)
            String testQuery = record.hasParaphrase() ? record.paraphrase() : record.query();

            // M.6 Gold Standard: Adversarial Noise Injection (Robustness)
            if (config.noiseProbability() > 0.0) {
                testQuery = noiseGenerator.injectNoise(testQuery, config.noiseProbability(), config.randomSeed());
            }

            CacheLookupResult lookupResult = cacheService.lookup(testQuery);
            String response;
            long llmLatencyMs = 0L;

            if (lookupResult.hit()) {
                // ──── CACHE HIT path ──────────────────────────────────────────────
                // The cache returned a response that satisfied similarity threshold θ.
                // No LLM call required; total cost = 0.
                response = lookupResult.response();

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
                response = llmService.generateSync(testQuery);
                llmLatencyMs = (System.nanoTime() - llmStart) / 1_000_000;

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

            if (queryLogs.size() % 1000 == 0) {
                log.info("Progress: {}/{} queries processed...", queryLogs.size(), testSize);
            }
        }

        log.info("Test phase complete: {} queries processed", testSet.size());
        return queryLogs;
    }

    /**
     * Generates a Zipfian distributed index in range [0, n-1].
     * P(i) = (1/i^s) / sum(1/j^s)
     */
    private int generateZipfianIndex(int n, double skew, Random rand) {
        double cp = 0.0;
        for (int i = 1; i <= n; i++)
            cp += 1.0 / Math.pow(i, skew);

        double target = rand.nextDouble() * cp;
        double sum = 0.0;
        for (int i = 1; i <= n; i++) {
            sum += 1.0 / Math.pow(i, skew);
            if (target <= sum)
                return i - 1;
        }
        return n - 1;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Legacy value types (retained for BenchmarkCommandLineRunner compatibility)
    // ─────────────────────────────────────────────────────────────────────────

    /** @deprecated Use {@link QueryLog} from {@link ExperimentResultExporter} */
    @Deprecated(since = "2.0")
    public record BenchmarkResultWithLogs(
            com.semcache.model.QueryDtos.BenchmarkResult result,
            List<QueryLog> logs) {
    }
}
