package com.semcache.benchmark;

import com.semcache.benchmark.DatasetLoader.DatasetRecord;
import com.semcache.config.BenchmarkProperties;
import com.semcache.service.SemanticCacheService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * Q1-Grade Mega Benchmark Suite Runner.
 *
 * <p><b>Scientific Purpose (§4, M.6):</b> Orchestrates the full combinatorial
 * experiment matrix (Datasets × Strategies × Seeds × ConcurrentUsers) within
 * a <em>single, continuously running JVM process</em>. This design eliminates
 * the JIT cold-start measurement bias that occurs when each trial is launched
 * as a separate {@code mvn spring-boot:run} invocation (see Implementation Plan,
 * "JIT Warmup & Measurement Bias").
 *
 * <p><b>Design rationale:</b>
 * <ul>
 *   <li>The JVM's C2 JIT compiler reaches steady-state after ~10 000 method
 *       invocations. By running a dedicated JIT warm-up phase first and then
 *       iterating over the experiment matrix in-process, all measured latencies
 *       reflect optimised code paths — the same condition under which a
 *       production microservice would operate.</li>
 *   <li>Spring context boot (≈5–6 s) happens only <em>once</em>, saving
 *       {@code N_experiments × 6 s} of calendar time (e.g. 2 304 × 6 s ≈ 4 h).</li>
 *   <li>Redis connection pool is reused across trials, preventing socket
 *       exhaustion and TTL disruptions.</li>
 * </ul>
 *
 * <p><b>Threat to validity addressed:</b> Between consecutive trials the
 * in-memory cache and all metric accumulators are flushed via
 * {@link SemanticCacheService#clearCache()} and
 * {@link MetricsCollector#reset()}, so measurements are independent.
 *
 * <p>Activated with {@code --benchmark.mode=mega}.
 */
@Component
@Profile("benchmark")
public class MegaBenchmarkSuiteRunner {

    private static final Logger log = LoggerFactory.getLogger(MegaBenchmarkSuiteRunner.class);

    /**
     * Number of warm-up iterations to ensure JIT has compiled hot paths.
     * JMH uses 10 warm-up iterations by default; we use 3 full trials
     * (each processing hundreds of queries) which is more than sufficient.
     */
    private static final int JIT_WARMUP_TRIALS = 3;

    private final BenchmarkRunner benchmarkRunner;
    private final ThroughputBenchmarkRunner throughputRunner;
    private final BenchmarkProperties properties;
    private final DatasetLoader datasetLoader;

    public MegaBenchmarkSuiteRunner(BenchmarkRunner benchmarkRunner,
                                    ThroughputBenchmarkRunner throughputRunner,
                                    BenchmarkProperties properties,
                                    DatasetLoader datasetLoader) {
        this.benchmarkRunner = benchmarkRunner;
        this.throughputRunner = throughputRunner;
        this.properties = properties;
        this.datasetLoader = datasetLoader;
    }

    /**
     * Executes the full mega benchmark suite.
     *
     * <p>The experiment matrix is defined by the configuration in
     * {@code application.yml} under the {@code benchmark} prefix. Each cell
     * in the matrix is a single trial executed by {@link BenchmarkRunner#run}.
     *
     * @param strategies      Cache strategies to test (e.g. EXACT_MATCH, SEMANTIC, HYBRID)
     * @param seeds           Random seeds for reproducibility (e.g. 42, 123, 456, 789, 1024)
     * @param concurrentUsers Concurrency levels for throughput tests (e.g. 10, 25, 50, 100)
     * @param resultsDir      Base directory for results output
     */
    public void runFullSuite(List<String> strategies,
                             List<Integer> seeds,
                             List<Integer> concurrentUsers,
                             String resultsDir) throws Exception {

        List<BenchmarkProperties.DatasetConfig> datasets = properties.getDatasets();
        if (datasets == null || datasets.isEmpty()) {
            throw new IllegalStateException("No datasets configured in benchmark.datasets");
        }

        // ── Calculate experiment matrix ──────────────────────────────────────
        int totalAccuracyExperiments = datasets.size() * strategies.size() * seeds.size();
        int totalThroughputExperiments = datasets.size() * strategies.size() * concurrentUsers.size();
        int totalExperiments = totalAccuracyExperiments + totalThroughputExperiments;

        log.info("╔════════════════════════════════════════════════════════════╗");
        log.info("║  MEGA BENCHMARK SUITE — SINGLE JVM (Q1 GRADE)            ║");
        log.info("╚════════════════════════════════════════════════════════════╝");
        log.info("  Datasets        : {} {}", datasets.size(), datasets.stream().map(BenchmarkProperties.DatasetConfig::getName).toList());
        log.info("  Strategies      : {} {}", strategies.size(), strategies);
        log.info("  Seeds           : {} {}", seeds.size(), seeds);
        log.info("  Concurrent Users: {} {}", concurrentUsers.size(), concurrentUsers);
        log.info("  ─────────────────────────────────────────────────────────");
        log.info("  Accuracy Trials    : {}", totalAccuracyExperiments);
        log.info("  Throughput Trials  : {}", totalThroughputExperiments);
        log.info("  TOTAL              : {}", totalExperiments);
        log.info("  Results Dir        : {}", resultsDir);
        log.info("  JIT Warmup Trials  : {}", JIT_WARMUP_TRIALS);
        log.info("");

        new File(resultsDir).mkdirs();

        // ── Phase 0: JIT Warm-up ─────────────────────────────────────────────
        for (String strategy : strategies) {
            runJitWarmup(datasets.get(0), strategy, seeds.get(0));
        }

        // ── Phase 1: Accuracy Experiments (Datasets × Strategies × Seeds) ────
        Instant suiteStart = Instant.now();
        int completed = 0;
        int failed = 0;

        log.info("═══════════════════════════════════════════════════════════════");
        log.info("  PHASE 1: ACCURACY EXPERIMENTS ({} trials)", totalAccuracyExperiments);
        log.info("═══════════════════════════════════════════════════════════════");

        for (BenchmarkProperties.DatasetConfig datasetCfg : datasets) {
            for (String strategy : strategies) {
                for (int seed : seeds) {
                    completed++;
                    String trialId = String.format("%s_%s_seed%d", datasetCfg.getName(), strategy, seed);
                    String outputFile = String.format("%s/%s.json", resultsDir, trialId);

                    logProgress(completed, totalExperiments, trialId, suiteStart);

                    try {
                        double threshold = properties.getSimilarityThreshold() != null
                                ? properties.getSimilarityThreshold() : 0.90;
                        double warmupRatio = properties.getWarmupRatio() != null
                                ? properties.getWarmupRatio() : 0.30;
                        String warmupStrategy = properties.getWarmupStrategy() != null
                                ? properties.getWarmupStrategy() : "BIDIRECTIONAL";
                        boolean hnswEnabled = properties.getHnswEnabled() != null
                                ? properties.getHnswEnabled() : true;
                        int knnK = properties.getKnnK() != null
                                ? properties.getKnnK() : 5;
                        int maxEntries = properties.getMaxCacheEntries() != null
                                ? properties.getMaxCacheEntries() : 50_000;
                        long ttlSec = properties.getTtlSeconds() != null
                                ? properties.getTtlSeconds() : 86400L;
                        double zipfSkew = properties.getZipfianSkew() != null
                                ? properties.getZipfianSkew() : 0.0;
                        double noiseProb = properties.getNoiseProbability() != null
                                ? properties.getNoiseProbability() : 0.0;

                        ExperimentConfig config = ExperimentConfig.createV2(
                                datasetCfg.getName(),
                                datasetCfg.getPath(),
                                "auto",
                                threshold,
                                warmupStrategy,
                                warmupRatio,
                                (long) seed,
                                properties.getSampleSize(),
                                hnswEnabled,
                                strategy,
                                knnK,
                                maxEntries,
                                ttlSec,
                                null,    // concurrentUsers — null for accuracy mode
                                zipfSkew,
                                noiseProb,
                                outputFile);

                        benchmarkRunner.run(config);
                        log.info("  ✅ {} completed", trialId);
                    } catch (Exception e) {
                        failed++;
                        log.error("  ❌ {} FAILED: {}", trialId, e.getMessage(), e);
                    }
                }
            }
        }

        // ── Phase 2: Throughput Experiments (Datasets × Strategies × Users) ──
        log.info("");
        log.info("═══════════════════════════════════════════════════════════════");
        log.info("  PHASE 2: THROUGHPUT EXPERIMENTS ({} trials)", totalThroughputExperiments);
        log.info("═══════════════════════════════════════════════════════════════");

        for (BenchmarkProperties.DatasetConfig datasetCfg : datasets) {
            for (String strategy : strategies) {
                // Load dataset once per (dataset, strategy) pair — reuse across user levels
                List<DatasetRecord> dataset = datasetLoader.load(datasetCfg.getPath());

                for (int users : concurrentUsers) {
                    completed++;
                    String trialId = String.format("throughput_%s_%s_%dusers",
                            datasetCfg.getName(), strategy, users);
                    String outputFile = String.format("%s/%s.json", resultsDir, trialId);

                    logProgress(completed, totalExperiments, trialId, suiteStart);

                    try {
                        // Use first seed for throughput tests (deterministic warmup)
                        long throughputSeed = seeds.get(0).longValue();
                        throughputRunner.runForUsers(users, dataset, outputFile, throughputSeed);
                        log.info("  ✅ {} completed", trialId);
                    } catch (Exception e) {
                        failed++;
                        log.error("  ❌ {} FAILED: {}", trialId, e.getMessage(), e);
                    }
                }
            }
        }

        // ── Summary ──────────────────────────────────────────────────────────
        Duration totalDuration = Duration.between(suiteStart, Instant.now());
        long hours = totalDuration.toHours();
        long minutes = totalDuration.toMinutesPart();
        long seconds = totalDuration.toSecondsPart();

        log.info("");
        log.info("╔════════════════════════════════════════════════════════════╗");
        log.info("║  MEGA BENCHMARK SUITE COMPLETE                            ║");
        log.info("╚════════════════════════════════════════════════════════════╝");
        log.info("  Total Trials   : {}", totalExperiments);
        log.info("  Successful     : {}", totalExperiments - failed);
        log.info("  Failed         : {}", failed);
        log.info("  Duration       : {}h {}m {}s", hours, minutes, seconds);
        log.info("  Results Dir    : {}", resultsDir);
        log.info("  Success Rate   : {}%",
                totalExperiments > 0 ? ((totalExperiments - failed) * 100 / totalExperiments) : 0);
        log.info("");

        if (failed > 0) {
            log.warn("⚠️  {} trials failed. Check logs above for details.", failed);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private — JIT Warm-up
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Runs a small number of throwaway trials to allow the JIT compiler to
     * reach steady-state (C2 compilation) before measurement begins.
     *
     * <p>Results from warm-up trials are written to a temporary directory
     * and ignored in the final analysis.
     */
    private void runJitWarmup(BenchmarkProperties.DatasetConfig dataset,
                              String strategy,
                              int seed) {
        log.info("═══════════════════════════════════════════════════════════════");
        log.info("  PHASE 0: JIT WARM-UP ({} trials — results discarded)", JIT_WARMUP_TRIALS);
        log.info("═══════════════════════════════════════════════════════════════");

        String warmupDir = "results/jit_warmup_discard";
        new File(warmupDir).mkdirs();

        for (int i = 1; i <= JIT_WARMUP_TRIALS; i++) {
            log.info("  JIT Warmup {}/{} ...", i, JIT_WARMUP_TRIALS);
            try {
                double threshold = properties.getSimilarityThreshold() != null
                        ? properties.getSimilarityThreshold() : 0.90;
                double warmupRatio = properties.getWarmupRatio() != null
                        ? properties.getWarmupRatio() : 0.30;

                ExperimentConfig config = ExperimentConfig.createV2(
                        dataset.getName(),
                        dataset.getPath(),
                        "auto",
                        threshold,
                        "BIDIRECTIONAL",
                        warmupRatio,
                        (long) seed,
                        500,   // Small sample for fast warmup
                        true,
                        strategy,
                        5, 50_000, 86400L,
                        null, 0.0, 0.0,
                        String.format("%s/warmup_%d.json", warmupDir, i));

                benchmarkRunner.run(config);
                log.info("  JIT Warmup {}/{} ✅", i, JIT_WARMUP_TRIALS);
            } catch (Exception e) {
                log.warn("  JIT Warmup {}/{} failed (non-fatal): {}", i, JIT_WARMUP_TRIALS, e.getMessage());
            }
        }

        log.info("  JIT warm-up complete. C2 compiler should be at steady-state.");
        log.info("");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private — Progress reporting
    // ─────────────────────────────────────────────────────────────────────────

    private void logProgress(int current, int total, String trialId, Instant suiteStart) {
        double progressPct = (current * 100.0) / total;
        Duration elapsed = Duration.between(suiteStart, Instant.now());

        String eta = "calculating...";
        if (current > 1) {
            long avgSecondsPerTrial = elapsed.getSeconds() / (current - 1);
            long remainingTrials = total - current;
            long etaSeconds = avgSecondsPerTrial * remainingTrials;
            long etaH = etaSeconds / 3600;
            long etaM = (etaSeconds % 3600) / 60;
            eta = String.format("%dh %dm", etaH, etaM);
        }

        log.info("[{}/{}] ({}) ETA: {} | Running: {}",
                current, total,
                String.format(Locale.US, "%.1f%%", progressPct),
                eta, trialId);
    }
}
