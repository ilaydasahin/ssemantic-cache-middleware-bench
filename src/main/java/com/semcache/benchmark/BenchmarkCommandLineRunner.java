package com.semcache.benchmark;

import com.semcache.config.BenchmarkProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;

@Component
@Profile("benchmark")
public class BenchmarkCommandLineRunner implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(BenchmarkCommandLineRunner.class);

    private final BenchmarkRunner benchmarkRunner;
    private final ThroughputBenchmarkRunner throughputRunner;
    private final EvictionStressTestRunner stressTestRunner;
    private final MegaBenchmarkSuiteRunner megaRunner;
    private final BenchmarkProperties properties;
    private final DatasetLoader datasetLoader;
    private final ApplicationContext applicationContext;

    public BenchmarkCommandLineRunner(BenchmarkRunner benchmarkRunner,
            ThroughputBenchmarkRunner throughputRunner,
            EvictionStressTestRunner stressTestRunner,
            MegaBenchmarkSuiteRunner megaRunner,
            BenchmarkProperties properties,
            DatasetLoader datasetLoader,
            ApplicationContext applicationContext) {
        this.benchmarkRunner = benchmarkRunner;
        this.throughputRunner = throughputRunner;
        this.stressTestRunner = stressTestRunner;
        this.megaRunner = megaRunner;
        this.properties = properties;
        this.datasetLoader = datasetLoader;
        this.applicationContext = applicationContext;
    }

    @Override
    public void run(String... args) throws Exception {
        // Ö-3: Verify Python scripts availability to ensure post-hoc analysis can run
        java.io.File analyzeScript = new java.io.File("scripts/analyze_results.py");
        if (!analyzeScript.exists()) {
            log.warn("WARNING: scripts/analyze_results.py not found. Post-hoc compute for avgBertScore/avgRougeL will fail, leaving nulls in outputs.");
        }

        // ── MEGA MODE (Q1 Grade — single JVM, full matrix) ───────────────────
        if ("mega".equalsIgnoreCase(properties.getMode())) {
            log.info("MEGA mode activated — running full experiment matrix in single JVM");
            try {
                runMegaBenchmark();
                System.exit(SpringApplication.exit(applicationContext, () -> 0));
            } catch (Exception e) {
                log.error("MEGA benchmark failed: {}", e.getMessage(), e);
                System.exit(SpringApplication.exit(applicationContext, () -> 1));
            }
            return;
        }

        // ── Legacy modes (unchanged) ─────────────────────────────────────────

        if (properties.isHeavyChurn() != null && properties.isHeavyChurn()) {
            log.info("Heavy churn mode enabled.");
            BenchmarkProperties.DatasetConfig datasetConfig = resolveDatasetConfig(resolveCurrentDatasetName());
            stressTestRunner.runHeavyChurnTest(datasetConfig.getPath(), datasetLoader, properties.getOutputFile());
            System.exit(SpringApplication.exit(applicationContext, () -> 0));
            return;
        }

        Integer singleConcurrentUsers = resolveThroughputUsers();
        if (singleConcurrentUsers != null) {
            log.info("Throughput mode: concurrent-users={}", singleConcurrentUsers);
            BenchmarkProperties.DatasetConfig datasetConfig = resolveDatasetConfig(resolveCurrentDatasetName());
            log.info("Loading dataset {} for throughput test...", datasetConfig.getName());
            java.util.List<DatasetLoader.DatasetRecord> dataset = datasetLoader.load(datasetConfig.getPath());
            long seed = properties.getCurrentSeed() != null ? properties.getCurrentSeed().longValue() : 42L;
            throughputRunner.runForUsers(singleConcurrentUsers, dataset, properties.getOutputFile(), seed);
            System.exit(SpringApplication.exit(applicationContext, () -> 0));
            return;
        }

        String datasetName = properties.getCurrentDataset();
        Integer seed = properties.getCurrentSeed();
        String outputFile = properties.getOutputFile();

        if (datasetName == null || seed == null) {
            log.error("Missing required parameters.");
            System.exit(SpringApplication.exit(applicationContext, () -> 1));
            return;
        }

        BenchmarkProperties.DatasetConfig datasetConfig = findDatasetConfig(datasetName);
        if (datasetConfig == null) {
            log.error("Dataset not found.");
            System.exit(SpringApplication.exit(applicationContext, () -> 1));
            return;
        }

        double zSkew = properties.getZipfianSkew() != null ? properties.getZipfianSkew() : 0.0;
        double nProb = properties.getNoiseProbability() != null ? properties.getNoiseProbability() : 0.0;

        ExperimentConfig config = ExperimentConfig.createV2(
                datasetName,
                datasetConfig.getPath(),
                "auto",
                properties.getSimilarityThreshold() != null ? properties.getSimilarityThreshold() : 0.90,
                properties.getWarmupStrategy() != null ? properties.getWarmupStrategy() : "BIDIRECTIONAL",
                properties.getWarmupRatio() != null ? properties.getWarmupRatio() : 0.30,
                seed.longValue(),
                properties.getSampleSize(),
                properties.getHnswEnabled() != null ? properties.getHnswEnabled() : true,
                properties.getStrategy() != null ? properties.getStrategy() : "SEMANTIC",
                properties.getKnnK() != null ? properties.getKnnK() : 5,
                properties.getMaxCacheEntries() != null ? properties.getMaxCacheEntries() : 50_000,
                properties.getTtlSeconds() != null ? properties.getTtlSeconds() : 86400L,
                null,
                zSkew,
                nProb,
                outputFile);

        try {
            benchmarkRunner.run(config);
            log.info("Benchmark complete — written to: {}", outputFile);
            System.exit(SpringApplication.exit(applicationContext, () -> 0));
        } catch (Exception e) {
            log.error("Benchmark run failed: {}", e.getMessage(), e);
            System.exit(SpringApplication.exit(applicationContext, () -> 1));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mega Benchmark Orchestration
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Parses mega-mode parameters from BenchmarkProperties and delegates to
     * MegaBenchmarkSuiteRunner.
     */
    private void runMegaBenchmark() throws Exception {
        // Strategies: from CLI --benchmark.strategy or default to all three
        List<String> strategies;
        if (properties.getStrategy() != null && !properties.getStrategy().isBlank()) {
            strategies = Arrays.asList(properties.getStrategy().split(","));
        } else {
            strategies = List.of("EXACT_MATCH", "SEMANTIC", "HYBRID");
        }

        // Seeds: from application.yml benchmark.seeds
        List<Integer> seeds = properties.getSeeds();
        if (seeds == null || seeds.isEmpty()) {
            seeds = List.of(42, 123, 456, 789, 1024);
        }

        // Concurrent users: from application.yml benchmark.concurrent-users
        List<Integer> concurrentUsers = properties.getConcurrentUsers();
        if (concurrentUsers == null || concurrentUsers.isEmpty()) {
            concurrentUsers = List.of(10, 25, 50, 100);
        }

        // Results directory with timestamp
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String resultsDir = String.format("results/mega_%s", timestamp);

        megaRunner.runFullSuite(strategies, seeds, concurrentUsers, resultsDir);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers (unchanged)
    // ─────────────────────────────────────────────────────────────────────────

    /** Returns the current dataset name, falling back to the first configured dataset. */
    private String resolveCurrentDatasetName() {
        String name = properties.getCurrentDataset();
        if (name == null && !properties.getDatasets().isEmpty()) {
            name = properties.getDatasets().get(0).getName();
        }
        return name;
    }

    /** Finds the dataset config for the given name, falling back to first if not matched. */
    private BenchmarkProperties.DatasetConfig resolveDatasetConfig(String name) {
        return properties.getDatasets().stream()
                .filter(d -> d.getName().equalsIgnoreCase(name))
                .findFirst()
                .orElse(properties.getDatasets().get(0));
    }

    private BenchmarkProperties.DatasetConfig findDatasetConfig(String name) {
        return properties.getDatasets().stream()
                .filter(d -> d.getName().equalsIgnoreCase(name))
                .findFirst()
                .orElse(null);
    }

    private Integer resolveThroughputUsers() {
        if (properties.getCurrentDataset() == null
                && properties.getConcurrentUsers() != null
                && !properties.getConcurrentUsers().isEmpty()) {
            return properties.getConcurrentUsers().get(0);
        }
        return null;
    }
}
