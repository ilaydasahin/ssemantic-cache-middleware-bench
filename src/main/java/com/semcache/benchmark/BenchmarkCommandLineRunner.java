package com.semcache.benchmark;

import com.semcache.config.BenchmarkProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

@Component
@Profile("benchmark")
public class BenchmarkCommandLineRunner implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(BenchmarkCommandLineRunner.class);

    private final BenchmarkRunner benchmarkRunner;
    private final ThroughputBenchmarkRunner throughputRunner;
    private final EvictionStressTestRunner stressTestRunner;
    private final BenchmarkProperties properties;
    private final DatasetLoader datasetLoader;

    public BenchmarkCommandLineRunner(BenchmarkRunner benchmarkRunner,
            ThroughputBenchmarkRunner throughputRunner,
            EvictionStressTestRunner stressTestRunner,
            BenchmarkProperties properties,
            DatasetLoader datasetLoader) {
        this.benchmarkRunner = benchmarkRunner;
        this.throughputRunner = throughputRunner;
        this.stressTestRunner = stressTestRunner;
        this.properties = properties;
        this.datasetLoader = datasetLoader;
    }

    @Override
    public void run(String... args) throws Exception {

        if (properties.isHeavyChurn() != null && properties.isHeavyChurn()) {
            log.info("Heavy churn mode enabled.");
            BenchmarkProperties.DatasetConfig datasetConfig = resolveDatasetConfig(resolveCurrentDatasetName());
            stressTestRunner.runHeavyChurnTest(datasetConfig.getPath(), datasetLoader, properties.getOutputFile());
            System.exit(0);
            return;
        }

        Integer singleConcurrentUsers = resolveThroughputUsers();
        if (singleConcurrentUsers != null) {
            log.info("Throughput mode: concurrent-users={}", singleConcurrentUsers);
            BenchmarkProperties.DatasetConfig datasetConfig = resolveDatasetConfig(resolveCurrentDatasetName());
            log.info("Loading dataset {} for throughput test...", datasetConfig.getName());
            java.util.List<DatasetLoader.DatasetRecord> dataset = datasetLoader.load(datasetConfig.getPath());
            throughputRunner.runForUsers(singleConcurrentUsers, dataset, properties.getOutputFile());
            System.exit(0);
            return;
        }

        String datasetName = properties.getCurrentDataset();
        Integer seed = properties.getCurrentSeed();
        String outputFile = properties.getOutputFile();

        if (datasetName == null || seed == null) {
            log.error("Missing required parameters.");
            System.exit(1);
            return;
        }

        BenchmarkProperties.DatasetConfig datasetConfig = findDatasetConfig(datasetName);
        if (datasetConfig == null) {
            log.error("Dataset not found.");
            System.exit(1);
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
            System.exit(0);
        } catch (Exception e) {
            log.error("Benchmark run failed: {}", e.getMessage(), e);
            System.exit(1);
        }
    }

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
