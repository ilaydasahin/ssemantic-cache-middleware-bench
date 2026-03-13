package com.semcache.benchmark;

import com.semcache.config.BenchmarkProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
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
            DatasetLoader datasetLoader,
            ObjectMapper objectMapper) {
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
            String datasetName = properties.getCurrentDataset();
            if (datasetName == null && !properties.getDatasets().isEmpty()) {
                datasetName = properties.getDatasets().get(0).getName();
            }
            final String finalDatasetName = datasetName;
            BenchmarkProperties.DatasetConfig datasetConfig = properties.getDatasets().stream()
                    .filter(d -> d.getName().equalsIgnoreCase(finalDatasetName))
                    .findFirst().orElse(properties.getDatasets().get(0));

            stressTestRunner.runHeavyChurnTest(datasetConfig.getPath(), datasetLoader, properties.getOutputFile());
            System.exit(0);
            return;
        }

        Integer singleConcurrentUsers = resolveThroughputUsers();
        if (singleConcurrentUsers != null) {
            log.info("Throughput mode: concurrent-users={}", singleConcurrentUsers);

            String datasetName = properties.getCurrentDataset();
            if (datasetName == null && !properties.getDatasets().isEmpty()) {
                datasetName = properties.getDatasets().get(0).getName(); // fallback
            }
            final String finalDatasetName = datasetName;
            BenchmarkProperties.DatasetConfig datasetConfig = properties.getDatasets().stream()
                    .filter(d -> d.getName().equalsIgnoreCase(finalDatasetName))
                    .findFirst().orElse(properties.getDatasets().get(0));

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

        BenchmarkProperties.DatasetConfig datasetConfig = properties.getDatasets().stream()
                .filter(d -> d.getName().equalsIgnoreCase(datasetName))
                .findFirst().orElse(null);

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
                (double) (properties.getSimilarityThreshold() != null ? properties.getSimilarityThreshold() : 0.90),
                (String) (properties.getWarmupStrategy() != null ? properties.getWarmupStrategy() : "BIDIRECTIONAL"),
                (double) (properties.getWarmupRatio() != null ? properties.getWarmupRatio() : 0.30),
                seed.longValue(),
                (Integer) properties.getSampleSize(),
                (boolean) (properties.getHnswEnabled() != null ? properties.getHnswEnabled() : true),
                (String) (properties.getStrategy() != null ? properties.getStrategy() : "SEMANTIC"),
                (int) (properties.getKnnK() != null ? properties.getKnnK() : 5),
                (int) (properties.getMaxCacheEntries() != null ? properties.getMaxCacheEntries() : 50_000),
                86400L,
                (Integer) null,
                zSkew,
                nProb,
                outputFile);

        try {
            benchmarkRunner.run(config);
            log.info("Benchmark complete — rwritten to: {}", outputFile);
            System.exit(0);
        } catch (Exception e) {
            log.error("Benchmark run failed: {}", e.getMessage(), e);
            System.exit(1);
        }
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
