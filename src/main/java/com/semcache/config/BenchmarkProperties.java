package com.semcache.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import java.util.List;

@Configuration
@ConfigurationProperties(prefix = "benchmark")
public class BenchmarkProperties {

    private List<Integer> seeds;
    private List<Integer> concurrentUsers;
    private List<DatasetConfig> datasets;

    // Runtime overrides for run_experiments.sh
    private String currentDataset;
    private Integer currentSeed;
    private String outputFile;
    private Double similarityThreshold;
    private Integer sampleSize;
    private String strategy;
    private String warmupStrategy;
    /**
     * Fraction of dataset for cache warming — injected from benchmark.warmup-ratio
     */
    private Double warmupRatio;
    private Boolean hnswEnabled;

    // M.8 Fix: allow CLI override for cache capacity (Deney 1) and HNSW neighbors
    private Integer maxCacheEntries;
    private Integer knnK;
    private Double zipfianSkew;
    private Double noiseProbability;
    private Boolean heavyChurn;

    public Boolean isHeavyChurn() {
        return heavyChurn;
    }

    public void setHeavyChurn(Boolean heavyChurn) {
        this.heavyChurn = heavyChurn;
    }

    public Integer getMaxCacheEntries() {
        return maxCacheEntries;
    }

    public void setMaxCacheEntries(Integer maxCacheEntries) {
        this.maxCacheEntries = maxCacheEntries;
    }

    public Integer getKnnK() {
        return knnK;
    }

    public void setKnnK(Integer knnK) {
        this.knnK = knnK;
    }

    public Boolean getHnswEnabled() {
        return hnswEnabled;
    }

    public void setHnswEnabled(Boolean hnswEnabled) {
        this.hnswEnabled = hnswEnabled;
    }

    public String getStrategy() {
        return strategy;
    }

    public void setStrategy(String strategy) {
        this.strategy = strategy;
    }

    public String getWarmupStrategy() {
        return warmupStrategy;
    }

    public void setWarmupStrategy(String warmupStrategy) {
        this.warmupStrategy = warmupStrategy;
    }

    public Double getWarmupRatio() {
        return warmupRatio;
    }

    public void setWarmupRatio(Double warmupRatio) {
        this.warmupRatio = warmupRatio;
    }

    public static class DatasetConfig {
        private String name;
        private String path;
        private int size;

        // Getters and Setters
        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getPath() {
            return path;
        }

        public void setPath(String path) {
            this.path = path;
        }

        public int getSize() {
            return size;
        }

        public void setSize(int size) {
            this.size = size;
        }
    }

    // Getters and Setters
    public List<Integer> getSeeds() {
        return seeds;
    }

    public void setSeeds(List<Integer> seeds) {
        this.seeds = seeds;
    }

    public List<Integer> getConcurrentUsers() {
        return concurrentUsers;
    }

    public void setConcurrentUsers(List<Integer> concurrentUsers) {
        this.concurrentUsers = concurrentUsers;
    }

    public List<DatasetConfig> getDatasets() {
        return datasets;
    }

    public void setDatasets(List<DatasetConfig> datasets) {
        this.datasets = datasets;
    }

    public String getCurrentDataset() {
        return currentDataset;
    }

    public void setCurrentDataset(String currentDataset) {
        this.currentDataset = currentDataset;
    }

    public Integer getCurrentSeed() {
        return currentSeed;
    }

    public void setCurrentSeed(Integer currentSeed) {
        this.currentSeed = currentSeed;
    }

    public String getOutputFile() {
        return outputFile;
    }

    public void setOutputFile(String outputFile) {
        this.outputFile = outputFile;
    }

    public Double getSimilarityThreshold() {
        return similarityThreshold;
    }

    public void setSimilarityThreshold(Double similarityThreshold) {
        this.similarityThreshold = similarityThreshold;
    }

    public Integer getSampleSize() {
        return sampleSize;
    }

    public void setSampleSize(Integer sampleSize) {
        this.sampleSize = sampleSize;
    }

    public Double getZipfianSkew() {
        return zipfianSkew;
    }

    public void setZipfianSkew(Double zipfianSkew) {
        this.zipfianSkew = zipfianSkew;
    }

    public Double getNoiseProbability() {
        return noiseProbability;
    }

    public void setNoiseProbability(Double noiseProbability) {
        this.noiseProbability = noiseProbability;
    }
}
