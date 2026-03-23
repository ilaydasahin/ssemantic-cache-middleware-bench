package com.semcache.config;

import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "cache")
public class CacheProperties {
    
    private static final Logger log = LoggerFactory.getLogger(CacheProperties.class);

    private double similarityThreshold = 0.90;
    private int knnK = 5;
    private int maxEntries = 50000;
    private int ttlSeconds = 86400;
    private String strategy = "SEMANTIC";
    private String warmupStrategy = "BIDIRECTIONAL";
    private boolean hnswEnabled = true;
    private double zipfianSkew = 0.0;
    private HnswConfig hnsw = new HnswConfig();
    
    /**
     * Validates configuration properties after initialization.
     * Throws IllegalArgumentException if any property is invalid.
     */
    @PostConstruct
    public void validateConfiguration() {
        if (similarityThreshold < 0.0 || similarityThreshold > 1.0) {
            throw new IllegalArgumentException(
                    "cache.similarityThreshold must be in [0, 1], got: " + similarityThreshold);
        }
        
        if (knnK <= 0) {
            throw new IllegalArgumentException(
                    "cache.knnK must be > 0, got: " + knnK);
        }
        
        if (maxEntries <= 0) {
            throw new IllegalArgumentException(
                    "cache.maxEntries must be > 0, got: " + maxEntries);
        }
        
        if (ttlSeconds <= 0) {
            throw new IllegalArgumentException(
                    "cache.ttlSeconds must be > 0, got: " + ttlSeconds);
        }
        
        if (zipfianSkew < 0.0) {
            throw new IllegalArgumentException(
                    "cache.zipfianSkew must be >= 0, got: " + zipfianSkew);
        }
        
        log.info("✅ Cache configuration validated: threshold={}, maxEntries={}, ttl={}s, strategy={}", 
                similarityThreshold, maxEntries, ttlSeconds, strategy);
    }

    // Getters and Setters
    public double getSimilarityThreshold() { return similarityThreshold; }
    public void setSimilarityThreshold(double similarityThreshold) { this.similarityThreshold = similarityThreshold; }

    public int getKnnK() { return knnK; }
    public void setKnnK(int knnK) { this.knnK = knnK; }

    public int getMaxEntries() { return maxEntries; }
    public void setMaxEntries(int maxEntries) { this.maxEntries = maxEntries; }

    public int getTtlSeconds() { return ttlSeconds; }
    public void setTtlSeconds(int ttlSeconds) { this.ttlSeconds = ttlSeconds; }

    public String getStrategy() { return strategy; }
    public void setStrategy(String strategy) { this.strategy = strategy; }

    public boolean isHnswEnabled() { return hnswEnabled; }
    public void setHnswEnabled(boolean hnswEnabled) { this.hnswEnabled = hnswEnabled; }

    public double getZipfianSkew() { return zipfianSkew; }
    public void setZipfianSkew(double zipfianSkew) { this.zipfianSkew = zipfianSkew; }

    public String getWarmupStrategy() { return warmupStrategy; }
    public void setWarmupStrategy(String warmupStrategy) { this.warmupStrategy = warmupStrategy; }

    public HnswConfig getHnsw() { return hnsw; }
    public void setHnsw(HnswConfig hnsw) { this.hnsw = hnsw; }

    public static class HnswConfig {
        private int m = 16;
        private int efConstruction = 200;
        private int efRuntime = 100;

        public int getM() { return m; }
        public void setM(int m) { this.m = m; }

        public int getEfConstruction() { return efConstruction; }
        public void setEfConstruction(int efConstruction) { this.efConstruction = efConstruction; }

        public int getEfRuntime() { return efRuntime; }
        public void setEfRuntime(int efRuntime) { this.efRuntime = efRuntime; }
    }
}
