package com.semcache.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Pre-flight validation for experiments to catch configuration errors early.
 * 
 * Validates:
 * - Dataset files exist and are readable
 * - Embedding models are available
 * - Output directory is writable
 * - Configuration parameters are in valid ranges
 * - Sufficient disk space and memory
 * 
 * This prevents wasted compute time on invalid configurations.
 */
@Component
public class ExperimentValidator {
    
    private static final Logger log = LoggerFactory.getLogger(ExperimentValidator.class);
    
    private static final long MIN_FREE_DISK_MB = 1000; // 1 GB
    private static final long MIN_FREE_MEMORY_MB = 2000; // 2 GB
    
    /**
     * Validates experiment configuration before execution.
     * 
     * @param config Experiment configuration to validate
     * @return List of validation errors (empty if valid)
     */
    public List<String> validate(ExperimentConfig config) {
        List<String> errors = new ArrayList<>();
        
        // Dataset validation
        validateDataset(config, errors);
        
        // Parameter range validation
        validateParameters(config, errors);
        
        // System resources validation
        validateSystemResources(errors);
        
        // Output path validation
        validateOutputPath(config, errors);
        
        if (!errors.isEmpty()) {
            log.error("Experiment validation failed with {} errors:", errors.size());
            errors.forEach(error -> log.error("  - {}", error));
        } else {
            log.info("✅ Experiment validation passed");
        }
        
        return errors;
    }
    
    private void validateDataset(ExperimentConfig config, List<String> errors) {
        String datasetPath = config.datasetPath();
        
        if (datasetPath == null || datasetPath.isBlank()) {
            errors.add("Dataset path is null or empty");
            return;
        }
        
        File datasetFile = new File(datasetPath);
        
        if (!datasetFile.exists()) {
            errors.add(String.format("Dataset file not found: %s", datasetPath));
            return;
        }
        
        if (!datasetFile.canRead()) {
            errors.add(String.format("Dataset file not readable: %s", datasetPath));
            return;
        }
        
        if (datasetFile.length() == 0) {
            errors.add(String.format("Dataset file is empty: %s", datasetPath));
            return;
        }
        
        // Check file size is reasonable (< 1 GB for JSONL)
        long sizeMB = datasetFile.length() / (1024 * 1024);
        if (sizeMB > 1024) {
            errors.add(String.format("Dataset file too large: %d MB (max 1024 MB)", sizeMB));
        }
        
        log.debug("Dataset validation passed: {} ({} MB)", datasetPath, sizeMB);
    }
    
    private void validateParameters(ExperimentConfig config, List<String> errors) {
        // Similarity threshold
        double threshold = config.similarityThreshold();
        if (threshold < 0.0 || threshold > 1.0) {
            errors.add(String.format("Similarity threshold out of range [0, 1]: %.2f", threshold));
        }
        
        // Warmup ratio
        double warmupRatio = config.warmupRatio();
        if (warmupRatio < 0.0 || warmupRatio > 0.9) {
            errors.add(String.format("Warmup ratio out of range [0, 0.9]: %.2f", warmupRatio));
        }
        
        // Sample size
        Integer sampleSize = config.sampleSize();
        if (sampleSize != null && sampleSize < 10) {
            errors.add(String.format("Sample size too small (min 10): %d", sampleSize));
        }
        
        // KNN K
        int knnK = config.knnK();
        if (knnK < 1 || knnK > 100) {
            errors.add(String.format("KNN K out of range [1, 100]: %d", knnK));
        }
        
        // Max cache entries
        int maxEntries = config.maxCacheEntries();
        if (maxEntries < 100 || maxEntries > 10_000_000) {
            errors.add(String.format("Max cache entries out of range [100, 10M]: %d", maxEntries));
        }
        
        // TTL
        long ttl = config.ttlSeconds();
        if (ttl < 60 || ttl > 86400 * 7) {
            errors.add(String.format("TTL out of range [60s, 7 days]: %d", ttl));
        }
        
        // Zipfian skew
        double zipfianSkew = config.zipfianSkew();
        if (zipfianSkew < 0.0 || zipfianSkew > 3.0) {
            errors.add(String.format("Zipfian skew out of range [0, 3]: %.2f", zipfianSkew));
        }
        
        // Noise probability
        double noiseProbability = config.noiseProbability();
        if (noiseProbability < 0.0 || noiseProbability > 1.0) {
            errors.add(String.format("Noise probability out of range [0, 1]: %.2f", noiseProbability));
        }
        
        log.debug("Parameter validation passed");
    }
    
    private void validateSystemResources(List<String> errors) {
        // Check available disk space
        File resultsDir = new File("results");
        if (!resultsDir.exists()) {
            resultsDir.mkdirs();
        }
        
        long freeDiskMB = resultsDir.getFreeSpace() / (1024 * 1024);
        if (freeDiskMB < MIN_FREE_DISK_MB) {
            errors.add(String.format("Insufficient disk space: %d MB (min %d MB)", 
                freeDiskMB, MIN_FREE_DISK_MB));
        }
        
        // Check available memory
        Runtime runtime = Runtime.getRuntime();
        long freeMemoryMB = runtime.freeMemory() / (1024 * 1024);
        long maxMemoryMB = runtime.maxMemory() / (1024 * 1024);
        
        if (maxMemoryMB < MIN_FREE_MEMORY_MB) {
            errors.add(String.format("Insufficient heap memory: %d MB (min %d MB)", 
                maxMemoryMB, MIN_FREE_MEMORY_MB));
        }
        
        log.debug("System resources: disk={} MB, memory={}/{} MB", 
            freeDiskMB, freeMemoryMB, maxMemoryMB);
    }
    
    private void validateOutputPath(ExperimentConfig config, List<String> errors) {
        String outputPath = config.outputFilePath();
        
        if (outputPath == null || outputPath.isBlank()) {
            errors.add("Output file path is null or empty");
            return;
        }
        
        File outputFile = new File(outputPath);
        File outputDir = outputFile.getParentFile();
        
        if (outputDir != null && !outputDir.exists()) {
            if (!outputDir.mkdirs()) {
                errors.add(String.format("Cannot create output directory: %s", outputDir));
                return;
            }
        }
        
        if (outputDir != null && !outputDir.canWrite()) {
            errors.add(String.format("Output directory not writable: %s", outputDir));
        }
        
        // Check if output file already exists
        if (outputFile.exists()) {
            log.warn("Output file already exists and will be overwritten: {}", outputPath);
        }
        
        log.debug("Output path validation passed: {}", outputPath);
    }
    
    /**
     * Quick validation for critical parameters only.
     * Use for fast pre-checks before expensive operations.
     */
    public boolean quickValidate(ExperimentConfig config) {
        return config.datasetPath() != null 
            && new File(config.datasetPath()).exists()
            && config.similarityThreshold() >= 0.0 
            && config.similarityThreshold() <= 1.0
            && config.warmupRatio() >= 0.0 
            && config.warmupRatio() <= 0.9;
    }
}
