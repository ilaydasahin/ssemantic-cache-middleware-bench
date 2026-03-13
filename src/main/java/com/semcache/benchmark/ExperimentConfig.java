package com.semcache.benchmark;

import java.time.Instant;
import java.util.UUID;

/**
 * Centralized, immutable configuration for a single experimental run.
 */
public record ExperimentConfig(
        String     experimentId,
        String     runTimestamp,
        String     javaVersion,
        String     datasetName,
        String     datasetPath,
        String     embeddingModelName,
        double     similarityThreshold,
        String     warmupStrategy,
        double     warmupRatio,
        long       randomSeed,
        Integer    sampleSize,
        boolean    hnswEnabled,
        String     cacheStrategy,
        int        knnK,
        int        maxCacheEntries,
        long       ttlSeconds,
        Integer    concurrentUsers,
        double     zipfianSkew,
        double     noiseProbability,
        String     outputFilePath
) {

    public static ExperimentConfig createV2(
            String  datasetName,
            String  datasetPath,
            String  embeddingModelName,
            double  similarityThreshold,
            String  warmupStrategy,
            double  warmupRatio,
            long    randomSeed,
            Integer sampleSize,
            boolean hnswEnabled,
            String  cacheStrategy,
            int     knnK,
            int     maxCacheEntries,
            long    ttlSeconds,
            Integer concurrentUsers,
            double  zipfianSkew,
            double  noiseProbability,
            String  outputFilePath) {

        return new ExperimentConfig(
                UUID.randomUUID().toString(),
                Instant.now().toString(),
                System.getProperty("java.version", "unknown"),
                datasetName,
                datasetPath,
                embeddingModelName,
                similarityThreshold,
                warmupStrategy,
                warmupRatio,
                randomSeed,
                sampleSize,
                hnswEnabled,
                cacheStrategy,
                knnK,
                maxCacheEntries,
                ttlSeconds,
                concurrentUsers,
                zipfianSkew,
                noiseProbability,
                outputFilePath);
    }

    public String toLogSummary() {
        return String.format(java.util.Locale.US,
                """
                ╔══════════════════════════════════════════════╗
                ║         EXPERIMENT CONFIGURATION             ║
                ╠══════════════════════════════════════════════╣
                ║ Experiment ID  : %s
                ║ Timestamp      : %s
                ║ JVM Version    : %s
                ║ Dataset        : %s (%s)
                ║ Embedding Model: %s
                ║ Threshold (θ)  : %.2f
                ║ Warmup Strategy: %s  (ratio=%.2f)
                ║ Random Seed    : %d
                ║ Sample Size    : %s
                ║ HNSW Enabled   : %s
                ║ Cache Strategy : %s  (k=%d)
                ║ Max Entries    : %d  TTL=%ds
                ║ Zipfian Skew   : %.2f
                ║ Noise Prob     : %.2f
                ║ Output File    : %s
                ╚══════════════════════════════════════════════╝""",
                experimentId, runTimestamp, javaVersion,
                datasetName, datasetPath,
                embeddingModelName,
                similarityThreshold,
                warmupStrategy, warmupRatio,
                randomSeed,
                sampleSize != null ? sampleSize.toString() : "full",
                hnswEnabled,
                cacheStrategy, knnK,
                maxCacheEntries, ttlSeconds,
                zipfianSkew,
                noiseProbability,
                outputFilePath);
    }
}
