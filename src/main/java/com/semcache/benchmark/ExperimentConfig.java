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
                "--- EXPERIMENT CONFIGURATION ---%n" +
                " Experiment ID  : %s%n" +
                " Timestamp      : %s%n" +
                " JVM Version    : %s%n" +
                " Dataset        : %s (%s)%n" +
                " Embedding Model: %s%n" +
                " Threshold (θ)  : %.2f%n" +
                " Warmup Strategy: %s  (ratio=%.2f)%n" +
                " Random Seed    : %d%n" +
                " Sample Size    : %s%n" +
                " HNSW Enabled   : %s%n" +
                " Cache Strategy : %s  (k=%d)%n" +
                " Max Entries    : %d  TTL=%ds%n" +
                " Zipfian Skew   : %.2f%n" +
                " Noise Prob     : %.2f%n" +
                " Output File    : %s%n" +
                "--------------------------------",
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
