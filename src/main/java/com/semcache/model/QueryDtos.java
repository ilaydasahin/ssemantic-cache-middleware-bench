package com.semcache.model;

/**
 * API request and response DTOs.
 */
public class QueryDtos {

    public record QueryRequest(
        String query,
        String sessionId
    ) {}

    public record QueryResponse(
        String query,
        String response,
        boolean cacheHit,
        double similarityScore,
        long totalLatencyMs,
        long embeddingLatencyMs,
        long llmLatencyMs,
        String cacheStrategy
    ) {}

    public record BenchmarkResult(
        String dataset,
        String embeddingModel,
        double threshold,
        String warmupStrategy,
        int totalQueries,
        int cacheHits,
        int cacheMisses,
        double hitRate,
        double avgBertScore,
        double avgRougeL,
        double costSavingsPercent,
        double p50LatencyMs,
        double p95LatencyMs,
        double p99LatencyMs,
        double avgEmbeddingLatencyMs,
        double avgLlmLatencyMs,
        double memoryUsageMb,
        int seed,
        int sampleSize
    ) {}
}
