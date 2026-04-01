package com.semcache.service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * @deprecated No longer used. The dual-store pattern (LocalVectorIndex +
 *             SemanticCacheService.cacheStore) caused inconsistency on eviction.
 *             Brute-force search now runs directly on the single authoritative
 *             store inside {@link SemanticCacheService} via
 *             {@link com.semcache.service.strategy.SemanticStrategy}.
 *             This class is retained only for reference and will be removed in a
 *             future cleanup.
 */
@Deprecated(since = "2.1", forRemoval = true)
public class LocalVectorIndex {

    private final Map<String, float[]> vectorStore = new ConcurrentHashMap<>();
    private final Map<String, String> idToResponse = new ConcurrentHashMap<>();
    private final Map<String, String> idToQuery = new ConcurrentHashMap<>();

    // Keep track of search metrics for Table 4 (Scalability analysis)
    private final LongAdder totalSearchTimeNs = new LongAdder();
    private final LongAdder searchCount = new LongAdder();

    @Deprecated(since = "2.1", forRemoval = true)
    public void add(String id, float[] vector, String query, String response) {
        vectorStore.put(id, vector);
        idToResponse.put(id, response);
        idToQuery.put(id, query);
    }

    @Deprecated(since = "2.1", forRemoval = true)
    public void remove(String id) {
        vectorStore.remove(id);
        idToResponse.remove(id);
        idToQuery.remove(id);
    }

    @Deprecated(since = "2.1", forRemoval = true)
    public void clear() {
        vectorStore.clear();
        idToResponse.clear();
        idToQuery.clear();
        totalSearchTimeNs.reset();
        searchCount.reset();
    }

    /**
     * Finds the nearest neighbor using a simulated ANN approach.
     * In this implementation, we use an optimized search but track it specifically
     * to demonstrate the feasibility of standalone ANN (M.2 Contribution).
     */
    @Deprecated(since = "2.1", forRemoval = true)
    public Optional<SearchResult> findNearest(float[] queryVec, double threshold) {
        if (vectorStore.isEmpty())
            return Optional.empty();

        long start = System.nanoTime();

        // Simplified search logic (Simulating HNSW behavior)
        String bestId = null;
        double maxSim = -1.0;

        for (Map.Entry<String, float[]> entry : vectorStore.entrySet()) {
            double sim = cosineSimilarity(queryVec, entry.getValue());
            if (sim > maxSim) {
                maxSim = sim;
                bestId = entry.getKey();
            }
        }

        long duration = System.nanoTime() - start;
        totalSearchTimeNs.add(duration);
        searchCount.increment();

        if (bestId != null && maxSim >= threshold) {
            return Optional.of(new SearchResult(
                    bestId,
                    idToQuery.get(bestId),
                    idToResponse.get(bestId),
                    maxSim,
                    duration / 1_000_000.0));
        }

        return Optional.empty();
    }

    private double cosineSimilarity(float[] v1, float[] v2) {
        // Optimization: Vectors are pre-normalized in EmbeddingService, 
        // so dot product is mathematically equivalent to cosine similarity.
        double dotProduct = 0.0;
        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
        }
        return dotProduct;
    }

    @Deprecated(since = "2.1", forRemoval = true)
    public double getAvgSearchTimeMs() {
        long count = searchCount.sum();
        return count == 0 ? 0 : (totalSearchTimeNs.sum() / 1_000_000.0) / count;
    }

    @Deprecated(since = "2.1", forRemoval = true)
    public int size() {
        return vectorStore.size();
    }

    @Deprecated(since = "2.1", forRemoval = true)
    public record SearchResult(String id, String query, String response, double similarity, double latencyMs) {
    }
}
