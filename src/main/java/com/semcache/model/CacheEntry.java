package com.semcache.model;

import java.util.Map;

/**
 * Represents a cache entry containing multiple query embeddings, text, and LLM response.
 * Support for multi-vector storage allows Hybrid Caching (§5.7).
 */
public record CacheEntry(
    String id,
    Map<String, float[]> embeddings,
    String queryText,
    String response,
    long timestamp,
    int hitCount
) {
    public CacheEntry withIncrementedHitCount() {
        return new CacheEntry(id, embeddings, queryText, response, timestamp, hitCount + 1);
    }

    /**
     * Legacy compatibility helper
     */
    public float[] embedding() {
        if (embeddings == null || embeddings.isEmpty()) return null;
        return embeddings.values().iterator().next();
    }
}
