package com.semcache.model;

/**
 * Result of a cache lookup operation.
 */
public record CacheLookupResult(
    boolean hit,
    String response,
    double similarityScore,
    long lookupTimeMs,
    long embeddingTimeMs,
    String matchedQueryText,
    float[] queryEmbedding
) {
    public static CacheLookupResult miss(long embeddingTimeMs, float[] queryEmbedding) {
        return new CacheLookupResult(false, null, 0.0, 0, embeddingTimeMs, null, queryEmbedding);
    }
    
    public static CacheLookupResult hit(String response, double similarity, 
                                         long lookupTimeMs, long embeddingTimeMs,
                                         String matchedQuery, float[] queryEmbedding) {
        return new CacheLookupResult(true, response, similarity, lookupTimeMs, 
                                     embeddingTimeMs, matchedQuery, queryEmbedding);
    }
}
