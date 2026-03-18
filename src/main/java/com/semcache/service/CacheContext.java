package com.semcache.service;

import com.semcache.model.CacheEntry;

import java.util.Map;

/**
 * Read-only view of the cache state passed to each {@link CacheLookupStrategy}.
 *
 * Strategies receive this context instead of direct references to the internal
 * maps, keeping the data store encapsulated inside SemanticCacheService.
 */
public interface CacheContext {

    /** Unmodifiable snapshot of all live (non-expired) entries. */
    Map<String, CacheEntry> entries();

    /** O(1) exact-match map: normalizedQuery → entryId. */
    Map<String, String> queryIndex();

    /** Current similarity threshold (θ), may be overridden at runtime. */
    double similarityThreshold();

    /** Whether HNSW (RedisSearch) is enabled for this run. */
    boolean hnswEnabled();

    /** TTL in milliseconds. */
    long ttlMs();

    /** Normalise a query string the same way the cache does internally. */
    String normalize(String query);
}
