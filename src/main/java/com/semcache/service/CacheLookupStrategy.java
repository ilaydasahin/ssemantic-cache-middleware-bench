package com.semcache.service;

import com.semcache.model.CacheLookupResult;

/**
 * Strategy interface for semantic cache lookup algorithms.
 *
 * Each implementation encapsulates a distinct lookup approach
 * (semantic, exact-match, hybrid, etc.) so that SemanticCacheService
 * is responsible only for orchestration, not for lookup logic.
 */
public interface CacheLookupStrategy {

    /**
     * Perform a cache lookup for the given query.
     *
     * @param query     the incoming query text
     * @param context   shared read-only view of the cache state
     * @return a hit or miss result
     */
    CacheLookupResult lookup(String query, CacheContext context);

    /** Identifies this strategy for logging and configuration matching. */
    String strategyName();
}
