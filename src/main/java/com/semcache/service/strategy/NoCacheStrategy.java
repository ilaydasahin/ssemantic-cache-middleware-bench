package com.semcache.service.strategy;

import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.CacheLookupStrategy;
import lombok.extern.slf4j.Slf4j;

/**
 * NO_CACHE Baseline Strategy - Q1 Publication Requirement
 *
 * <p>This strategy ALWAYS returns a cache miss, forcing all queries to go to the LLM.
 * It serves as the control baseline for measuring cache effectiveness.
 *
 * <p><b>Purpose:</b>
 * <ul>
 *   <li>Establish baseline performance (100% LLM calls)</li>
 *   <li>Measure absolute cost savings of caching</li>
 *   <li>Validate that cache overhead is minimal</li>
 *   <li>Required for Q1 publication standards</li>
 * </ul>
 *
 * <p><b>Expected Results:</b>
 * <ul>
 *   <li>Hit Rate: 0%</li>
 *   <li>Latency: Maximum (all LLM calls)</li>
 *   <li>Cost: Maximum (no savings)</li>
 *   <li>Throughput: Minimum (LLM bottleneck)</li>
 * </ul>
 *
 * <p><b>Q1 Reviewer Expectations:</b>
 * Reviewers will compare all caching strategies against this baseline to validate:
 * <ol>
 *   <li>Hit rate improvements are real (not artifacts)</li>
 *   <li>Latency reductions are significant</li>
 *   <li>Cost savings justify cache overhead</li>
 *   <li>System adds value over no caching</li>
 * </ol>
 *
 * @see com.semcache.model.CacheStrategy#NONE
 * @since 1.0.0 (Q1 Publication Update)
 */
@Slf4j
public class NoCacheStrategy implements CacheLookupStrategy {

    /**
     * Always returns a cache miss.
     *
     * <p>This method intentionally does NOT check the cache, ensuring that:
     * <ul>
     *   <li>All queries go to the LLM</li>
     *   <li>No cache overhead is incurred</li>
     *   <li>Baseline performance is accurately measured</li>
     * </ul>
     *
     * @param query   the incoming query (ignored)
     * @param context the cache context (ignored)
     * @return always a MISS result with null cached response
     */
    @Override
    public CacheLookupResult lookup(String query, CacheContext context) {
        // Log for debugging (can be disabled in production)
        if (log.isTraceEnabled()) {
            log.trace("[NO_CACHE] Query bypassing cache: {}", 
                     query.length() > 50 ? query.substring(0, 50) + "..." : query);
        }

        // Always return miss - force LLM call
        return CacheLookupResult.miss();
    }

    @Override
    public String strategyName() {
        return "NO_CACHE";
    }

    /**
     * Returns a human-readable description for logging and reporting.
     */
    @Override
    public String toString() {
        return "NoCacheStrategy{baseline=true, hitRate=0%, purpose=control}";
    }
}
