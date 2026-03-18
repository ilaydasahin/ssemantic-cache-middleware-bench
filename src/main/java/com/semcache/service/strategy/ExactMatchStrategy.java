package com.semcache.service.strategy;

import com.semcache.model.CacheEntry;
import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.CacheLookupStrategy;

/**
 * O(1) exact-match lookup via the normalised query index.
 */
public class ExactMatchStrategy implements CacheLookupStrategy {

    @Override
    public CacheLookupResult lookup(String query, CacheContext ctx) {
        long start = System.nanoTime();
        String id = ctx.queryIndex().get(ctx.normalize(query));

        if (id != null) {
            CacheEntry entry = ctx.entries().get(id);
            if (entry != null && !isExpired(entry, ctx.ttlMs())) {
                long lookupMs = (System.nanoTime() - start) / 1_000_000;
                return CacheLookupResult.hit(entry.response(), 1.0, lookupMs, 0,
                        entry.queryText(), entry.embedding());
            }
        }
        return CacheLookupResult.miss(0, null);
    }

    @Override
    public String strategyName() {
        return "EXACT_MATCH";
    }

    private boolean isExpired(CacheEntry entry, long ttlMs) {
        return System.currentTimeMillis() - entry.timestamp() > ttlMs;
    }
}
