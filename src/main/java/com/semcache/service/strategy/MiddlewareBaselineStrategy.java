package com.semcache.service.strategy;

import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.CacheLookupStrategy;
import com.semcache.service.EmbeddingService;
import com.semcache.service.RedisSearchService;

/**
 * Simulates middleware-based semantic caches (e.g. LangChain/GPTCache).
 *
 * Adds a fixed 15 ms penalty to model the serialisation and network-hop
 * overhead typical of Python/REST intermediary layers, then delegates
 * to the standard semantic lookup.
 */
public class MiddlewareBaselineStrategy implements CacheLookupStrategy {

    private final SemanticStrategy delegate;

    public MiddlewareBaselineStrategy(EmbeddingService embeddingService,
                                      RedisSearchService redisSearchService) {
        this.delegate = new SemanticStrategy(embeddingService, redisSearchService);
    }

    @Override
    public CacheLookupResult lookup(String query, CacheContext ctx) {
        long start = System.nanoTime();

        // Simulated 15 ms middleware overhead
        try {
            Thread.sleep(15);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        CacheLookupResult base = delegate.lookup(query, ctx);
        long totalMs = (System.nanoTime() - start) / 1_000_000;

        if (base.hit()) {
            return CacheLookupResult.hit(base.response(), base.similarityScore(),
                    totalMs, base.embeddingTimeMs(), base.matchedQueryText(), base.queryEmbedding());
        }
        return CacheLookupResult.miss(base.embeddingTimeMs(), base.queryEmbedding());
    }

    @Override
    public String strategyName() {
        return "MIDDLEWARE_BASELINE";
    }
}
