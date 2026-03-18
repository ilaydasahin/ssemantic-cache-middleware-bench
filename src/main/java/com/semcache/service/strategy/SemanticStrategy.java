package com.semcache.service.strategy;

import com.semcache.model.CacheEntry;
import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.CacheLookupStrategy;
import com.semcache.service.EmbeddingService;
import com.semcache.service.RedisSearchService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.search.Document;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Semantic lookup: L1 exact-match shortcut → HNSW (Redis) → brute-force fallback.
 *
 * Implements Algorithm 1 from the paper (§3.1).
 */
public class SemanticStrategy implements CacheLookupStrategy {

    private static final Logger log = LoggerFactory.getLogger(SemanticStrategy.class);

    private final EmbeddingService embeddingService;
    private final RedisSearchService redisSearchService;

    public SemanticStrategy(EmbeddingService embeddingService, RedisSearchService redisSearchService) {
        this.embeddingService = embeddingService;
        this.redisSearchService = redisSearchService;
    }

    @Override
    public CacheLookupResult lookup(String query, CacheContext ctx) {
        // --- L1: O(1) exact-match shortcut ---
        String existingId = ctx.queryIndex().get(ctx.normalize(query));
        if (existingId != null) {
            CacheEntry entry = ctx.entries().get(existingId);
            if (entry != null && !isExpired(entry, ctx.ttlMs())) {
                return CacheLookupResult.hit(entry.response(), 1.0, 0, 0,
                        entry.queryText(), entry.embedding());
            }
        }

        // --- L2: Embedding + ANN search ---
        long embedStart = System.nanoTime();
        float[] queryVec = embeddingService.encode(query);
        long embeddingTimeMs = (System.nanoTime() - embedStart) / 1_000_000;

        if (ctx.entries().isEmpty()) {
            return CacheLookupResult.miss(embeddingTimeMs, queryVec);
        }

        long lookupStart = System.nanoTime();

        if (!ctx.hnswEnabled()) {
            return bruteForceLookup(queryVec, lookupStart, embeddingTimeMs, ctx);
        }

        // --- HNSW via RedisSearch ---
        if (redisSearchService.isAvailable()) {
            try {
                Optional<List<Document>> results = redisSearchService.search(queryVec, 1);
                if (results.isPresent() && !results.get().isEmpty()) {
                    Document doc = results.get().get(0);
                    double similarity = doc.getScore();
                    if (similarity >= ctx.similarityThreshold()) {
                        long totalLookupMs = (System.nanoTime() - lookupStart) / 1_000_000;
                        return CacheLookupResult.hit(doc.getString("response"), similarity,
                                totalLookupMs, embeddingTimeMs, doc.getString("query"), queryVec);
                    }
                }
            } catch (Exception e) {
                log.warn("RedisSearch failed: {}. Falling back to brute-force.", e.getMessage());
            }
        }

        return bruteForceLookup(queryVec, lookupStart, embeddingTimeMs, ctx);
    }

    @Override
    public String strategyName() {
        return "SEMANTIC";
    }

    private CacheLookupResult bruteForceLookup(float[] queryVec, long lookupStart,
                                                long embeddingTimeMs, CacheContext ctx) {
        long now = System.currentTimeMillis();
        long ttlMs = ctx.ttlMs();

        var best = ctx.entries().values().parallelStream()
                .filter(e -> now - e.timestamp() <= ttlMs)
                .map(e -> new ScoredEntry(e, embeddingService.cosineSimilarity(queryVec, e.embedding())))
                .max(Comparator.comparingDouble(r -> r.score));

        if (best.isPresent() && best.get().score >= ctx.similarityThreshold()) {
            long totalLookupMs = (System.nanoTime() - lookupStart) / 1_000_000;
            ScoredEntry hit = best.get();
            return CacheLookupResult.hit(hit.entry.response(), hit.score,
                    totalLookupMs, embeddingTimeMs, hit.entry.queryText(), queryVec);
        }

        return CacheLookupResult.miss(embeddingTimeMs, queryVec);
    }

    private boolean isExpired(CacheEntry entry, long ttlMs) {
        return System.currentTimeMillis() - entry.timestamp() > ttlMs;
    }

    private record ScoredEntry(CacheEntry entry, double score) {}
}
