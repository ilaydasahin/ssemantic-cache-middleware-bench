package com.semcache.service.strategy;

import com.semcache.model.CacheEntry;
import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.CacheLookupStrategy;
import com.semcache.service.EmbeddingService;

import java.util.Comparator;
import java.util.List;

/**
 * Hybrid Cascaded Lookup (§5.7):
 * Tier-1 fast recall with MiniLM, Tier-2 precision verification with MPNet.
 */
public class HybridCascadeStrategy implements CacheLookupStrategy {

    private final EmbeddingService embeddingService;

    public HybridCascadeStrategy(EmbeddingService embeddingService) {
        this.embeddingService = embeddingService;
    }

    @Override
    public CacheLookupResult lookup(String query, CacheContext ctx) {
        long start = System.nanoTime();

        float[] minilmVec = embeddingService.encode(query, "minilm");
        long embedTime = (System.nanoTime() - start) / 1_000_000;

        double threshold = ctx.similarityThreshold();

        List<ScoredEntry> candidates = ctx.entries().values().parallelStream()
                .map(entry -> {
                    float[] cached = entry.embeddings().get("minilm");
                    double sim = cached != null
                            ? embeddingService.cosineSimilarity(minilmVec, cached) : 0.0;
                    return new ScoredEntry(entry, sim);
                })
                .filter(r -> r.score >= threshold - 0.05)
                .sorted(Comparator.comparingDouble((ScoredEntry r) -> r.score).reversed())
                .limit(3)
                .toList();

        if (candidates.isEmpty()) {
            return CacheLookupResult.miss(embedTime, minilmVec);
        }

        // Strong hit — skip expensive verification
        if (candidates.get(0).score >= threshold + 0.05) {
            ScoredEntry hit = candidates.get(0);
            return CacheLookupResult.hit(hit.entry.response(), hit.score,
                    (System.nanoTime() - start) / 1_000_000, embedTime,
                    hit.entry.queryText(), minilmVec);
        }

        // Tier-2: MPNet verification
        float[] mpnetVec = embeddingService.encode(query, "mpnet");
        for (ScoredEntry candidate : candidates) {
            float[] cachedMPNet = candidate.entry.embeddings().get("mpnet");
            if (cachedMPNet != null) {
                double actualSim = embeddingService.cosineSimilarity(mpnetVec, cachedMPNet);
                if (actualSim >= threshold) {
                    return CacheLookupResult.hit(candidate.entry.response(), actualSim,
                            (System.nanoTime() - start) / 1_000_000, embedTime,
                            candidate.entry.queryText(), mpnetVec);
                }
            }
        }

        return CacheLookupResult.miss(embedTime, minilmVec);
    }

    @Override
    public String strategyName() {
        return "HYBRID";
    }

    private record ScoredEntry(CacheEntry entry, double score) {}
}
