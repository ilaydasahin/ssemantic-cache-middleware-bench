package com.semcache.service.strategy;

import com.semcache.model.CacheEntry;
import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.CacheLookupStrategy;
import com.semcache.service.EmbeddingService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * GPTCache-inspired baseline strategy for Q1 publication comparison.
 * 
 * Implements a simplified version of GPTCache's approach:
 * 1. Embedding-based similarity search (like GPTCache)
 * 2. Single-tier cache (no cascade)
 * 3. Fixed threshold (0.9)
 * 
 * Reference: GPTCache (https://github.com/zilliztech/GPTCache)
 * 
 * This baseline allows fair comparison with state-of-the-art LLM caching systems.
 */
@Component
public class GPTCacheBaselineStrategy implements CacheLookupStrategy {
    
    private static final Logger log = LoggerFactory.getLogger(GPTCacheBaselineStrategy.class);
    private static final double GPTCACHE_THRESHOLD = 0.9;  // GPTCache default
    
    private final EmbeddingService embeddingService;
    
    public GPTCacheBaselineStrategy(EmbeddingService embeddingService) {
        this.embeddingService = embeddingService;
    }
    
    @Override
    public CacheLookupResult lookup(String query, CacheContext ctx) {
        long startNs = System.nanoTime();
        
        // Encode query
        float[] queryEmbedding = embeddingService.encode(query);
        long embeddingMs = (System.nanoTime() - startNs) / 1_000_000;
        
        // Search for similar entries
        double bestScore = 0.0;
        CacheEntry bestMatch = null;
        
        for (Map.Entry<String, CacheEntry> entry : ctx.entries().entrySet()) {
            CacheEntry cached = entry.getValue();
            
            // Skip expired entries
            if (System.currentTimeMillis() - cached.timestamp() > ctx.ttlMs()) {
                continue;
            }
            
            // Calculate cosine similarity
            double similarity = cosineSimilarity(queryEmbedding, cached.embedding());
            
            if (similarity > bestScore) {
                bestScore = similarity;
                bestMatch = cached;
            }
        }
        
        long totalMs = (System.nanoTime() - startNs) / 1_000_000;
        
        // Use GPTCache's fixed threshold
        if (bestMatch != null && bestScore >= GPTCACHE_THRESHOLD) {
            log.debug("GPTCache baseline HIT: query='{}', score={:.3f}", query, bestScore);
            return CacheLookupResult.hit(
                bestMatch.response(), 
                bestScore, 
                totalMs, 
                embeddingMs,
                bestMatch.queryText(), 
                queryEmbedding
            );
        }
        
        return CacheLookupResult.miss(embeddingMs, queryEmbedding);
    }
    
    @Override
    public String strategyName() {
        return "GPTCACHE_BASELINE";
    }
    
    private double cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) return 0.0;
        
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        if (normA == 0.0 || normB == 0.0) return 0.0;
        
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}

