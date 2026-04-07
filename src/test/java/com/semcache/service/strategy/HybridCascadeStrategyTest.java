package com.semcache.service.strategy;

import com.semcache.model.CacheEntry;
import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.EmbeddingService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

/**
 * Unit tests for HybridCascadeStrategy.
 * 
 * Tests the two-tier cascade: MiniLM (fast) → MPNet (accurate).
 */
class HybridCascadeStrategyTest {

    @Mock
    private EmbeddingService embeddingService;

    @Mock
    private CacheContext context;

    private HybridCascadeStrategy strategy;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        strategy = new HybridCascadeStrategy(embeddingService);
    }

    @Test
    @DisplayName("Should hit on L1 (MiniLM) when similarity exceeds threshold")
    void testL1Hit() {
        // Arrange
        String query = "What is machine learning?";
        float[] queryEmbedding = new float[]{0.1f, 0.2f, 0.3f};
        
        Map<String, float[]> embeddings = new HashMap<>();
        embeddings.put("minilm", new float[]{0.1f, 0.2f, 0.3f}); // Perfect match
        embeddings.put("mpnet", new float[]{0.0f, 0.0f, 0.0f});
        
        CacheEntry entry = new CacheEntry(
            "id1", embeddings, "What is ML?", "ML is...", 
            System.currentTimeMillis(), 0
        );
        
        Map<String, CacheEntry> entries = Map.of("id1", entry);
        Map<String, String> queryIndex = Map.of();
        
        when(context.entries()).thenReturn(entries);
        when(context.queryIndex()).thenReturn(queryIndex);
        when(context.similarityThreshold()).thenReturn(0.90);
        when(context.ttlMs()).thenReturn(3600000L);
        when(context.normalize(any())).thenReturn(query.toLowerCase());
        
        when(embeddingService.encode(eq(query), eq("minilm"))).thenReturn(queryEmbedding);
        when(embeddingService.cosineSimilarity(any(), any())).thenReturn(0.95);
        
        // Act
        CacheLookupResult result = strategy.lookup(query, context);
        
        // Assert
        assertThat(result.hit()).isTrue();
        assertThat(result.response()).isEqualTo("ML is...");
        assertThat(result.similarityScore()).isGreaterThanOrEqualTo(0.90);
    }

    @Test
    @DisplayName("Should cascade to L2 (MPNet) when L1 misses")
    void testL2Cascade() {
        // Arrange
        String query = "Explain deep learning";
        float[] miniLMEmbedding = new float[]{0.1f, 0.2f, 0.3f};
        float[] mpnetEmbedding = new float[]{0.5f, 0.6f, 0.7f};
        
        Map<String, float[]> embeddings = new HashMap<>();
        embeddings.put("minilm", new float[]{0.0f, 0.0f, 0.1f}); // Low similarity with query
        embeddings.put("mpnet", new float[]{0.5f, 0.6f, 0.7f}); // High similarity with query
        
        CacheEntry entry = new CacheEntry(
            "id1", embeddings, "What is deep learning?", "Deep learning is...", 
            System.currentTimeMillis(), 0
        );
        
        Map<String, CacheEntry> entries = Map.of("id1", entry);
        
        when(context.entries()).thenReturn(entries);
        when(context.queryIndex()).thenReturn(Map.of());
        when(context.similarityThreshold()).thenReturn(0.90);
        when(context.ttlMs()).thenReturn(3600000L);
        when(context.normalize(any())).thenReturn(query.toLowerCase());
        
        when(embeddingService.encode(eq(query), eq("minilm"))).thenReturn(miniLMEmbedding);
        when(embeddingService.encode(eq(query), eq("mpnet"))).thenReturn(mpnetEmbedding);
        
        // MiniLM returns medium similarity (0.85 - within threshold-0.05 range, so it's a candidate)
        when(embeddingService.cosineSimilarity(miniLMEmbedding, embeddings.get("minilm")))
            .thenReturn(0.85);
        
        // MPNet returns high similarity (hit)
        when(embeddingService.cosineSimilarity(mpnetEmbedding, embeddings.get("mpnet")))
            .thenReturn(0.95);
        
        // Act
        CacheLookupResult result = strategy.lookup(query, context);
        
        // Assert
        assertThat(result.hit()).isTrue();
        assertThat(result.response()).isEqualTo("Deep learning is...");
        assertThat(result.similarityScore()).isGreaterThanOrEqualTo(0.90);
    }

    @Test
    @DisplayName("Should miss when both L1 and L2 are below threshold")
    void testBothTiersMiss() {
        // Arrange
        String query = "Unrelated query";
        float[] miniLMEmbedding = new float[]{0.1f, 0.2f, 0.3f};
        float[] mpnetEmbedding = new float[]{0.5f, 0.6f, 0.7f};
        
        Map<String, float[]> embeddings = new HashMap<>();
        embeddings.put("minilm", new float[]{0.9f, 0.8f, 0.7f});
        embeddings.put("mpnet", new float[]{0.1f, 0.1f, 0.1f});
        
        CacheEntry entry = new CacheEntry(
            "id1", embeddings, "Cached query", "Cached response", 
            System.currentTimeMillis(), 0
        );
        
        when(context.entries()).thenReturn(Map.of("id1", entry));
        when(context.queryIndex()).thenReturn(Map.of());
        when(context.similarityThreshold()).thenReturn(0.90);
        when(context.ttlMs()).thenReturn(3600000L);
        when(context.normalize(any())).thenReturn(query.toLowerCase());
        
        when(embeddingService.encode(eq(query), eq("minilm"))).thenReturn(miniLMEmbedding);
        when(embeddingService.encode(eq(query), eq("mpnet"))).thenReturn(mpnetEmbedding);
        when(embeddingService.cosineSimilarity(any(), any())).thenReturn(0.60); // Below threshold
        
        // Act
        CacheLookupResult result = strategy.lookup(query, context);
        
        // Assert
        assertThat(result.hit()).isFalse();
        assertThat(result.queryEmbedding()).isNotNull();
    }

    @Test
    @DisplayName("Should return strategy name")
    void testStrategyName() {
        assertThat(strategy.strategyName()).isEqualTo("HYBRID");
    }
}
