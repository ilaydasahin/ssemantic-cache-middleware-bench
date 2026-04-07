package com.semcache.service.strategy;

import com.semcache.model.CacheEntry;
import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.EmbeddingService;
import com.semcache.service.RedisSearchService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/**
 * Unit tests for MiddlewareBaselineStrategy.
 * 
 * Tests the baseline that adds 15ms overhead without actual caching.
 */
class MiddlewareBaselineStrategyTest {

    @Mock
    private EmbeddingService embeddingService;

    @Mock
    private RedisSearchService redisSearchService;

    @Mock
    private CacheContext context;

    private MiddlewareBaselineStrategy strategy;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        strategy = new MiddlewareBaselineStrategy(embeddingService, redisSearchService);
    }

    @Test
    @DisplayName("Should always miss (baseline has no cache)")
    void testAlwaysMiss() {
        // Arrange
        String query = "What is machine learning?";
        float[] embedding = new float[]{0.1f, 0.2f, 0.3f};
        
        Map<String, CacheEntry> emptyEntries = new HashMap<>();
        Map<String, String> emptyIndex = new HashMap<>();
        
        when(context.entries()).thenReturn(emptyEntries);
        when(context.queryIndex()).thenReturn(emptyIndex);
        when(context.similarityThreshold()).thenReturn(0.90);
        when(context.ttlMs()).thenReturn(3600000L);
        when(context.normalize(any())).thenReturn(query.toLowerCase());
        when(embeddingService.encode(any())).thenReturn(embedding);
        
        // Act
        CacheLookupResult result = strategy.lookup(query, context);
        
        // Assert
        assertThat(result.hit()).isFalse();
        assertThat(result.queryEmbedding()).isEqualTo(embedding);
    }

    @Test
    @DisplayName("Should add 15ms overhead")
    void testOverhead() {
        // Arrange
        String query = "Test query";
        float[] embedding = new float[]{0.1f, 0.2f, 0.3f};
        
        Map<String, CacheEntry> emptyEntries = new HashMap<>();
        Map<String, String> emptyIndex = new HashMap<>();
        
        when(context.entries()).thenReturn(emptyEntries);
        when(context.queryIndex()).thenReturn(emptyIndex);
        when(context.similarityThreshold()).thenReturn(0.90);
        when(context.ttlMs()).thenReturn(3600000L);
        when(context.normalize(any())).thenReturn(query.toLowerCase());
        when(embeddingService.encode(any())).thenReturn(embedding);
        
        // Act
        long start = System.nanoTime();
        CacheLookupResult result = strategy.lookup(query, context);
        long durationMs = (System.nanoTime() - start) / 1_000_000;
        
        // Assert
        assertThat(result.hit()).isFalse();
        assertThat(durationMs).isGreaterThanOrEqualTo(15); // Should have ~15ms overhead
    }

    @Test
    @DisplayName("Should return strategy name")
    void testStrategyName() {
        assertThat(strategy.strategyName()).isEqualTo("MIDDLEWARE_BASELINE");
    }
}
