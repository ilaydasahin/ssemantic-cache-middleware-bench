package com.semcache.service;

import com.semcache.model.CacheLookupResult;
import com.semcache.config.CacheProperties;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

public class SemanticCacheServiceTest {

    private SemanticCacheService cacheService;
    private EmbeddingService embeddingServiceMock;
    private RedisSearchService redisSearchServiceMock;
    private CacheProperties cachePropertiesMock;

    @BeforeEach
    public void setUp() {
        embeddingServiceMock = Mockito.mock(EmbeddingService.class);
        when(embeddingServiceMock.getModelName()).thenReturn("minilm");
        redisSearchServiceMock = Mockito.mock(RedisSearchService.class);
        cachePropertiesMock = Mockito.mock(CacheProperties.class);
        
        when(cachePropertiesMock.getTtlSeconds()).thenReturn(86400); // 24 hours
        when(cachePropertiesMock.getMaxEntries()).thenReturn(50000);
        
        MeterRegistry meterRegistry = new SimpleMeterRegistry();
        
        cacheService = new SemanticCacheService(
                cachePropertiesMock, 
                embeddingServiceMock, 
                redisSearchServiceMock, 
                meterRegistry);
        
        cacheService.setSimilarityThreshold(0.90);
        cacheService.setStrategy("SEMANTIC");
        cacheService.setWarmupStrategy("ORIGINAL");
        cacheService.init();
    }

    @Test
    public void testCacheHitWithExactMatch() {
        // Arrange
        String query = "What is the capital of France?";
        float[] queryEmbedding = new float[]{0.1f, 0.2f, 0.3f};
        String answer = "Paris";

        when(embeddingServiceMock.encode(query)).thenReturn(queryEmbedding);

        cacheService.store(query, queryEmbedding, answer);

        // Act
        CacheLookupResult result = cacheService.lookup(query);

        // Assert
        assertTrue(result.hit());
        assertEquals(answer, result.response());
        assertEquals(1.0, result.similarityScore(), 0.001); // Exact match
    }

    @Test
    public void testCacheHitWithSemanticSimilarity() {
        // Arrange
        String cachedQuery = "How do I fix a flat tire?";
        float[] cachedEmbedding = new float[]{0.9f, 0.1f, 0.1f};
        String answer = "Use a patch kit...";
        
        String newQuery = "What is the process to repair a punctured tire?";
        float[] newEmbedding = new float[]{0.85f, 0.15f, 0.1f};

        when(embeddingServiceMock.encode(cachedQuery)).thenReturn(cachedEmbedding);
        when(embeddingServiceMock.encode(newQuery)).thenReturn(newEmbedding);
        when(embeddingServiceMock.cosineSimilarity(any(float[].class), any(float[].class))).thenReturn(0.95);

        cacheService.store(cachedQuery, cachedEmbedding, answer);

        // Act
        CacheLookupResult result = cacheService.lookup(newQuery);

        // Assert
        assertTrue(result.hit());
        assertEquals(answer, result.response());
        assertEquals(0.95, result.similarityScore(), 0.001);
    }

    @Test
    public void testCacheMissWhenBelowThreshold() {
        // Arrange
        String cachedQuery = "How to bake a cake?";
        float[] cachedEmbedding = new float[]{0.1f, 0.9f, 0.1f};
        String answer = "Mix flour, sugar...";
        
        String newQuery = "How to bake cookies?";
        float[] newEmbedding = new float[]{0.2f, 0.8f, 0.1f};

        when(embeddingServiceMock.encode(cachedQuery)).thenReturn(cachedEmbedding);
        when(embeddingServiceMock.encode(newQuery)).thenReturn(newEmbedding);
        // Similarity below threshold 0.90
        when(embeddingServiceMock.cosineSimilarity(any(float[].class), any(float[].class))).thenReturn(0.85);

        cacheService.store(cachedQuery, cachedEmbedding, answer);

        // Act
        CacheLookupResult result = cacheService.lookup(newQuery);

        // Assert
        assertFalse(result.hit());
        assertNull(result.response());
    }

    @Test
    public void testEvictionClearsCache() throws InterruptedException {
        // Mock properties to return 1 sec TTL for testing
        when(cachePropertiesMock.getTtlSeconds()).thenReturn(1);
        when(cachePropertiesMock.getMaxEntries()).thenReturn(50000);
        
        MeterRegistry meterRegistry = new SimpleMeterRegistry();
        cacheService = new SemanticCacheService(
                cachePropertiesMock, 
                embeddingServiceMock, 
                redisSearchServiceMock, 
                meterRegistry);
        cacheService.setStrategy("SEMANTIC");
        cacheService.init();

        String query = "Test query";
        float[] queryEmbedding = new float[]{0.5f};
        when(embeddingServiceMock.encode(query)).thenReturn(queryEmbedding);

        cacheService.store(query, queryEmbedding, "Test Answer");
        assertEquals(1, cacheService.getCacheSize());

        // Wait for eviction 
        Thread.sleep(1200);

        // Perform any operation to trigger eviction checks if lazy, or wait for background task
        // We will just do a mock look-up to give enough time
        CacheLookupResult result = cacheService.lookup(query);

        // The exact timing might be flaky in CI, but testing the mechanism is good
        // In local environments this is enough. If flaky, consider using Awaitility.
        assertFalse(result.hit(), "Entry should have been evicted");
    }
}
