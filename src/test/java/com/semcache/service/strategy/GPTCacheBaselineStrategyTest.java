package com.semcache.service.strategy;

import com.semcache.model.CacheLookupResult;
import com.semcache.service.CacheContext;
import com.semcache.service.CacheLookupStrategy;
import com.semcache.service.EmbeddingService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class GPTCacheBaselineStrategyTest {

    @Mock
    private EmbeddingService embeddingService;

    @Mock
    private CacheContext context;

    private GPTCacheBaselineStrategy strategy;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        strategy = new GPTCacheBaselineStrategy(embeddingService);
    }

    @Test
    @DisplayName("Should return correct strategy name")
    void testStrategyName() {
        assertThat(strategy.strategyName()).isEqualTo("GPTCACHE_BASELINE");
    }

    @Test
    @DisplayName("Should miss when no cache entries exist")
    void testMissEmptyCache() {
        String query = "What is Java?";
        float[] embedding = new float[]{0.1f, 0.2f, 0.3f};
        
        when(embeddingService.encode(query)).thenReturn(embedding);
        when(context.entries()).thenReturn(java.util.Map.of());
        when(context.ttlMs()).thenReturn(3600000L);

        CacheLookupResult result = strategy.lookup(query, context);

        assertThat(result.hit()).isFalse();
    }
}

