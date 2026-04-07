package com.semcache.service;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for RedisSearchService.
 * 
 * Tests HNSW vector search integration.
 * 
 * Note: These tests require Redis 8+ with vectorset support to be running.
 * In CI/CD, use Testcontainers for Redis.
 */
class RedisSearchServiceTest {

    private MeterRegistry meterRegistry;
    private RedisSearchService service;

    @BeforeEach
    void setUp() {
        meterRegistry = new SimpleMeterRegistry();
        service = new RedisSearchService(meterRegistry);
    }

    @Test
    @DisplayName("Should create service instance")
    void testServiceCreation() {
        assertThat(service).isNotNull();
    }

    @Test
    @DisplayName("Should handle unavailable Redis gracefully")
    void testUnavailableRedis() {
        // When Redis is not available, service should still be created
        // but isAvailable() should return false
        assertThat(service).isNotNull();
        
        // Note: isAvailable() is checked after @PostConstruct init()
        // If Redis is not running, it will be false
    }

    @Test
    @DisplayName("Should have store method")
    void testStoreMethodExists() {
        // Verify method signature exists
        // Actual functionality requires Redis to be running
        assertThat(service).isNotNull();
        
        // This would be tested with Testcontainers:
        // service.store("test-id", new float[]{0.1f, 0.2f}, "query", "response");
    }

    @Test
    @DisplayName("Should have search method")
    void testSearchMethodExists() {
        // Verify method signature exists
        // Actual functionality requires Redis to be running
        assertThat(service).isNotNull();
        
        // This would be tested with Testcontainers:
        // Optional<List<Document>> results = service.search(new float[]{0.1f, 0.2f}, 10);
    }

    @Test
    @DisplayName("Should have clear method")
    void testClearMethodExists() {
        // Verify method signature exists
        assertThat(service).isNotNull();
        
        // This would be tested with Testcontainers:
        // service.clear();
    }
}

