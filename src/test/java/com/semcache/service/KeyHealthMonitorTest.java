package com.semcache.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for KeyHealthMonitor
 */
class KeyHealthMonitorTest {

    private KeyHealthMonitor monitor;

    @BeforeEach
    void setUp() {
        monitor = new KeyHealthMonitor();
    }

    @Test
    void testRecordSuccess() {
        monitor.recordSuccess("key1", 100);
        monitor.recordSuccess("key1", 200);
        
        assertThat(monitor.isKeyHealthy("key1")).isTrue();
    }

    @Test
    void testRecordError() {
        monitor.recordError("key1");
        
        assertThat(monitor.isKeyHealthy("key1")).isTrue(); // Still healthy with 1 error
    }

    @Test
    void testAutoDisableOnHighErrorRate() {
        // Record 10 errors to trigger auto-disable
        for (int i = 0; i < 10; i++) {
            monitor.recordError("key1");
        }
        
        assertThat(monitor.isKeyHealthy("key1")).isFalse();
    }

    @Test
    void testSuccessRateCalculation() {
        monitor.recordSuccess("key1", 100);
        monitor.recordSuccess("key1", 100);
        monitor.recordError("key1");
        
        // 2 success, 1 error = 66.67% success rate
        assertThat(monitor.isKeyHealthy("key1")).isTrue();
    }

    @Test
    void testNewKeyIsHealthy() {
        assertThat(monitor.isKeyHealthy("new-key")).isTrue();
    }
}
