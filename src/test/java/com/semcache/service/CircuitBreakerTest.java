package com.semcache.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for CircuitBreaker.
 * 
 * Tests the circuit breaker pattern for LLM API resilience.
 */
class CircuitBreakerTest {

    private CircuitBreaker circuitBreaker;

    @BeforeEach
    void setUp() {
        // failureThreshold=3, timeout=1s, successThreshold=2
        circuitBreaker = new CircuitBreaker(3, Duration.ofSeconds(1), 2);
    }

    @Test
    @DisplayName("Should start in CLOSED state")
    void testInitialState() {
        assertThat(circuitBreaker.getState()).isEqualTo("CLOSED");
        assertThat(circuitBreaker.allowRequest()).isTrue();
    }

    @Test
    @DisplayName("Should open after threshold failures")
    void testOpenAfterFailures() {
        // Record 3 failures
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        
        assertThat(circuitBreaker.getState()).isEqualTo("OPEN");
        assertThat(circuitBreaker.allowRequest()).isFalse();
    }

    @Test
    @DisplayName("Should transition to HALF_OPEN after timeout")
    void testHalfOpenAfterTimeout() throws InterruptedException {
        // Open circuit
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        
        assertThat(circuitBreaker.getState()).isEqualTo("OPEN");
        
        // Wait for timeout
        Thread.sleep(1100);
        
        // Should allow request (transition to HALF_OPEN)
        assertThat(circuitBreaker.allowRequest()).isTrue();
        assertThat(circuitBreaker.getState()).isEqualTo("HALF_OPEN");
    }

    @Test
    @DisplayName("Should close after successful recovery")
    void testCloseAfterRecovery() throws InterruptedException {
        // Open circuit
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        
        // Wait for timeout
        Thread.sleep(1100);
        circuitBreaker.allowRequest(); // Transition to HALF_OPEN
        
        // Record 2 successes (threshold)
        circuitBreaker.recordSuccess();
        circuitBreaker.recordSuccess();
        
        assertThat(circuitBreaker.getState()).isEqualTo("CLOSED");
        assertThat(circuitBreaker.getFailureCount()).isEqualTo(0);
    }

    @Test
    @DisplayName("Should reopen on failure during HALF_OPEN")
    void testReopenOnHalfOpenFailure() throws InterruptedException {
        // Open circuit
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        
        // Wait for timeout
        Thread.sleep(1100);
        circuitBreaker.allowRequest(); // Transition to HALF_OPEN
        
        // Fail during recovery
        circuitBreaker.recordFailure();
        
        assertThat(circuitBreaker.getState()).isEqualTo("OPEN");
    }

    @Test
    @DisplayName("Should reset failure count on success in CLOSED state")
    void testResetOnSuccess() {
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        
        assertThat(circuitBreaker.getFailureCount()).isEqualTo(2);
        
        circuitBreaker.recordSuccess();
        
        assertThat(circuitBreaker.getFailureCount()).isEqualTo(0);
    }

    @Test
    @DisplayName("Should allow manual reset")
    void testManualReset() {
        // Open circuit
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        circuitBreaker.recordFailure();
        
        assertThat(circuitBreaker.getState()).isEqualTo("OPEN");
        
        // Manual reset
        circuitBreaker.reset();
        
        assertThat(circuitBreaker.getState()).isEqualTo("CLOSED");
        assertThat(circuitBreaker.getFailureCount()).isEqualTo(0);
        assertThat(circuitBreaker.allowRequest()).isTrue();
    }
}
