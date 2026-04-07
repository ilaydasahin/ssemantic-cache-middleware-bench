package com.semcache.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Circuit Breaker pattern implementation for LLM API calls.
 * 
 * Prevents cascading failures by temporarily blocking requests when error rate exceeds threshold.
 * 
 * States:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Too many failures, requests blocked immediately
 * - HALF_OPEN: Testing if service recovered, limited requests allowed
 * 
 * Configuration:
 * - failureThreshold: Number of failures before opening circuit (default: 5)
 * - timeout: Duration to wait before attempting recovery (default: 60s)
 * - successThreshold: Successes needed in HALF_OPEN to close circuit (default: 2)
 */
public class CircuitBreaker {
    
    private static final Logger log = LoggerFactory.getLogger(CircuitBreaker.class);
    
    private enum State { CLOSED, OPEN, HALF_OPEN }
    
    private final int failureThreshold;
    private final Duration timeout;
    private final int successThreshold;
    
    private final AtomicInteger failureCount = new AtomicInteger(0);
    private final AtomicInteger successCount = new AtomicInteger(0);
    private final AtomicReference<State> state = new AtomicReference<>(State.CLOSED);
    private final AtomicReference<Instant> lastFailureTime = new AtomicReference<>(Instant.now());
    
    public CircuitBreaker(int failureThreshold, Duration timeout, int successThreshold) {
        this.failureThreshold = failureThreshold;
        this.timeout = timeout;
        this.successThreshold = successThreshold;
    }
    
    public CircuitBreaker() {
        this(5, Duration.ofSeconds(60), 2);
    }
    
    /**
     * Checks if request should be allowed through.
     * 
     * @return true if request can proceed, false if circuit is open
     */
    public boolean allowRequest() {
        State currentState = state.get();
        
        if (currentState == State.OPEN) {
            // Check if timeout has elapsed
            if (Duration.between(lastFailureTime.get(), Instant.now()).compareTo(timeout) > 0) {
                log.info("Circuit breaker transitioning from OPEN to HALF_OPEN");
                state.set(State.HALF_OPEN);
                successCount.set(0);
                return true;
            }
            return false;
        }
        
        return true;
    }
    
    /**
     * Records a successful request.
     */
    public void recordSuccess() {
        State currentState = state.get();
        
        if (currentState == State.HALF_OPEN) {
            int successes = successCount.incrementAndGet();
            if (successes >= successThreshold) {
                log.info("Circuit breaker transitioning from HALF_OPEN to CLOSED (successes: {})", successes);
                state.set(State.CLOSED);
                failureCount.set(0);
                successCount.set(0);
            }
        } else if (currentState == State.CLOSED) {
            // Reset failure count on success
            failureCount.set(0);
        }
    }
    
    /**
     * Records a failed request.
     */
    public void recordFailure() {
        lastFailureTime.set(Instant.now());
        State currentState = state.get();
        
        if (currentState == State.HALF_OPEN) {
            log.warn("Circuit breaker transitioning from HALF_OPEN to OPEN (failure during recovery)");
            state.set(State.OPEN);
            successCount.set(0);
        } else if (currentState == State.CLOSED) {
            int failures = failureCount.incrementAndGet();
            if (failures >= failureThreshold) {
                log.error("Circuit breaker transitioning from CLOSED to OPEN (failures: {})", failures);
                state.set(State.OPEN);
            }
        }
    }
    
    /**
     * Gets current circuit breaker state.
     */
    public String getState() {
        return state.get().name();
    }
    
    /**
     * Gets current failure count.
     */
    public int getFailureCount() {
        return failureCount.get();
    }
    
    /**
     * Manually resets the circuit breaker to CLOSED state.
     */
    public void reset() {
        log.info("Circuit breaker manually reset to CLOSED");
        state.set(State.CLOSED);
        failureCount.set(0);
        successCount.set(0);
    }
}
