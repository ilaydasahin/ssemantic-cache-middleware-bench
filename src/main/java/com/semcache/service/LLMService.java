package com.semcache.service;

import reactor.core.publisher.Mono;

/**
 * Interface for LLM services to allow switching between real and mock
 * implementations.
 */
public interface LLMService {
    /**
     * Generate a response asynchronously.
     */
    Mono<String> generate(String query);

    /**
     * Generate a response synchronously (benchmarking).
     */
    String generateSync(String query);

    /**
     * Estimate cost of an LLM call.
     */
    double estimateCost(String query, String response);
}
