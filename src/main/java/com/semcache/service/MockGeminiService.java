package com.semcache.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Enhanced Mock LLM service with realistic cost estimation (§5.8).
 * Uses Gemini 2.0 Flash pricing models for benchmark fidelity.
 */
@Service
@Profile("benchmark-mock")
public class MockGeminiService implements LLMService {

    private static final Logger log = LoggerFactory.getLogger(MockGeminiService.class);

    // Pricing: $0.10 / 1M input, $0.40 / 1M output tokens
    private static final double INPUT_COST_PER_M = 0.10;
    private static final double OUTPUT_COST_PER_M = 0.40;

    private final Map<String, String> groundTruthRegistry = new HashMap<>();

    public void registerGroundTruth(String query, String response) {
        groundTruthRegistry.put(query, response);
    }

    public void clearRegistry() {
        groundTruthRegistry.clear();
    }

    @Override
    public Mono<String> generate(String query) {
        // M.6 Realistic Simulation: 5ms - 25ms API latency (optimized for benchmark
        // speed)
        long delayMs = 5 + ThreadLocalRandom.current().nextInt(20);
        return Mono.delay(Duration.ofMillis(delayMs))
                .map(init -> generateSyncWithoutDelay(query));
    }

    @Override
    public String generateSync(String query) {
        // M.6 Realistic Simulation: 5ms - 25ms API latency (optimized for benchmark
        // speed)
        long delayMs = 5 + ThreadLocalRandom.current().nextInt(20);
        try {
            Thread.sleep(delayMs);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return generateSyncWithoutDelay(query);
    }

    private String generateSyncWithoutDelay(String query) {
        String response = groundTruthRegistry.getOrDefault(query,
                "Mock response for: " + query + " (Ground truth not found in registry)");

        log.debug("Mock LLM call for query: '{}'. Registry hit: {}", query, groundTruthRegistry.containsKey(query));
        return response;
    }

    @Override
    public double estimateCost(String query, String response) {
        // Realistic token estimation (1 token approx 4 characters)
        int inputTokens = (query != null) ? query.length() / 4 : 0;
        int outputTokens = (response != null) ? response.length() / 4 : 0;

        double cost = (inputTokens * INPUT_COST_PER_M / 1_000_000.0) +
                (outputTokens * OUTPUT_COST_PER_M / 1_000_000.0);
        return cost;
    }
}
