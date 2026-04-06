package com.semcache.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Enhanced Mock LLM service with realistic cost estimation (§5.8).
 * Uses Gemini 2.0 Flash pricing models for benchmark fidelity.
 */
@Service
@Profile("benchmark-mock")
public class MockGeminiService implements LLMService {

    private static final Logger log = LoggerFactory.getLogger(MockGeminiService.class);

    /**
     * Fixed simulated LLM latency in milliseconds.
     *
     * <p><b>Design Rationale (K-4 / §5.8):</b> Using a constant delay instead of
     * {@code ThreadLocalRandom} ensures that latency measurements are fully
     * deterministic across runs with the same seed. A real Gemini 2.0 Flash
     * call typically takes 200–800 ms; we use 15 ms here because the benchmark
     * is measuring <em>cache</em> behaviour, not LLM latency. This value must
     * be reported explicitly in the paper's experimental setup section (§4).
     */
    private static final long MOCK_LLM_LATENCY_MS = 15L;

    private final Map<String, String> groundTruthRegistry = new ConcurrentHashMap<>();

    public void registerGroundTruth(String query, String response) {
        groundTruthRegistry.put(query, response);
    }

    public void clearRegistry() {
        groundTruthRegistry.clear();
    }

    @Override
    public Mono<String> generate(String query) {
        return Mono.delay(Duration.ofMillis(MOCK_LLM_LATENCY_MS))
                .map(ignored -> generateSyncWithoutDelay(query));
    }

    @Override
    public String generateSync(String query) {
        try {
            Thread.sleep(MOCK_LLM_LATENCY_MS);
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
        // Pricing: $0.10 / 1M input, $0.40 / 1M output tokens (Gemini 2.0 Flash)
        final double inputCostPerM  = 0.10;
        final double outputCostPerM = 0.40;
        // Realistic token estimation (1 token ≈ 4 characters)
        int inputTokens  = (query    != null) ? query.length()    / 4 : 0;
        int outputTokens = (response != null) ? response.length() / 4 : 0;
        return (inputTokens * inputCostPerM / 1_000_000.0)
             + (outputTokens * outputCostPerM / 1_000_000.0);
    }
}
