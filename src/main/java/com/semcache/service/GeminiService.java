package com.semcache.service;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.context.annotation.Profile;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

/**
 * Gemini API Service — Handles LLM API calls with rate limiting and retry
 * logic.
 * 
 * Uses Google Gemini API (gemini-2.0-flash) for generating responses.
 * Temperature is set to 0.0 for deterministic outputs (reproducibility
 * requirement per M.8).
 */
@Service
@Profile("!benchmark-mock")
public class GeminiService implements LLMService {

    private static final Logger log = LoggerFactory.getLogger(GeminiService.class);
    private static final String GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/";

    @Value("${llm.api-key:}")
    private String apiKey;

    @Value("${llm.model:gemini-2.0-flash}")
    private String model;

    @Value("${llm.temperature:0.0}")
    private double temperature;

    @Value("${llm.max-output-tokens:1024}")
    private int maxOutputTokens;

    private final MeterRegistry meterRegistry;
    private WebClient webClient;
    private Timer llmTimer;
    private Counter llmCallCounter;
    private Counter llmErrorCounter;

    public GeminiService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    @PostConstruct
    public void init() {
        this.webClient = WebClient.builder()
                .baseUrl(GEMINI_API_URL)
                .build();

        this.llmTimer = Timer.builder("llm.latency")
                .tag("model", model)
                .description("Time for LLM API call")
                .register(meterRegistry);

        this.llmCallCounter = Counter.builder("llm.calls")
                .description("Number of LLM API calls")
                .register(meterRegistry);

        this.llmErrorCounter = Counter.builder("llm.errors")
                .description("Number of LLM API errors")
                .register(meterRegistry);

        log.info("GeminiService initialized: model={}, temperature={}, maxTokens={}",
                model, temperature, maxOutputTokens);
    }

    /**
     * Generate a response from the Gemini API.
     * 
     * Algorithm 1, Line 9: response ← L.generate(user_query)
     * 
     * @param query the user query
     * @return LLM-generated response text
     */
    public Mono<String> generate(String query) {
        return Mono.defer(() -> {
            llmCallCounter.increment();

            Map<String, Object> requestBody = Map.of(
                    "contents", List.of(
                            Map.of("parts", List.of(
                                    Map.of("text", query)))),
                    "generationConfig", Map.of(
                            "temperature", temperature,
                            "maxOutputTokens", maxOutputTokens));

            long start = System.nanoTime();
            return webClient.post()
                    .uri(uriBuilder -> uriBuilder
                        .path(model + ":generateContent")
                        .queryParam("key", apiKey)
                        .build())
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(new org.springframework.core.ParameterizedTypeReference<Map<String, Object>>() {})
                    .map(this::extractResponseText)
                    .map(response -> {
                        long durationMs = (System.nanoTime() - start) / 1_000_000;
                        llmTimer.record(java.time.Duration.ofMillis(durationMs));
                        log.debug("Gemini response (Reactive): {}ms", durationMs);
                        return response != null ? response : "No response generated.";
                    })
                    .retryWhen(reactor.util.retry.Retry.backoff(5, java.time.Duration.ofSeconds(2))
                        .filter(e -> {
                            Throwable current = e;
                            while (current != null) {
                                if (current.getMessage() != null && current.getMessage().contains("429")) return true;
                                current = current.getCause();
                            }
                            return false;
                        })
                        .doBeforeRetry(sig -> log.warn("Gemini Rate Limited (429). Retrying... (Attempt {}/5)", sig.totalRetriesInARow() + 1)))
                    .doOnError(e -> {
                        llmErrorCounter.increment();
                        log.error("Gemini API error (Reactive): {}", e.getMessage());
                    });
        });
    }

    /**
     * Generate a response synchronously (for benchmark use).
     */
    public String generateSync(String query) {
        try {
            return generate(query).block();
        } catch (Exception e) {
            throw new RuntimeException("LLM API call failed: " + e.getMessage(), e);
        }
    }

    /**
     * Estimate the cost of an API call based on token count.
     * Gemini 2.0 Flash pricing (approximate):
     * - Input: $0.10 per 1M tokens
     * - Output: $0.40 per 1M tokens
     */
    public double estimateCost(String query, String response) {
        int inputTokens = estimateTokens(query);
        int outputTokens = estimateTokens(response);
        double inputCost = inputTokens * 0.10 / 1_000_000;
        double outputCost = outputTokens * 0.40 / 1_000_000;
        return inputCost + outputCost;
    }

    private int estimateTokens(String text) {
        // Approximate: 1 token ≈ 4 characters for English text
        return text != null ? text.length() / 4 : 0;
    }

    @SuppressWarnings("unchecked")
    private String extractResponseText(Map<String, Object> responseMap) {
        try {
            List<Map<String, Object>> candidates = (List<Map<String, Object>>) responseMap.get("candidates");
            if (candidates != null && !candidates.isEmpty()) {
                Map<String, Object> content = (Map<String, Object>) candidates.get(0).get("content");
                List<Map<String, Object>> parts = (List<Map<String, Object>>) content.get("parts");
                if (parts != null && !parts.isEmpty()) {
                    return (String) parts.get(0).get("text");
                }
            }
        } catch (Exception e) {
            log.error("Failed to parse Gemini response: {}", e.getMessage());
        }
        return null;
    }
}
