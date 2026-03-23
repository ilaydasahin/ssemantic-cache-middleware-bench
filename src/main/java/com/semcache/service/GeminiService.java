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
import java.time.Duration;
import java.util.concurrent.Semaphore;

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

    @Value("${llm.api-keys:}")
    private String apiKeysString;

    private String[] apiKeys;
    private final java.util.concurrent.atomic.AtomicInteger currentKeyIndex = new java.util.concurrent.atomic.AtomicInteger(0);

    @Value("${llm.model:gemini-2.5-flash}")
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
    private Counter keyRotationCounter;
    
    // M.6/M.8 Robustness: Parallel Rate Limiting
    // Free tier: 15 RPM per key, 1500 RPD per key
    // With 20 keys: 300 RPM total, 30,000 RPD total
    private Semaphore parallelLimiter;
    private final Map<String, java.util.concurrent.atomic.AtomicLong> lastKeyCallTime = new java.util.concurrent.ConcurrentHashMap<>();
    private final Map<String, java.util.concurrent.atomic.AtomicInteger> keyDailyUsage = new java.util.concurrent.ConcurrentHashMap<>();
    private static final Duration CALL_SPACING = Duration.ofMillis(4800); // 12.5 RPM (safe buffer for 15 RPM)
    private static final int DAILY_QUOTA_PER_KEY = 1450; // Safe buffer for 1500 RPD limit
    private volatile long lastQuotaResetTime = System.currentTimeMillis();
    private static final long QUOTA_RESET_INTERVAL_MS = 24 * 60 * 60 * 1000; // 24 hours
    
    /**
     * Calculate time until next PST midnight (when quotas reset)
     */
    private long getTimeUntilPSTMidnight() {
        java.time.ZonedDateTime now = java.time.ZonedDateTime.now(java.time.ZoneId.of("America/Los_Angeles"));
        java.time.ZonedDateTime nextMidnight = now.toLocalDate().plusDays(1).atStartOfDay(java.time.ZoneId.of("America/Los_Angeles"));
        return java.time.Duration.between(now, nextMidnight).toMillis();
    }

    public GeminiService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    @PostConstruct
    public void init() {
        if (apiKeysString != null && !apiKeysString.isEmpty()) {
            this.apiKeys = apiKeysString.split(",");
            for (int i = 0; i < apiKeys.length; i++) {
                apiKeys[i] = apiKeys[i].trim();
                lastKeyCallTime.put(apiKeys[i], new java.util.concurrent.atomic.AtomicLong(0));
                keyDailyUsage.put(apiKeys[i], new java.util.concurrent.atomic.AtomicInteger(0));
            }
        } else {
            this.apiKeys = new String[0];
            log.error("No API keys provided in llm.api-keys");
        }

        // Parallel permits = number of keys (20 keys = 20 concurrent requests)
        this.parallelLimiter = new Semaphore(Math.max(1, apiKeys.length));

        // Configure WebClient with timeouts to prevent hangs
        io.netty.channel.ChannelOption channelOption = io.netty.channel.ChannelOption.CONNECT_TIMEOUT_MILLIS;
        reactor.netty.http.client.HttpClient httpClient = reactor.netty.http.client.HttpClient.create()
                .option(channelOption, 30000) // 30s connection timeout
                .responseTimeout(Duration.ofSeconds(120)); // 120s response timeout (LLM can be slow)
        
        this.webClient = WebClient.builder()
                .baseUrl(GEMINI_API_URL)
                .clientConnector(new org.springframework.http.client.reactive.ReactorClientHttpConnector(httpClient))
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

        this.keyRotationCounter = Counter.builder("llm.key.rotations")
                .description("Number of API key rotations")
                .register(meterRegistry);

        log.info("GeminiService initialized: model={}, keysLoaded={}, temperature={}, maxTokens={}, totalCapacity={}RPM/{}RPD",
                model, apiKeys.length, temperature, maxOutputTokens, 
                apiKeys.length * 15, apiKeys.length * 1500);
        
        if (apiKeys.length >= 20) {
            log.info("✅ Multi-key mode: {} keys detected. Total capacity: ~{}RPM, ~{}RPD (free tier safe)",
                    apiKeys.length, apiKeys.length * 12, apiKeys.length * 1450);
        }
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
        return Mono.fromCallable(() -> {
            parallelLimiter.acquire();
            return query;
        })
        .flatMap(q -> attemptGenerate(q, 0))
        .doFinally(signalType -> parallelLimiter.release());
    }

    private Mono<String> attemptGenerate(String query, int attempt) {
        if (apiKeys.length == 0) {
            return Mono.error(new RuntimeException("No API keys available."));
        }
        
        // Max retry limit: 100 attempts (prevents infinite loops)
        if (attempt > 100) {
            return Mono.error(new RuntimeException("Max retry attempts exceeded (100)"));
        }
        
        // Check if we need to reset daily quotas (every 24 hours)
        checkAndResetDailyQuotas();
        
        // Pick a key that: 1) hasn't hit daily quota, 2) has been idle for CALL_SPACING
        String currentKey = null;
        int keyIndex = -1;
        
        synchronized (apiKeys) {
            for (int i = 0; i < apiKeys.length; i++) {
                int idx = (currentKeyIndex.get() + i) % apiKeys.length;
                String k = apiKeys[idx];
                
                // Check daily quota first
                int dailyUsage = keyDailyUsage.get(k).get();
                if (dailyUsage >= DAILY_QUOTA_PER_KEY) {
                    log.debug("Key {} exhausted daily quota ({}/{}), skipping", 
                            idx, dailyUsage, DAILY_QUOTA_PER_KEY);
                    continue;
                }
                
                // Check rate limit (4.8s spacing)
                long lastCall = lastKeyCallTime.get(k).get();
                if (System.currentTimeMillis() - lastCall >= CALL_SPACING.toMillis()) {
                    currentKey = k;
                    keyIndex = idx;
                    lastKeyCallTime.get(k).set(System.currentTimeMillis());
                    keyDailyUsage.get(k).incrementAndGet();
                    break;
                }
            }
        }

        // If no key is ready, wait and retry
        if (currentKey == null) {
            // Check if ALL keys exhausted daily quota
            boolean allExhausted = keyDailyUsage.values().stream()
                    .allMatch(usage -> usage.get() >= DAILY_QUOTA_PER_KEY);
            
            if (allExhausted) {
                // Calculate time until next PST midnight (actual quota reset time)
                long timeUntilReset = getTimeUntilPSTMidnight();
                long hoursUntilReset = timeUntilReset / (60 * 60 * 1000);
                long minutesUntilReset = (timeUntilReset % (60 * 60 * 1000)) / (60 * 1000);
                
                int totalCalls = keyDailyUsage.values().stream()
                        .mapToInt(java.util.concurrent.atomic.AtomicInteger::get).sum();
                
                java.time.ZonedDateTime resetTime = java.time.ZonedDateTime.now(java.time.ZoneId.of("America/Los_Angeles"))
                        .toLocalDate().plusDays(1).atStartOfDay(java.time.ZoneId.of("America/Los_Angeles"));
                
                log.warn("⏰ ALL {} keys exhausted ({} total calls). Waiting for PST midnight quota reset...", 
                        apiKeys.length, totalCalls);
                log.warn("💤 Reset time: {} PST | Time remaining: {}h {}m", 
                        resetTime.format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")),
                        hoursUntilReset, minutesUntilReset);
                log.warn("📊 Checkpoint saved. System will auto-resume.");
                
                // Wait 5 minutes and retry (will trigger quota reset check)
                return Mono.delay(Duration.ofMinutes(5))
                           .flatMap(d -> attemptGenerate(query, 0));
            }
            
            // Otherwise just rate-limited, wait and retry
            return Mono.delay(Duration.ofMillis(500))
                       .flatMap(d -> attemptGenerate(query, attempt));
        }

        final int finalKeyIndex = keyIndex;
        final String finalKey = currentKey;

        Map<String, Object> requestBody = Map.of(
                "contents", List.of(
                        Map.of("parts", List.of(
                                Map.of("text", query)))),
                "generationConfig", Map.of(
                        "temperature", temperature,
                        "maxOutputTokens", maxOutputTokens));
        
        long start = System.nanoTime();
        llmCallCounter.increment();
        
        return webClient.post()
                .uri(uriBuilder -> uriBuilder
                    .path(model + ":generateContent")
                    .queryParam("key", finalKey)
                    .build())
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(new org.springframework.core.ParameterizedTypeReference<Map<String, Object>>() {})
                .map(this::extractResponseText)
                .map(response -> {
                    long durationMs = (System.nanoTime() - start) / 1_000_000;
                    llmTimer.record(Duration.ofMillis(durationMs));
                    
                    // Log progress every 100 calls
                    int totalCalls = keyDailyUsage.values().stream()
                            .mapToInt(java.util.concurrent.atomic.AtomicInteger::get).sum();
                    if (totalCalls % 100 == 0) {
                        log.info("Progress: {} total calls across {} keys (avg {}/key)", 
                                totalCalls, apiKeys.length, totalCalls / apiKeys.length);
                    }
                    
                    return response != null ? response : "No response generated.";
                })
                .onErrorResume(e -> {
                    String errorMsg = e.getMessage() != null ? e.getMessage() : "";
                    
                    // If we hit a rate limit (429) or quota exceeded error
                    if (errorMsg.contains("429") || errorMsg.toLowerCase().contains("quota") || 
                        errorMsg.toLowerCase().contains("resource_exhausted")) {
                        
                        log.warn("Key {} hit quota/rate limit. Usage: {}/{}. Rotating...", 
                                finalKeyIndex, keyDailyUsage.get(finalKey).get(), DAILY_QUOTA_PER_KEY);
                        
                        // Mark this key as exhausted
                        keyDailyUsage.get(finalKey).set(DAILY_QUOTA_PER_KEY);
                        currentKeyIndex.incrementAndGet();
                        keyRotationCounter.increment();
                        
                        // Retry with next key - NO LIMIT, will wait if all exhausted
                        return attemptGenerate(query, attempt + 1);
                    }
                    
                    // Network errors, timeouts, etc - RETRY with exponential backoff
                    if (errorMsg.toLowerCase().contains("timeout") || 
                        errorMsg.toLowerCase().contains("connection") ||
                        errorMsg.toLowerCase().contains("network") ||
                        e instanceof java.net.ConnectException ||
                        e instanceof java.io.IOException) {
                        
                        llmErrorCounter.increment();
                        
                        // Exponential backoff: 1s, 2s, 4s, 8s... max 60s
                        long backoffMs = Math.min(1000L * (long) Math.pow(2, attempt % 6), 60000L);
                        log.warn("Network error (key {}): {}. Retrying in {}ms...", 
                                finalKeyIndex, errorMsg, backoffMs);
                        
                        return Mono.delay(Duration.ofMillis(backoffMs))
                                   .flatMap(d -> attemptGenerate(query, attempt + 1));
                    }
                    
                    // Other errors - log and retry with backoff
                    llmErrorCounter.increment();
                    long backoffMs = Math.min(1000L * (long) Math.pow(2, attempt % 6), 60000L);
                    log.error("LLM API error (key {}): {}. Retrying in {}ms...", 
                            finalKeyIndex, errorMsg, backoffMs);
                    
                    return Mono.delay(Duration.ofMillis(backoffMs))
                               .flatMap(d -> attemptGenerate(query, attempt + 1));
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
     * Gemini 1.5 Flash pricing (approximate):
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
    
    /**
     * Check if 24 hours have passed since last reset and reset all quotas
     */
    private void checkAndResetDailyQuotas() {
        long now = System.currentTimeMillis();
        long timeSinceReset = now - lastQuotaResetTime;
        
        if (timeSinceReset >= QUOTA_RESET_INTERVAL_MS) {
            synchronized (apiKeys) {
                // Double-check after acquiring lock
                if (now - lastQuotaResetTime >= QUOTA_RESET_INTERVAL_MS) {
                    int totalCallsBeforeReset = keyDailyUsage.values().stream()
                            .mapToInt(java.util.concurrent.atomic.AtomicInteger::get).sum();
                    
                    // Reset all key quotas
                    keyDailyUsage.values().forEach(counter -> counter.set(0));
                    lastQuotaResetTime = now;
                    
                    log.info("🔄 Daily quota reset completed! All {} keys refreshed. Previous 24h total: {} calls",
                            apiKeys.length, totalCallsBeforeReset);
                    log.info("✅ New capacity available: ~{}RPM, ~{}RPD", 
                            apiKeys.length * 12, apiKeys.length * 1450);
                }
            }
        }
    }
}
