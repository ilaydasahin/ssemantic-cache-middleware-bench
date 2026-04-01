package com.semcache.service;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Map;

/**
 * Ollama Local LLM Service - Tamamen ücretsiz ve yerel model desteği
 * 
 * 16 GB RAM için optimize edilmiş, internet bağlantısı gerektirmez.
 * Desteklenen modeller: llama3.2, phi3, mistral, gemma2
 */
@Service
@Profile("ollama")
public class OllamaService implements LLMService {

    private static final Logger log = LoggerFactory.getLogger(OllamaService.class);
    
    @Value("${llm.ollama.url:http://localhost:11434}")
    private String ollamaUrl;
    
    @Value("${llm.ollama.model:llama3.2}")
    private String model;
    
    @Value("${llm.ollama.temperature:0.0}")
    private double temperature;
    
    @Value("${llm.ollama.max-tokens:1024}")
    private int maxTokens;
    
    @Value("${llm.ollama.timeout-seconds:600}")
    private int timeoutSeconds;  // 10 dakika default (yerel model için)
    
    private final MeterRegistry meterRegistry;
    private WebClient webClient;
    private Timer llmTimer;
    private Counter llmCallCounter;
    private Counter llmErrorCounter;

    public OllamaService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    @PostConstruct
    public void init() {
        // WebClient yapılandırması - Yerel model için uzun timeout
        reactor.netty.http.client.HttpClient httpClient = reactor.netty.http.client.HttpClient.create()
                .responseTimeout(Duration.ofSeconds(timeoutSeconds));
        
        this.webClient = WebClient.builder()
                .baseUrl(ollamaUrl)
                .clientConnector(new org.springframework.http.client.reactive.ReactorClientHttpConnector(httpClient))
                .build();

        this.llmTimer = Timer.builder("llm.latency")
                .tag("model", model)
                .tag("provider", "ollama")
                .description("Ollama LLM çağrı süresi")
                .register(meterRegistry);

        this.llmCallCounter = Counter.builder("llm.calls")
                .tag("provider", "ollama")
                .description("Ollama LLM çağrı sayısı")
                .register(meterRegistry);

        this.llmErrorCounter = Counter.builder("llm.errors")
                .tag("provider", "ollama")
                .description("Ollama LLM hata sayısı")
                .register(meterRegistry);

        log.info("🚀 OllamaService başlatıldı: url={}, model={}, temperature={}, maxTokens={}, timeout={}s", 
                ollamaUrl, model, temperature, maxTokens, timeoutSeconds);
        log.info("✅ Tamamen yerel ve ücretsiz mod aktif - internet bağlantısı gerekmez");
        
        // Ollama bağlantısını test et
        testConnection();
    }

    private void testConnection() {
        try {
            webClient.get()
                    .uri("/api/tags")
                    .retrieve()
                    .bodyToMono(String.class)
                    .timeout(Duration.ofSeconds(5))
                    .block();
            log.info("✅ Ollama bağlantısı başarılı: {}", ollamaUrl);
        } catch (Exception e) {
            log.error("❌ Ollama bağlantısı başarısız! Lütfen Ollama'nın çalıştığından emin olun: {}", e.getMessage());
            log.error("💡 Başlatmak için: ollama serve");
        }
    }

    @Override
    public Mono<String> generate(String query) {
        long start = System.nanoTime();
        llmCallCounter.increment();
        
        Map<String, Object> requestBody = Map.of(
                "model", model,
                "prompt", query,
                "stream", false,
                "options", Map.of(
                        "temperature", temperature,
                        "num_predict", maxTokens
                )
        );
        
        return webClient.post()
                .uri("/api/generate")
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(new org.springframework.core.ParameterizedTypeReference<Map<String, Object>>() {})
                .map(response -> {
                    long durationMs = (System.nanoTime() - start) / 1_000_000;
                    llmTimer.record(Duration.ofMillis(durationMs));
                    
                    String responseText = (String) response.get("response");
                    log.debug("Ollama yanıt alındı: {} ms", durationMs);
                    return responseText != null ? responseText : "Yanıt oluşturulamadı.";
                })
                .onErrorResume(e -> {
                    llmErrorCounter.increment();
                    log.error("❌ Ollama API hatası: {}", e.getMessage());
                    return Mono.error(new RuntimeException("Ollama çağrısı başarısız: " + e.getMessage(), e));
                });
    }

    @Override
    public String generateSync(String query) {
        try {
            return generate(query).block();
        } catch (Exception e) {
            throw new RuntimeException("Ollama API çağrısı başarısız: " + e.getMessage(), e);
        }
    }

    @Override
    public double estimateCost(String query, String response) {
        // Yerel model - maliyet sıfır!
        return 0.0;
    }
}
