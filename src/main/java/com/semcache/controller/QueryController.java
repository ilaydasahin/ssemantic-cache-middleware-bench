package com.semcache.controller;

import com.semcache.model.CacheLookupResult;
import com.semcache.model.QueryDtos.*;
import com.semcache.service.EmbeddingService;
import com.semcache.service.LLMService;
import com.semcache.service.SemanticCacheService;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * REST Controller — API Gateway for semantic cache operations.
 * 
 * Endpoints:
 * POST /api/query — Submit a query (with caching)
 * GET /api/stats — Get cache statistics
 * POST /api/cache/clear — Clear the cache
 */
@RestController
@RequestMapping("/api")
public class QueryController {

    private final SemanticCacheService cacheService;
    private final EmbeddingService embeddingService;
    private final LLMService llmService;

    public QueryController(SemanticCacheService cacheService,
            EmbeddingService embeddingService,
            LLMService llmService) {
        this.cacheService = cacheService;
        this.embeddingService = embeddingService;
        this.llmService = llmService;
    }

    /**
     * Main query endpoint — implements the full cache pipeline.
     * 
     * Flow: Query → Embedding → Cache Lookup → Hit: return cached / Miss: call LLM
     * → Store → Return
     */
    @PostMapping("/query")
    public Mono<QueryResponse> query(@RequestBody QueryRequest request) {
        return Mono.defer(() -> {
            long totalStart = System.nanoTime();

            // Step 1: Cache lookup (includes embedding generation)
            CacheLookupResult lookupResult = cacheService.lookup(request.query());

            if (lookupResult.hit()) {
                // Cache HIT — return cached response immediately
                long totalMs = (System.nanoTime() - totalStart) / 1_000_000;

                return Mono.just(new QueryResponse(
                        request.query(),
                        lookupResult.response(),
                        true,
                        lookupResult.similarityScore(),
                        totalMs,
                        lookupResult.embeddingTimeMs(),
                        0, // no LLM call
                        "SEMANTIC"));
            } else {
                // Cache MISS — call LLM
                return llmService.generate(request.query())
                        .map(llmResponse -> {
                            // Store safely strictly within the execution chain
                            float[] embedding = lookupResult.queryEmbedding() != null ? 
                                              lookupResult.queryEmbedding() : 
                                              embeddingService.encode(request.query());
                            
                            cacheService.store(request.query(), embedding, llmResponse);

                            long llmEnd = System.nanoTime();
                            long totalMs = (llmEnd - totalStart) / 1_000_000;
                            long llmMs = totalMs - lookupResult.embeddingTimeMs();

                            return new QueryResponse(
                                    request.query(),
                                    llmResponse,
                                    false,
                                    0.0,
                                    totalMs,
                                    lookupResult.embeddingTimeMs(),
                                    llmMs,
                                    "SEMANTIC");
                        });
            }
        });
    }

    /**
     * Get cache statistics.
     */
    @GetMapping("/stats")
    public Map<String, Object> stats() {
        return cacheService.getStats();
    }

    /**
     * Clear the cache.
     */
    @PostMapping("/cache/clear")
    public Map<String, String> clearCache() {
        cacheService.clearCache();
        return Map.of("status", "Cache cleared", "size", "0");
    }

    /**
     * Health check endpoint.
     */
    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of(
                "status", "UP",
                "cacheSize", cacheService.getCacheSize(),
                "embeddingModel", embeddingService.getModelName(),
                "embeddingDimension", embeddingService.getEmbeddingDimension());
    }

    /**
     * Global Exception Handler for the Controller.
     * Prevents swallowed exceptions and provides clear HTTP 500 responses.
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, String>> handleExceptions(Exception e) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Map.of(
                        "error", "Internal Server Error",
                        "message", e.getMessage() != null ? e.getMessage() : "Unknown semantic cache error"
                ));
    }
}
