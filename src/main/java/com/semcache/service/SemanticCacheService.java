package com.semcache.service;

import com.semcache.config.CacheProperties;
import com.semcache.model.CacheEntry;
import com.semcache.model.CacheLookupResult;
import com.semcache.service.strategy.ExactMatchStrategy;
import com.semcache.service.strategy.GPTCacheBaselineStrategy;
import com.semcache.service.strategy.HybridCascadeStrategy;
import com.semcache.service.strategy.MiddlewareBaselineStrategy;
import com.semcache.service.strategy.NoCacheStrategy;
import com.semcache.service.strategy.SemanticStrategy;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Semantic Cache Service — Implements Algorithm 1 from the paper.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Single authoritative data store ({@code cacheStore} + {@code queryIndex})</li>
 *   <li>Delegating lookup logic to the active {@link CacheLookupStrategy}</li>
 *   <li>Background LFU eviction daemon</li>
 *   <li>Redis parity writes (best-effort, non-blocking)</li>
 * </ul>
 *
 * <p>LocalVectorIndex has been removed. It was a redundant copy of the same
 * data already held in {@code cacheStore}, creating a dual-store inconsistency
 * risk on every write and eviction. All brute-force search now operates
 * directly on {@code cacheStore} via {@link SemanticStrategy}.
 */
@Service
public class SemanticCacheService {

    private static final Logger log = LoggerFactory.getLogger(SemanticCacheService.class);

    private final CacheProperties cacheProperties;
    private final EmbeddingService embeddingService;
    private final RedisSearchService redisSearchService;
    private final MeterRegistry meterRegistry;

    // ── Concurrency control ───────────────────────────────────────────────────
    // ConcurrentHashMap provides thread-safe reads/writes.
    // Eviction uses a dedicated lock to serialise batch removals.
    private final Object evictionLock = new Object();

    // ── Single authoritative data store ──────────────────────────────────────
    /** L2: id → CacheEntry (embedding + response + metadata) */
    private final Map<String, CacheEntry> cacheStore = new ConcurrentHashMap<>();
    /** L1: normalizedQuery → entryId  (O(1) exact-match shortcut) */
    private final Map<String, String> queryIndex = new ConcurrentHashMap<>();

    // ── Background eviction ───────────────────────────────────────────────────
    private final ScheduledExecutorService evictionScheduler =
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "cache-eviction");
                t.setDaemon(true);
                return t;
            });
    private final AtomicBoolean isEvicting = new AtomicBoolean(false);

    // ── Runtime overrides (benchmark knobs) ──────────────────────────────────
    private volatile Double thresholdOverride = null;
    private volatile String strategyOverride = null;
    private volatile String warmupStrategyOverride = null;
    private volatile Boolean hnswEnabledOverride = null;

    // ── Metrics ───────────────────────────────────────────────────────────────
    private Counter cacheHitCounter;
    private Counter cacheMissCounter;

    // ── Strategy registry ─────────────────────────────────────────────────────
    private Map<String, CacheLookupStrategy> strategies;

    public SemanticCacheService(CacheProperties cacheProperties,
                                EmbeddingService embeddingService,
                                RedisSearchService redisSearchService,
                                MeterRegistry meterRegistry) {
        this.cacheProperties = cacheProperties;
        this.embeddingService = embeddingService;
        this.redisSearchService = redisSearchService;
        this.meterRegistry = meterRegistry;

        // 10-second tick reduces CPU overhead during benchmarks while still
        // keeping the cache within capacity bounds (§5 eviction experiment).
        evictionScheduler.scheduleAtFixedRate(
                this::backgroundBatchEviction, 10, 10, TimeUnit.SECONDS);
    }

    @PostConstruct
    public void init() {
        cacheHitCounter = Counter.builder("cache.hits")
                .description("Number of cache hits")
                .register(meterRegistry);
        cacheMissCounter = Counter.builder("cache.misses")
                .description("Number of cache misses")
                .register(meterRegistry);

        // Validation is already performed by CacheProperties.@PostConstruct

        // Build strategy registry — each strategy is stateless and reusable
        // Q1 Publication: Added NO_CACHE baseline for control comparison
        strategies = new HashMap<>();
        strategies.put("SEMANTIC", new SemanticStrategy(embeddingService, redisSearchService));
        strategies.put("HYBRID", new HybridCascadeStrategy(embeddingService));
        strategies.put("EXACT_MATCH", new ExactMatchStrategy());
        strategies.put("MIDDLEWARE_BASELINE", new MiddlewareBaselineStrategy(embeddingService, redisSearchService));
        strategies.put("GPTCACHE_BASELINE", new GPTCacheBaselineStrategy(embeddingService));  // Q1: SOTA baseline
        strategies.put("NONE", new NoCacheStrategy());  // Q1: Control baseline

        log.info("SemanticCacheService initialized: strategy={}, threshold={}, k={}",
                cacheProperties.getStrategy(),
                cacheProperties.getSimilarityThreshold(),
                cacheProperties.getKnnK());
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Algorithm 1: Semantic Cache Lookup.
     * Delegates to the active {@link CacheLookupStrategy} and updates counters.
     */
    public CacheLookupResult lookup(String query) {
        String strategyName = getStrategy().toUpperCase();

        if ("NONE".equals(strategyName)) {
            cacheMissCounter.increment();
            return CacheLookupResult.miss(0, null);
        }

        CacheLookupStrategy strategy = strategies.get(strategyName);
        if (strategy == null) {
            throw new IllegalArgumentException("Unknown cache strategy: " + strategyName);
        }

        CacheContext ctx = buildContext();
        CacheLookupResult result = strategy.lookup(query, ctx);

        if (result.hit()) {
            cacheHitCounter.increment();
            updateLfuCount(ctx.queryIndex().get(ctx.normalize(query)) != null
                    ? ctx.queryIndex().get(ctx.normalize(query))
                    : findIdByResponse(result));
        } else {
            cacheMissCounter.increment();
        }

        return result;
    }

    /**
     * Store a query-response pair in the cache (Algorithm 1, Line 10).
     * Atomically updates both L1 (queryIndex) and L2 (cacheStore).
     */
    public void store(String query, float[] embedding, String response) {
        // ConcurrentHashMap provides thread-safe put; no external lock needed.
        triggerEvictionIfNeeded();

        String id = UUID.randomUUID().toString();

        Map<String, float[]> embeddings = new HashMap<>();
        embeddings.put(embeddingService.getModelName().toLowerCase(), embedding);

        // Secondary embedding only for HYBRID strategy (avoids wasted ONNX calls)
        if ("HYBRID".equalsIgnoreCase(getStrategy())) {
            String secondary = embeddingService.getModelName().equalsIgnoreCase("minilm") ? "mpnet" : "minilm";
            try {
                embeddings.put(secondary, embeddingService.encode(query, secondary));
            } catch (Exception e) {
                log.warn("Could not generate secondary embedding for hybrid cache: {}", e.getMessage());
            }
        }

        CacheEntry entry = new CacheEntry(id, embeddings, query, response,
                System.currentTimeMillis(), 0);

        cacheStore.put(id, entry);
        queryIndex.put(normalize(query), id);

        // Best-effort Redis parity write (non-blocking on failure)
        if (redisSearchService.isAvailable()) {
            redisSearchService.store(id, embedding, query, response);
        }

        log.debug("Stored cache entry: id={}, query_preview='{}'",
                id, query.substring(0, Math.min(50, query.length())));
    }

    /** Clears all in-memory and Redis state. */
    public void clearCache() {
        cacheStore.clear();
        queryIndex.clear();
        if (redisSearchService.isAvailable()) {
            redisSearchService.clear();
        }
        
        // Reset micrometer counters by removing and re-registering
        if (cacheHitCounter != null) {
            meterRegistry.remove(cacheHitCounter);
            cacheHitCounter = Counter.builder("cache.hits").description("Number of cache hits").register(meterRegistry);
        }
        if (cacheMissCounter != null) {
            meterRegistry.remove(cacheMissCounter);
            cacheMissCounter = Counter.builder("cache.misses").description("Number of cache misses").register(meterRegistry);
        }
        
        log.info("Cache cleared (local + Redis + counters)");
    }

    public Map<String, Object> getStats() {
        double hits  = cacheHitCounter.count();
        double misses = cacheMissCounter.count();
        double total  = hits + misses;
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("size", cacheStore.size());
        stats.put("maxEntries", cacheProperties.getMaxEntries());
        stats.put("strategy", getStrategy());
        stats.put("threshold", getSimilarityThreshold());
        stats.put("hits", hits);
        stats.put("misses", misses);
        stats.put("hitRate", total > 0 ? hits / total : 0.0);
        return stats;
    }

    // ── Runtime benchmark knobs ───────────────────────────────────────────────

    public void setSimilarityThreshold(double threshold) { this.thresholdOverride = threshold; }
    public double getSimilarityThreshold() {
        return thresholdOverride != null ? thresholdOverride : cacheProperties.getSimilarityThreshold();
    }

    public void setStrategy(String strategy)  { this.strategyOverride = strategy; }
    public String getStrategy() {
        return strategyOverride != null ? strategyOverride : cacheProperties.getStrategy();
    }

    public void setHnswEnabled(boolean enabled) { this.hnswEnabledOverride = enabled; }
    public boolean isHnswEnabled() {
        return hnswEnabledOverride != null ? hnswEnabledOverride : cacheProperties.isHnswEnabled();
    }

    public void setWarmupStrategy(String strategy) { this.warmupStrategyOverride = strategy; }
    public String getWarmupStrategy() {
        return warmupStrategyOverride != null ? warmupStrategyOverride : cacheProperties.getWarmupStrategy();
    }

    public int getCacheSize() { return cacheStore.size(); }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Builds an immutable {@link CacheContext} snapshot for the current call.
     * Strategies receive unmodifiable views so they cannot mutate the store.
     */
    private CacheContext buildContext() {
        double threshold = getSimilarityThreshold();
        boolean hnsw     = isHnswEnabled();
        long ttlMs       = cacheProperties.getTtlSeconds() * 1000L;

        // Snapshot copies prevent concurrent-modification during brute-force scan.
        // O(n) copy is acceptable because brute-force itself is already O(n).
        Map<String, CacheEntry> entriesView = Map.copyOf(cacheStore);
        Map<String, String>     indexView   = Map.copyOf(queryIndex);

        return new CacheContext() {
            @Override public Map<String, CacheEntry> entries()       { return entriesView; }
            @Override public Map<String, String>     queryIndex()    { return indexView; }
            @Override public double similarityThreshold()            { return threshold; }
            @Override public boolean hnswEnabled()                   { return hnsw; }
            @Override public long ttlMs()                            { return ttlMs; }
            @Override public String normalize(String q)              { return q.toLowerCase().trim(); }
        };
    }

    private String normalize(String query) {
        return query.toLowerCase().trim();
    }

    private void updateLfuCount(String id) {
        if (id != null) {
            cacheStore.computeIfPresent(id, (k, v) -> v.withIncrementedHitCount());
        }
    }

    /** Resolves the entry id from a hit result via reverse lookup on queryIndex. */
    private String findIdByResponse(CacheLookupResult result) {
        if (result.matchedQueryText() == null) return null;
        return queryIndex.get(normalize(result.matchedQueryText()));
    }

    private void triggerEvictionIfNeeded() {
        if (cacheStore.size() >= cacheProperties.getMaxEntries() && !isEvicting.get()) {
            CompletableFuture.runAsync(this::backgroundBatchEviction);
        }
    }

    /**
     * Background LFU batch eviction daemon.
     * Evicts 5 % of capacity when the store exceeds 95 % full.
     * Uses fine-grained locking (chunks of 50) to minimise p99 jitter.
     */
    private void backgroundBatchEviction() {
        int capacity    = cacheProperties.getMaxEntries();
        int currentSize = cacheStore.size();

        if (currentSize < capacity * 0.95 || !isEvicting.compareAndSet(false, true)) {
            return;
        }

        try {
            int targetEvictionCount = (int) (capacity * 0.05);
            log.debug("Eviction triggered: {}/{}, target={}", currentSize, capacity, targetEvictionCount);

            int sampleSize = Math.min(targetEvictionCount * 3, cacheStore.size());
            if (sampleSize == 0) return;

            Iterator<CacheEntry> it = cacheStore.values().iterator();
            List<CacheEntry> candidates = new ArrayList<>(sampleSize);
            for (int i = 0; i < sampleSize && it.hasNext(); i++) {
                candidates.add(it.next());
            }
            candidates.sort(Comparator.comparingInt(CacheEntry::hitCount));

            List<String> victims = new ArrayList<>(targetEvictionCount);
            for (int i = 0; i < Math.min(targetEvictionCount, candidates.size()); i++) {
                victims.add(candidates.get(i).id());
            }

            // Fine-grained locking: process in chunks of 50 to reduce p99 jitter.
            // Dedicated evictionLock prevents concurrent eviction runs from interleaving.
            int chunkSize = 50;
            for (int i = 0; i < victims.size(); i += chunkSize) {
                List<String> batch = victims.subList(i, Math.min(i + chunkSize, victims.size()));
                synchronized (evictionLock) {
                    for (String victimId : batch) {
                        CacheEntry entry = cacheStore.remove(victimId);
                        if (entry != null) {
                            queryIndex.remove(normalize(entry.queryText()));
                        }
                    }
                }
                Thread.yield();
            }

            // Async Redis cleanup — non-blocking
            if (redisSearchService.isAvailable() && !victims.isEmpty()) {
                CompletableFuture.runAsync(() -> victims.forEach(redisSearchService::remove));
            }

            log.info("Eviction complete: {} entries removed.", victims.size());
        } catch (Exception e) {
            // Log full stack trace for debugging, but don't let scheduler die
            log.error("Background eviction failed (scheduler will continue): {}", e.getMessage(), e);
        } finally {
            isEvicting.set(false);
        }
    }

    @PreDestroy
    public void tearDown() {
        evictionScheduler.shutdown();
        try {
            if (!evictionScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                evictionScheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            evictionScheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
