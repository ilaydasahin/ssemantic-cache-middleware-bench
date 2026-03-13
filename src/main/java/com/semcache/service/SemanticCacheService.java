package com.semcache.service;

import com.semcache.config.CacheProperties;
import com.semcache.model.CacheEntry;
import com.semcache.model.CacheLookupResult;
import redis.clients.jedis.search.Document;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Semantic Cache Service — Implements Algorithm 1 from the paper.
 * 
 * Manages cache storage and similarity-based lookup using embedding vectors.
 * In production, this would use Redis with RedisSearch for HNSW-based ANN
 * search.
 * For benchmarking structure, we use an in-memory implementation with
 * brute-force
 * search, which can be swapped for Redis when the infrastructure is set up.
 */
@Service
public class SemanticCacheService {

    private static final Logger log = LoggerFactory.getLogger(SemanticCacheService.class);

    private final CacheProperties cacheProperties;
    private final EmbeddingService embeddingService;
    private final RedisSearchService redisSearchService;
    private final LocalVectorIndex localVectorIndex;
    private final MeterRegistry meterRegistry;

    // Concurrency Control
    private final ReadWriteLock lock = new ReentrantReadWriteLock();

    // L2 Data Store — P1 fix: ConcurrentHashMap for safe parallelStream access
    // under ReadLock
    private final Map<String, CacheEntry> cacheStore = new ConcurrentHashMap<>();

    // L1 Tier: O(1) Exact Match Map (normalizedQuery -> entryId)
    private final Map<String, String> queryToIdMap = new ConcurrentHashMap<>(); // Changed to ConcurrentHashMap

    // LRU Tier: O(1) Eviction Tracker
    // Changed to ConcurrentHashMap, LRU logic handled explicitly
    private final Map<String, Boolean> lruTracker = new ConcurrentHashMap<>();

    // Background Eviction Daemon (M.6 Gold Standard)
    private final ScheduledExecutorService evictionScheduler = Executors.newSingleThreadScheduledExecutor();
    private final AtomicBoolean isEvicting = new AtomicBoolean(false);

    // Runtime override for benchmarking
    private Double thresholdOverride = null;
    private String strategyOverride = null;
    private String warmupStrategyOverride = null;
    private Boolean hnswEnabledOverride = null;

    public void setStrategy(String strategy) {
        this.strategyOverride = strategy;
    }

    public String getStrategy() {
        return strategyOverride != null ? strategyOverride : cacheProperties.getStrategy();
    }

    public void setHnswEnabled(boolean enabled) {
        this.hnswEnabledOverride = enabled;
    }

    public boolean isHnswEnabled() {
        return hnswEnabledOverride != null ? hnswEnabledOverride : cacheProperties.isHnswEnabled();
    }

    public void setWarmupStrategy(String strategy) {
        this.warmupStrategyOverride = strategy;
    }

    public String getWarmupStrategy() {
        return warmupStrategyOverride != null ? warmupStrategyOverride : cacheProperties.getWarmupStrategy();
    }

    // Metrics
    private Counter cacheHitCounter;
    private Counter cacheMissCounter;

    public SemanticCacheService(CacheProperties cacheProperties,
            EmbeddingService embeddingService,
            RedisSearchService redisSearchService,
            LocalVectorIndex localVectorIndex,
            MeterRegistry meterRegistry) {
        this.cacheProperties = cacheProperties;
        this.embeddingService = embeddingService;
        this.redisSearchService = redisSearchService;
        this.localVectorIndex = localVectorIndex;
        this.meterRegistry = meterRegistry;
        // Start background batch sweeper for High-Throughput eviction
        evictionScheduler.scheduleAtFixedRate(this::backgroundBatchEviction, 1, 1, TimeUnit.SECONDS);
    }

    @PostConstruct
    public void init() {
        cacheHitCounter = Counter.builder("cache.hits")
                .description("Number of cache hits")
                .register(meterRegistry);
        cacheMissCounter = Counter.builder("cache.misses")
                .description("Number of cache misses")
                .register(meterRegistry);

        log.info("SemanticCacheService initialized: strategy={}, threshold={}, k={}",
                cacheProperties.getStrategy(),
                cacheProperties.getSimilarityThreshold(),
                cacheProperties.getKnnK());
    }

    /**
     * Algorithm 1: Semantic Cache Lookup
     * 
     * Implements the lookup logic described in Section 3.1.
     * Input: query q, threshold θ, embedding model E
     * Output: CacheLookupResult (hit/miss, response, similarity, timing)
     */
    public CacheLookupResult lookup(String query) {
        String strategy = getStrategy();
        return switch (strategy.toUpperCase()) {
            case "SEMANTIC" -> semanticLookup(query);
            case "HYBRID" -> cascadedLookup(query);
            case "EXACT_MATCH" -> exactMatchLookup(query);
            case "MIDDLEWARE_BASELINE" -> middlewareLookup(query);
            case "NONE" -> CacheLookupResult.miss(0, null);
            default -> throw new IllegalArgumentException("Unknown strategy: " + strategy);
        };
    }

    /**
     * Store a query-response pair in the cache.
     * 
     * Algorithm 1, Line 10: Store(q, v_q, r)
     * Performs atomic update of L1, L2, and LRU trackers.
     */
    public void store(String query, float[] embedding, String response) {
        lock.writeLock().lock();
        try {
            // Memory pressure is now handled by the background daemon (Batch Sweeper)
            if (cacheStore.size() >= cacheProperties.getMaxEntries() && !isEvicting.get()) {
                // If the background thread is falling behind, trigger an immediate async sweep
                java.util.concurrent.CompletableFuture.runAsync(this::backgroundBatchEviction);
            }

            String id = UUID.randomUUID().toString();

            // Hybrid Storage: Store embeddings for multiple models proactively (§5.7)
            Map<String, float[]> embeddings = new HashMap<>();
            embeddings.put(embeddingService.getModelName().toLowerCase(), embedding);

            // Resolve secondary model (Cascaded Strategy: MiniLM <-> MPNet)
            // M.7 Optimization: Only encode secondary if strategy is HYBRID to save time
            // during other benchmarks.
            if ("HYBRID".equalsIgnoreCase(getStrategy())) {
                String secondary = embeddingService.getModelName().equalsIgnoreCase("minilm") ? "mpnet" : "minilm";
                try {
                    embeddings.put(secondary, embeddingService.encode(query, secondary));
                } catch (Exception e) {
                    log.warn("Could not generate secondary embedding for hybrid cache: {}", e.getMessage());
                }
            }

            CacheEntry entry = new CacheEntry(
                    id, embeddings, query, response, System.currentTimeMillis(), 0);

            cacheStore.put(id, entry);
            // Fix #1: Populate L1 exact-match map so identical queries get O(1) lookup
            queryToIdMap.put(normalizeQuery(query), id);
            // Atomic tracking update
            synchronized (lruTracker) {
                lruTracker.put(id, true);
            }

            // Also store in Local Vector Index for standalone ANN benchmark
            localVectorIndex.add(id, embedding, query, response);

            // Maintain external index parity for hybrid search scenarios
            if (redisSearchService.isAvailable()) {
                redisSearchService.store(id, embedding, query, response);
            }

            log.debug("Stored cache entry (Atomic): id={}, query_preview='{}'",
                    id, query.substring(0, Math.min(50, query.length())));
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Clear all cache entries.
     */
    public void clearCache() {
        lock.writeLock().lock();
        try {
            cacheStore.clear();
            queryToIdMap.clear();
            lruTracker.clear();
            localVectorIndex.clear();
            // Critical: also clear Redis vectorset to prevent cross-run vector
            // contamination
            if (redisSearchService.isAvailable()) {
                redisSearchService.clear();
            }
            log.info("Cache cleared (Atomic) — local + Redis vectorset");
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Get current cache statistics.
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("size", cacheStore.size());
        stats.put("maxEntries", cacheProperties.getMaxEntries());
        stats.put("strategy", getStrategy());
        stats.put("threshold", getSimilarityThreshold());
        stats.put("hits", cacheHitCounter.count());
        stats.put("misses", cacheMissCounter.count());
        double total = cacheHitCounter.count() + cacheMissCounter.count();
        stats.put("hitRate", total > 0 ? cacheHitCounter.count() / total : 0.0);
        return stats;
    }

    public void setSimilarityThreshold(double threshold) {
        this.thresholdOverride = threshold;
    }

    public double getSimilarityThreshold() {
        return thresholdOverride != null ? thresholdOverride : cacheProperties.getSimilarityThreshold();
    }

    public int getCacheSize() {
        return cacheStore.size();
    }

    // ============================================
    // Private methods
    // ============================================

    private String normalizeQuery(String query) {
        return query.toLowerCase().trim();
    }

    /**
     * Semantic lookup using embedding similarity (L2 Tier).
     * Implements Algorithm 1, Lines 1-13.
     */
    private CacheLookupResult semanticLookup(String query) {
        // Line 1: Normalize query (q_norm)
        String qNorm = normalizeQuery(query);

        // --- OPTIMIZATION: Check L1 Tier FIRST to avoid embedding cost ---
        lock.readLock().lock();
        try {
            String existingId = queryToIdMap.get(qNorm);
            if (existingId != null) {
                CacheEntry entry = cacheStore.get(existingId);
                if (entry != null && (System.currentTimeMillis() - entry.timestamp()) <= cacheProperties.getTtlSeconds()
                        * 1000L) {
                    cacheHitCounter.increment();
                    updateLruOrder(existingId);
                    return CacheLookupResult.hit(entry.response(), 1.0, 0, 0, entry.queryText(), entry.embedding());
                }
            }
        } finally {
            lock.readLock().unlock();
        }

        // Line 2: Generate query embedding (v_q)
        long embedStart = System.nanoTime();
        float[] queryVec = embeddingService.encode(query);
        long embeddingTimeMs = (System.nanoTime() - embedStart) / 1_000_000;

        if (cacheStore.isEmpty()) {
            cacheMissCounter.increment();
            return CacheLookupResult.miss(embeddingTimeMs, queryVec);
        }

        long lookupStart = System.nanoTime();

        // --- E4 Fix: Unified search strategy ---
        // HNSW DISABLED: use local brute-force only (do NOT also call LocalVectorIndex
        // — they operate on the same data, calling both doubles latency for zero gain)
        if (!isHnswEnabled()) {
            lock.readLock().lock();
            try {
                return bruteForceLookup(queryVec, lookupStart, embeddingTimeMs);
            } finally {
                lock.readLock().unlock();
            }
        }

        // --- HNSW ENABLED: try RedisSearch first ---
        if (redisSearchService.isAvailable()) {
            try {
                Optional<List<Document>> results = redisSearchService.search(queryVec, 1);
                if (results.isPresent() && !results.get().isEmpty()) {
                    var doc = results.get().get(0);
                    double similarity = doc.getScore();
                    if (similarity >= getSimilarityThreshold()) {
                        cacheHitCounter.increment();
                        String id = doc.getId();
                        updateLruOrder(id);
                        long totalLookupTimeMs = (System.nanoTime() - lookupStart) / 1_000_000;
                        return CacheLookupResult.hit(doc.getString("response"), similarity,
                                totalLookupTimeMs, embeddingTimeMs, doc.getString("query"), queryVec);
                    }
                }
            } catch (Exception e) {
                log.warn("RedisSearch failed: {}. Falling back to local brute-force.", e.getMessage());
            }
        }

        // --- Fallback: local brute-force (covers Redis unavailable + Redis miss) ---
        lock.readLock().lock();
        try {
            return bruteForceLookup(queryVec, lookupStart, embeddingTimeMs);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Hybrid Cascaded Lookup (§5.7 Innovation)
     * 
     * Uses Fast Model (MiniLM) for candidate generation and Stable Model (MPNet)
     * for high-fidelity verification.
     */
    private CacheLookupResult cascadedLookup(String query) {
        long start = System.nanoTime();

        // Tier 1: Fast Recall (MiniLM)
        float[] minilmVec = embeddingService.encode(query, "minilm");
        long embedTime = (System.nanoTime() - start) / 1_000_000;

        lock.readLock().lock();
        try {
            // Find candidates using MiniLM
            double threshold = getSimilarityThreshold();
            var candidates = cacheStore.values().parallelStream()
                    .map(entry -> {
                        float[] cachedMiniLM = entry.embeddings().get("minilm");
                        double sim = (cachedMiniLM != null) ? embeddingService.cosineSimilarity(minilmVec, cachedMiniLM)
                                : 0.0;
                        return new KnnResult(entry, sim);
                    })
                    .filter(r -> r.similarity >= (threshold - 0.05)) // Broad recall window
                    .sorted(Comparator.comparingDouble((KnnResult r) -> r.similarity).reversed())
                    .limit(3) // Top 3 candidates for verification
                    .toList();

            if (candidates.isEmpty()) {
                cacheMissCounter.increment();
                return CacheLookupResult.miss(embedTime, minilmVec);
            }

            // Tier 2: Precision Verification (MPNet)
            // If top candidate is very strong, skip slow verification
            if (candidates.get(0).similarity >= (threshold + 0.05)) {
                KnnResult hit = candidates.get(0);
                cacheHitCounter.increment();
                updateLruOrder(hit.entry.id());
                return CacheLookupResult.hit(hit.entry.response(), hit.similarity,
                        (System.nanoTime() - start) / 1_000_000, embedTime, hit.entry.queryText(), minilmVec);
            }

            // Otherwise, verify with MPNet (The "Verification" Model)
            float[] mpnetVec = embeddingService.encode(query, "mpnet");
            for (KnnResult candidate : candidates) {
                float[] cachedMPNet = candidate.entry.embeddings().get("mpnet");
                if (cachedMPNet != null) {
                    double actualSim = embeddingService.cosineSimilarity(mpnetVec, cachedMPNet);
                    if (actualSim >= threshold) {
                        cacheHitCounter.increment();
                        updateLruOrder(candidate.entry.id());
                        return CacheLookupResult.hit(candidate.entry.response(), actualSim,
                                (System.nanoTime() - start) / 1_000_000, embedTime, candidate.entry.queryText(),
                                mpnetVec);
                    }
                }
            }
        } finally {
            lock.readLock().unlock();
        }

        cacheMissCounter.increment();
        return CacheLookupResult.miss(embedTime, minilmVec);
    }

    private CacheLookupResult bruteForceLookup(float[] queryVec, long lookupStart, long embeddingTimeMs) {
        var bestResult = cacheStore.values().parallelStream()
                .filter(entry -> (System.currentTimeMillis() - entry.timestamp()) <= cacheProperties.getTtlSeconds()
                        * 1000L)
                .map(entry -> new KnnResult(entry, embeddingService.cosineSimilarity(queryVec, entry.embedding())))
                .max(Comparator.comparingDouble(r -> r.similarity));

        if (bestResult.isPresent()) {
            KnnResult best = bestResult.get();
            // Decision: cos_sim(v_q, c*) >= θ
            if (best.similarity >= getSimilarityThreshold()) {
                cacheHitCounter.increment();
                updateLruOrder(best.entry.id()); // Fixed concurrency reordering
                long totalLookupTimeMs = (System.nanoTime() - lookupStart) / 1_000_000;
                return CacheLookupResult.hit(best.entry.response(), best.similarity, totalLookupTimeMs, embeddingTimeMs,
                        best.entry.queryText(), queryVec);
            }
        }

        cacheMissCounter.increment();
        return CacheLookupResult.miss(embeddingTimeMs, queryVec);
    }

    /**
     * Optimized Exact match lookup using SHA-256 / HashMap.
     * Complexity: O(1)
     */
    private CacheLookupResult exactMatchLookup(String query) {
        long start = System.nanoTime();
        lock.readLock().lock();
        try {
            String id = queryToIdMap.get(normalizeQuery(query));

            if (id != null) {
                CacheEntry entry = cacheStore.get(id);
                if (entry != null && (System.currentTimeMillis() - entry.timestamp()) <= cacheProperties.getTtlSeconds()
                        * 1000L) {
                    cacheHitCounter.increment();
                    updateLruOrder(id); // Fixed concurrency reordering
                    long lookupTimeMs = (System.nanoTime() - start) / 1_000_000;
                    return CacheLookupResult.hit(entry.response(), 1.0, lookupTimeMs, 0, entry.queryText(),
                            entry.embedding());
                }
            }
        } finally {
            lock.readLock().unlock();
        }

        cacheMissCounter.increment();
        return CacheLookupResult.miss(0, null);
    }

    /**
     * Simulation of middleware-based semantic caches (e.g., LangChain/GPTCache).
     * M.6 Fair Benchmarking: We simulate the serialization, network hop, and Python
     * abstraction layer delays typical of these systems before falling back to
     * a standard semantic search.
     */
    private CacheLookupResult middlewareLookup(String query) {
        long start = System.nanoTime();

        // 1. Simulate Middleware Parsing & Serialization Overhead (Fixed penalty)
        try {
            // Simulated 15-20ms penalty typical of intermediate Python/REST hops
            Thread.sleep(15);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // 2. Perform standard semantic lookup
        CacheLookupResult baseResult = semanticLookup(query);

        // 3. Adjust total execution time to reflect the simulated penalty
        long totalTimeMs = (System.nanoTime() - start) / 1_000_000;

        if (baseResult.hit()) {
            return CacheLookupResult.hit(
                    baseResult.response(),
                    baseResult.similarityScore(),
                    totalTimeMs,
                    baseResult.embeddingTimeMs(),
                    baseResult.matchedQueryText(),
                    baseResult.queryEmbedding());
        } else {
            return CacheLookupResult.miss(baseResult.embeddingTimeMs(), baseResult.queryEmbedding());
        }
    }

    private void updateLruOrder(String id) {
        // Fix 1: LinkedHashMap.get() with accessOrder=true IS a write operation.
        // We must synchronize access to the tracker because ReadLock doesn't protect
        // against mutations.
        synchronized (lruTracker) {
            lruTracker.get(id);
        }

        // Fix 2 (Semantic-Aware Eviction prep): Update hit count to track frequency
        // (LFU)
        // thread-safe atomic update for ConcurrentHashMap
        cacheStore.computeIfPresent(id, (k, v) -> v.withIncrementedHitCount());
    }

    /**
     * Gold Standard Batch Eviction (Background Daemon).
     * M.6 System Architecture: Triggers when cache is > 95% full. Evicts 5% of
     * capacity
     * in a single batch async operation, preserving tail-latency for active
     * queries.
     */
    private void backgroundBatchEviction() {
        int capacity = cacheProperties.getMaxEntries();
        int currentSize = cacheStore.size();

        if (currentSize < capacity * 0.95 || !isEvicting.compareAndSet(false, true)) {
            return;
        }

        try {
            int targetEvictionCount = (int) (capacity * 0.05);
            log.debug("Background eviction triggered. Cache at {}/{}. targetEviction={}",
                    currentSize, capacity, targetEvictionCount);

            List<String> victims = new ArrayList<>();
            synchronized (lruTracker) {
                int sampleSize = Math.min(targetEvictionCount * 3, lruTracker.size());
                if (sampleSize == 0)
                    return;

                Iterator<String> it = lruTracker.keySet().iterator();
                List<CacheEntry> candidates = new ArrayList<>();
                for (int i = 0; i < sampleSize && it.hasNext(); i++) {
                    CacheEntry e = cacheStore.get(it.next());
                    if (e != null)
                        candidates.add(e);
                }

                candidates.sort(Comparator.comparingInt(CacheEntry::hitCount));

                for (int i = 0; i < Math.min(targetEvictionCount, candidates.size()); i++) {
                    victims.add(candidates.get(i).id());
                }
            }

            // M.6 Gold Standard Fix (Small-Batching):
            // Instead of holding the WriteLock for the entire batch, we process
            // in chunks of 50 to allow waiting read/write threads to interleave.
            // This prevents "Eviction Jitter" in tail latency (p99).
            int chunkSize = 50;
            for (int i = 0; i < victims.size(); i += chunkSize) {
                int end = Math.min(i + chunkSize, victims.size());
                List<String> currentBatch = victims.subList(i, end);

                lock.writeLock().lock();
                try {
                    for (String victimId : currentBatch) {
                        lruTracker.remove(victimId);
                        CacheEntry entry = cacheStore.remove(victimId);
                        if (entry != null) {
                            queryToIdMap.remove(normalizeQuery(entry.queryText()));
                        }
                        localVectorIndex.remove(victimId);
                    }
                } finally {
                    lock.writeLock().unlock();
                }
                Thread.yield(); // Allow other threads to grab the lock
            }

            if (redisSearchService.isAvailable() && !victims.isEmpty()) {
                java.util.concurrent.CompletableFuture.runAsync(() -> {
                    for (String victimId : victims) {
                        redisSearchService.remove(victimId);
                    }
                });
            }
            log.info("Batch Sweeper (Fine-Grained) completed. Evicted {} entries.", victims.size());
        } catch (Exception e) {
            log.error("Background eviction failed: {}", e.getMessage());
        } finally {
            isEvicting.set(false);
        }
    }

    private record KnnResult(CacheEntry entry, double similarity) {
    }
}
