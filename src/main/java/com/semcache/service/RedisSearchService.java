package com.semcache.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import redis.clients.jedis.JedisPooled;
import redis.clients.jedis.ConnectionPoolConfig;
import redis.clients.jedis.search.Document;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

/**
 * RedisSearchService — Adapted for Redis 8.x native vectorset (VADD/VSIM).
 *
 * Redis 8 ships with a native vectorset module that replaces the old
 * RediSearch FT.CREATE approach. This implementation uses raw Jedis
 * sendCommand() for VADD (store) and VSIM (search) operations.
 *
 * VADD syntax: VADD key FP32 <blob> element [SETATTR json]
 * VSIM syntax: VSIM key FP32 <blob> [COUNT k] [WITHSCORES]
 */
@Service
public class RedisSearchService {

    private static final Logger log = LoggerFactory.getLogger(RedisSearchService.class);

    @Value("${spring.data.redis.host:localhost}")
    private String redisHost;

    @Value("${spring.data.redis.port:6379}")
    private int redisPort;

    private final ObjectMapper objectMapper = new ObjectMapper();
    private JedisPooled jedis;
    private boolean available = false;

    // We store metadata (query, response) in a separate Hash key alongside the
    // vector
    private static final String VSET_KEY = "semcache:vset";

    @PostConstruct
    public void init() {
        try {
            ConnectionPoolConfig poolConfig = new ConnectionPoolConfig();
            poolConfig.setMaxTotal(64);
            poolConfig.setMaxIdle(32);
            poolConfig.setMinIdle(8);
            poolConfig.setTestOnBorrow(true);

            jedis = new JedisPooled(poolConfig, redisHost, redisPort);
            // Test connectivity
            jedis.ping();
            // Test vectorset availability via a probe VADD/DEL
            available = probeVectorset();
            if (available) {
                log.info("RedisSearchService initialized with Redis 8 native vectorset: {}:{}", redisHost, redisPort);
            } else {
                log.warn(
                        "Redis vectorset not available or incompatible. HNSW phase will be skipped (local-only mode).");
            }
        } catch (Exception e) {
            available = false;
            log.error("Failed to initialize Redis connection: {}. Falling back to local-only mode.", e.getMessage());
        }
    }

    /**
     * Probe whether the Redis 8 vectorset module is available by running a minimal
     * VADD then deleting the probe key.
     */
    private boolean probeVectorset() {
        try {
            String probeKey = "semcache:probe";
            byte[] probeVec = floatToBytesLE(new float[] { 0.0f, 0.0f, 0.0f });
            // VADD probeKey FP32 <vec> probe_element
            jedis.sendCommand(() -> "VADD".getBytes(), probeKey.getBytes(), "FP32".getBytes(), probeVec,
                    "probe".getBytes());
            jedis.del(probeKey);
            log.info("Redis 8 vectorset probe successful.");
            return true;
        } catch (Exception e) {
            log.warn("Redis vectorset probe failed: {}. HNSW mode disabled.", e.getMessage());
            return false;
        }
    }

    /**
     * Store a vector entry.
     * Uses: VADD key FP32 <blob> element SETATTR '{"query":..., "response":...}'
     */
    public void store(String id, float[] embedding, String query, String response) {
        if (!available || jedis == null)
            return;

        try {
            byte[] vecBytes = floatToBytesLE(embedding);
            // Fix #9: Use Jackson to safely build attrJson — handles quotes, newlines,
            // unicode
            Map<String, String> attrMap = new LinkedHashMap<>();
            attrMap.put("query", query);
            attrMap.put("response", response);
            String attrJson = objectMapper.writeValueAsString(attrMap);

            // VADD VSET_KEY FP32 <bytes> id SETATTR attrJson
            jedis.sendCommand(() -> "VADD".getBytes(),
                    VSET_KEY.getBytes(),
                    "FP32".getBytes(),
                    vecBytes,
                    id.getBytes(),
                    "SETATTR".getBytes(),
                    attrJson.getBytes());
        } catch (Exception e) {
            log.debug("VADD failed for id={}: {}", id, e.getMessage());
        }
    }

    /**
     * Remove a vector entry from the vectorset.
     * Uses: VDEL key element
     */
    public void remove(String id) {
        if (!available || jedis == null)
            return;
        try {
            jedis.sendCommand(() -> "VREM".getBytes(), VSET_KEY.getBytes(), id.getBytes());
        } catch (Exception e) {
            log.debug("VDEL failed for id={}: {}", id, e.getMessage());
        }
    }

    /**
     * Search for nearest neighbors using Redis 8 VSIM.
     * Uses: VSIM key FP32 <blob> COUNT k WITHSCORES GETATTR
     *
     * Returns Optional<List<Document>>.
     */
    public Optional<List<Document>> search(float[] queryVector, int k) {
        if (!available || jedis == null)
            return Optional.empty();

        try {
            byte[] vecBytes = floatToBytesLE(queryVector);

            // VSIM VSET_KEY FP32 <bytes> COUNT k WITHSCORES GETATTR
            Object raw = jedis.sendCommand(() -> "VSIM".getBytes(),
                    VSET_KEY.getBytes(),
                    "FP32".getBytes(),
                    vecBytes,
                    "COUNT".getBytes(),
                    String.valueOf(k).getBytes(),
                    "WITHSCORES".getBytes(),
                    "GETATTR".getBytes());

            if (raw == null)
                return Optional.empty();

            // Parse response: list of [id, score, attrJson] triples
            List<Document> docs = parseVsimResponse(raw);
            if (docs.isEmpty())
                return Optional.empty();

            return Optional.of(docs);

        } catch (Exception e) {
            log.debug("VSIM search failed: {}", e.getMessage());
            return Optional.empty();
        }
    }

    /**
     * Clear all stored vectors.
     */
    public void clear() {
        if (!available || jedis == null)
            return;
        try {
            jedis.del(VSET_KEY);
            log.info("Vectorset cleared.");
        } catch (Exception e) {
            log.error("Failed to clear vectorset: {}", e.getMessage());
        }
    }

    public boolean isAvailable() {
        return available;
    }

    // ============================================================
    // Private helpers
    // ============================================================

    /**
     * Parse VSIM WITHSCORES GETATTR response.
     * Expected Redis reply structure (list): [id, score, attrJson, id, score,
     * attrJson, ...]
     */
    @SuppressWarnings("unchecked")
    private List<Document> parseVsimResponse(Object raw) {
        List<Document> docs = new ArrayList<>();
        try {
            List<Object> items = (List<Object>) raw;
            // Items: [id, score, attrJson, ...]
            for (int i = 0; i + 2 < items.size(); i += 3) {
                String id = new String((byte[]) items.get(i));
                double score;
                try {
                    score = Double.parseDouble(new String((byte[]) items.get(i + 1)));
                } catch (Exception e) {
                    score = 0.0;
                }
                // attrJson — parse query and response
                String attrJson = items.get(i + 2) instanceof byte[]
                        ? new String((byte[]) items.get(i + 2))
                        : String.valueOf(items.get(i + 2));

                String query = extractJsonField(attrJson, "query");
                String response = extractJsonField(attrJson, "response");

                // VSIM returns similarity score for cosine similarity (1.0 = identical).
                // Previously, this was mistakenly inverted as 1.0 - score, which caused
                // the "Fidelity Paradox" by mapping low similarities to high values.
                double similarity = score;

                // Build properties map
                Map<String, Object> properties = new HashMap<>();
                properties.put("score", String.valueOf(score));
                properties.put("query", query);
                properties.put("response", response);

                // Document constructor: (String id, Map<String, Object> properties, double
                // score)
                // Note: Jedis 5.x uses (id, properties, score)
                Document doc = new Document(id, properties, similarity);
                docs.add(doc);
            }
        } catch (Exception e) {
            log.debug("Failed to parse VSIM response: {}", e.getMessage());
        }
        return docs;
    }

    /**
     * Convert float[] to little-endian byte[] (Redis FP32 format).
     */
    private byte[] floatToBytesLE(float[] input) {
        ByteBuffer buf = ByteBuffer.allocate(input.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : input) {
            buf.putFloat(v);
        }
        return buf.array();
    }

    /**
     * Fix #9: Extract a JSON field using Jackson ObjectMapper.
     * Handles nested quotes, unicode, newlines properly.
     */
    @SuppressWarnings("unchecked")
    private String extractJsonField(String json, String field) {
        try {
            Map<String, Object> map = objectMapper.readValue(json, Map.class);
            Object val = map.get(field);
            return val != null ? val.toString() : "";
        } catch (Exception e) {
            // Fallback to simple extraction for malformed JSON
            try {
                String marker = "\"" + field + "\":\"";
                int start = json.indexOf(marker);
                if (start < 0)
                    return "";
                start += marker.length();
                int end = json.indexOf("\"", start);
                return end > start ? json.substring(start, end).replace("\\\"", "\"") : "";
            } catch (Exception ex) {
                return "";
            }
        }
    }

    @PreDestroy
    public void tearDown() {
        if (jedis != null) {
            jedis.close();
        }
    }
}
