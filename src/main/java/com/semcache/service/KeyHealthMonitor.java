package com.semcache.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Monitors health and performance of individual API keys
 */
@Component
public class KeyHealthMonitor {
    
    private static final Logger log = LoggerFactory.getLogger(KeyHealthMonitor.class);
    
    private final Map<String, KeyStats> keyStats = new ConcurrentHashMap<>();
    
    public static class KeyStats {
        public final AtomicInteger successCount = new AtomicInteger(0);
        public final AtomicInteger errorCount = new AtomicInteger(0);
        public final AtomicLong totalLatencyMs = new AtomicLong(0);
        public volatile boolean disabled = false;
        
        public double getSuccessRate() {
            int total = successCount.get() + errorCount.get();
            return total > 0 ? (successCount.get() * 100.0 / total) : 100.0;
        }
        
        public double getAverageLatency() {
            int count = successCount.get();
            return count > 0 ? (totalLatencyMs.get() / (double) count) : 0.0;
        }
    }
    
    public void recordSuccess(String key, long latencyMs) {
        KeyStats stats = keyStats.computeIfAbsent(key, k -> new KeyStats());
        stats.successCount.incrementAndGet();
        stats.totalLatencyMs.addAndGet(latencyMs);
    }
    
    public void recordError(String key) {
        KeyStats stats = keyStats.computeIfAbsent(key, k -> new KeyStats());
        stats.errorCount.incrementAndGet();
        
        // Auto-disable if error rate > 50% and at least 10 attempts
        int total = stats.successCount.get() + stats.errorCount.get();
        if (total >= 10 && stats.getSuccessRate() < 50.0) {
            stats.disabled = true;
            log.warn("Key auto-disabled due to high error rate: {}% success", 
                    String.format(java.util.Locale.US, "%.1f", stats.getSuccessRate()));
        }
    }
    
    public boolean isKeyHealthy(String key) {
        KeyStats stats = keyStats.get(key);
        return stats == null || !stats.disabled;
    }
    
    public void logHealthReport() {
        if (keyStats.isEmpty()) return;
        
        log.info("=== Key Health Report ===");
        keyStats.forEach((key, stats) -> {
            int keyIndex = key.hashCode() % 100; // Simple index for logging
            log.info("Key {}: Success Rate: {}% | Avg Latency: {}ms | Calls: {} | Status: {}",
                    Math.abs(keyIndex),
                    String.format(java.util.Locale.US, "%.1f", stats.getSuccessRate()),
                    String.format(java.util.Locale.US, "%.0f", stats.getAverageLatency()),
                    stats.successCount.get() + stats.errorCount.get(),
                    stats.disabled ? "DISABLED" : "ACTIVE");
        });
    }
}
