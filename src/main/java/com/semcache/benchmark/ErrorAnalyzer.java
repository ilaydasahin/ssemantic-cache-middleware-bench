package com.semcache.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Analyzes errors and provides insights for optimization
 */
@Component
public class ErrorAnalyzer {
    
    private static final Logger log = LoggerFactory.getLogger(ErrorAnalyzer.class);
    
    private final Map<String, AtomicInteger> errorTypes = new ConcurrentHashMap<>();
    private final Map<Double, AtomicInteger> thresholdPerformance = new ConcurrentHashMap<>();
    private final List<String> failedQueries = Collections.synchronizedList(new ArrayList<>());
    
    public void recordError(String errorType, String query) {
        errorTypes.computeIfAbsent(errorType, k -> new AtomicInteger()).incrementAndGet();
        if (failedQueries.size() < 100) { // Keep first 100 failed queries
            failedQueries.add(query);
        }
    }
    
    public void recordThresholdPerformance(double threshold, boolean hit) {
        thresholdPerformance.computeIfAbsent(threshold, k -> new AtomicInteger())
                .addAndGet(hit ? 1 : 0);
    }
    
    public void generateInsights() {
        if (errorTypes.isEmpty() && thresholdPerformance.isEmpty()) {
            log.info("✅ No errors detected - system running smoothly!");
            return;
        }
        
        log.info("=== Error Analysis & Insights ===");
        
        // Error type analysis
        if (!errorTypes.isEmpty()) {
            log.info("Error Types:");
            errorTypes.entrySet().stream()
                    .sorted((a, b) -> b.getValue().get() - a.getValue().get())
                    .forEach(entry -> log.info("  - {}: {} occurrences", 
                            entry.getKey(), entry.getValue().get()));
        }
        
        // Threshold optimization suggestion
        if (!thresholdPerformance.isEmpty()) {
            double bestThreshold = thresholdPerformance.entrySet().stream()
                    .max(Comparator.comparingInt(e -> e.getValue().get()))
                    .map(Map.Entry::getKey)
                    .orElse(0.90);
            
            log.info("💡 Insight: Threshold {:.2f} shows best performance", bestThreshold);
            log.info("💡 Recommendation: Consider using threshold={:.2f} for optimal hit rate", 
                    bestThreshold);
        }
        
        // Failed query patterns
        if (!failedQueries.isEmpty()) {
            log.info("Failed Query Patterns (sample):");
            failedQueries.stream().limit(5).forEach(q -> 
                    log.info("  - {}", q.length() > 80 ? q.substring(0, 80) + "..." : q));
        }
    }
    
    public void reset() {
        errorTypes.clear();
        thresholdPerformance.clear();
        failedQueries.clear();
    }
}
