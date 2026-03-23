package com.semcache.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Compares experiment results and generates improvement reports
 */
@Component
public class ResultComparator {
    
    private static final Logger log = LoggerFactory.getLogger(ResultComparator.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    public void compareWithPrevious(String currentResultFile) {
        try {
            File currentFile = new File(currentResultFile);
            if (!currentFile.exists()) {
                log.warn("Current result file not found: {}", currentResultFile);
                return;
            }
            
            // Find previous results for same dataset
            String dataset = extractDatasetName(currentResultFile);
            List<File> previousResults = findPreviousResults(dataset, currentFile);
            
            if (previousResults.isEmpty()) {
                log.info("No previous results found for comparison");
                return;
            }
            
            // Compare with most recent
            File previousFile = previousResults.get(0);
            log.info("📊 Comparing with previous result: {}", previousFile.getName());
            
            Map<String, Object> current = objectMapper.readValue(currentFile, 
                    new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {});
            Map<String, Object> previous = objectMapper.readValue(previousFile, 
                    new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {});
            
            generateComparisonReport(current, previous);
            
        } catch (Exception e) {
            log.error("Failed to compare results", e);
        }
    }
    
    private void generateComparisonReport(Map<String, Object> current, Map<String, Object> previous) {
        log.info("=== Performance Comparison ===");
        
        compareMetric("Hit Rate", previous, current, "hitRate", "%", true);
        compareMetric("P50 Latency", previous, current, "p50LatencyMs", "ms", false);
        compareMetric("P99 Latency", previous, current, "p99LatencyMs", "ms", false);
        compareMetric("Cost Savings", previous, current, "costSavingsPercent", "%", true);
        compareMetric("Memory Usage", previous, current, "memoryUsageMb", "MB", false);
    }
    
    private void compareMetric(String name, Map<String, Object> prev, Map<String, Object> curr, 
                               String key, String unit, boolean higherIsBetter) {
        try {
            double prevValue = getMetricValue(prev, key);
            double currValue = getMetricValue(curr, key);
            double change = currValue - prevValue;
            double changePercent = prevValue != 0 ? (change / prevValue * 100) : 0;
            
            String arrow = change > 0 ? (higherIsBetter ? "📈" : "📉") : (higherIsBetter ? "📉" : "📈");
            String sign = change > 0 ? "+" : "";
            
            log.info("{} {}: {:.2f}{} → {:.2f}{} ({}{:.1f}%)",
                    arrow, name, prevValue, unit, currValue, unit, sign, changePercent);
                    
        } catch (Exception e) {
            log.debug("Could not compare metric: {}", key);
        }
    }
    
    private double getMetricValue(Map<String, Object> data, String key) {
        Object value = data.get(key);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return 0.0;
    }
    
    private String extractDatasetName(String filename) {
        String name = new File(filename).getName();
        if (name.contains("msmarco")) return "msmarco";
        if (name.contains("nq")) return "nq";
        if (name.contains("qqp")) return "qqp";
        return "unknown";
    }
    
    private List<File> findPreviousResults(String dataset, File currentFile) {
        try {
            Path resultsDir = Paths.get("results");
            if (!Files.exists(resultsDir)) return Collections.emptyList();
            
            return Files.walk(resultsDir)
                    .filter(Files::isRegularFile)
                    .map(Path::toFile)
                    .filter(f -> f.getName().contains(dataset))
                    .filter(f -> f.getName().endsWith(".json"))
                    .filter(f -> !f.equals(currentFile))
                    .sorted((a, b) -> Long.compare(b.lastModified(), a.lastModified()))
                    .collect(Collectors.toList());
                    
        } catch (Exception e) {
            log.error("Failed to find previous results", e);
            return Collections.emptyList();
        }
    }
}
