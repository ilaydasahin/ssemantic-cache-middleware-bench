package com.semcache.integration;

import com.semcache.benchmark.BenchmarkRunner;
import com.semcache.benchmark.ExperimentConfig;
import com.semcache.service.SemanticCacheService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration test for end-to-end benchmark execution.
 * 
 * Tests the complete pipeline: dataset loading → warmup → test → results export.
 */
@SpringBootTest
@ActiveProfiles({"test", "benchmark-mock"})
class EndToEndBenchmarkTest {

    @Autowired
    private BenchmarkRunner benchmarkRunner;

    @Autowired
    private SemanticCacheService cacheService;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        cacheService.clearCache();
    }

    @Test
    @DisplayName("Should complete full benchmark with mock LLM")
    void testFullBenchmarkExecution() throws Exception {
        // Arrange
        Path outputFile = tempDir.resolve("test_results.json");
        
        ExperimentConfig config = ExperimentConfig.createV2(
            "msmarco",                                      // datasetName
            "data/msmarco_sample_with_paraphrases.jsonl",  // datasetPath
            "minilm",                                       // embeddingModelName
            0.90,                                           // similarityThreshold
            "UNIDIRECTIONAL",                               // warmupStrategy
            0.30,                                           // warmupRatio
            42L,                                            // randomSeed
            100,                                            // sampleSize (small for fast test)
            false,                                          // hnswEnabled (disabled for test)
            "SEMANTIC",                                     // cacheStrategy
            10,                                             // knnK
            10000,                                          // maxCacheEntries
            3600L,                                          // ttlSeconds
            1,                                              // concurrentUsers
            0.0,                                            // zipfianSkew (no skew)
            0.0,                                            // noiseProbability (no noise)
            outputFile.toString()                           // outputFilePath
        );

        // Act
        benchmarkRunner.run(config);

        // Assert
        assertThat(outputFile).exists();
        assertThat(Files.size(outputFile)).isGreaterThan(100); // Non-empty JSON
        
        String content = Files.readString(outputFile);
        assertThat(content).contains("\"hitRate\"");
        assertThat(content).contains("\"p50LatencyMs\"");  // Changed from avgLatencyMs
        assertThat(content).contains("\"p99LatencyMs\"");
        assertThat(content).contains("\"costSavingsPercent\"");
        
        // Verify cache was populated
        assertThat(cacheService.getCacheSize()).isGreaterThan(0);
    }

    @Test
    @DisplayName("Should handle different strategies")
    void testMultipleStrategies() throws Exception {
        String[] strategies = {"SEMANTIC", "EXACT_MATCH", "HYBRID"};
        
        for (String strategy : strategies) {
            // Arrange
            cacheService.clearCache();
            Path outputFile = tempDir.resolve("test_" + strategy + ".json");
            
            ExperimentConfig config = ExperimentConfig.createV2(
                "msmarco",                                      // datasetName
                "data/msmarco_sample_with_paraphrases.jsonl",  // datasetPath
                "minilm",                                       // embeddingModelName
                0.90,                                           // similarityThreshold
                "UNIDIRECTIONAL",                               // warmupStrategy
                0.30,                                           // warmupRatio
                42L,                                            // randomSeed
                50,                                             // sampleSize
                false,                                          // hnswEnabled
                strategy,                                       // cacheStrategy
                10,                                             // knnK
                10000,                                          // maxCacheEntries
                3600L,                                          // ttlSeconds
                1,                                              // concurrentUsers
                0.0,                                            // zipfianSkew
                0.0,                                            // noiseProbability
                outputFile.toString()                           // outputFilePath
            );

            // Act
            benchmarkRunner.run(config);

            // Assert
            assertThat(outputFile).exists();
            String content = Files.readString(outputFile);
            assertThat(content).contains("\"cacheStrategy\"");  // Just check field exists
            assertThat(content).contains("\"" + strategy + "\"");  // Check strategy value exists
        }
    }

    @Test
    @DisplayName("Should produce reproducible results with same seed")
    void testReproducibility() throws Exception {
        // Run 1
        Path output1 = tempDir.resolve("run1.json");
        ExperimentConfig config1 = ExperimentConfig.createV2(
            "msmarco", "data/msmarco_sample_with_paraphrases.jsonl",
            "minilm", 0.90, "UNIDIRECTIONAL", 0.30,
            42L, 50, false, "SEMANTIC",
            10, 10000, 3600L, 1, 0.0, 0.0, output1.toString()
        );
        benchmarkRunner.run(config1);
        
        // Run 2 (same seed)
        cacheService.clearCache();
        Path output2 = tempDir.resolve("run2.json");
        ExperimentConfig config2 = ExperimentConfig.createV2(
            "msmarco", "data/msmarco_sample_with_paraphrases.jsonl",
            "minilm", 0.90, "UNIDIRECTIONAL", 0.30,
            42L, 50, false, "SEMANTIC",
            10, 10000, 3600L, 1, 0.0, 0.0, output2.toString()
        );
        benchmarkRunner.run(config2);
        
        // Assert: Results should be identical (within floating point precision)
        String content1 = Files.readString(output1);
        String content2 = Files.readString(output2);
        
        // Extract hit rates (should be identical)
        double hitRate1 = extractHitRate(content1);
        double hitRate2 = extractHitRate(content2);
        
        assertThat(hitRate1).isEqualTo(hitRate2);
    }

    private double extractHitRate(String json) {
        // Simple regex extraction (in real code, use JSON parser)
        String pattern = "\"hitRate\":(\\d+\\.\\d+)";
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(pattern);
        java.util.regex.Matcher m = p.matcher(json);
        if (m.find()) {
            return Double.parseDouble(m.group(1));
        }
        return 0.0;
    }
}
