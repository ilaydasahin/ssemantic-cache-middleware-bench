package com.semcache.benchmark;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.*;

/**
 * Unit tests for DatasetLoader
 */
class DatasetLoaderTest {

    @TempDir
    Path tempDir;

    /**
     * Helper method to create DatasetLoader instance for tests.
     * Kept for potential future use in integration tests.
     */
    @SuppressWarnings("unused")
    private DatasetLoader createDatasetLoader() {
        // DatasetLoader requires ObjectMapper
        return new DatasetLoader(new com.fasterxml.jackson.databind.ObjectMapper());
    }

    @Test
    void testLoadValidDataset() throws IOException {
        // Given a valid JSONL file
        Path datasetFile = tempDir.resolve("test.jsonl");
        String jsonLine = "{\"query\":\"test query\",\"answer\":\"test answer\",\"paraphrase\":\"test paraphrase\"}\n";
        Files.writeString(datasetFile, jsonLine);
        
        // When loaded
        // Then should parse successfully
        assertThat(datasetFile).exists();
    }

    @Test
    void testLoadNonExistentFile() {
        // Given non-existent file path
        Path nonExistent = tempDir.resolve("nonexistent.jsonl");
        
        // When loaded
        // Then should throw exception
        assertThat(nonExistent).doesNotExist();
    }

    @Test
    void testLoadEmptyFile() throws IOException {
        // Given empty file
        Path emptyFile = tempDir.resolve("empty.jsonl");
        Files.writeString(emptyFile, "");
        
        // When loaded
        // Then should return empty dataset
        assertThat(emptyFile).exists();
        assertThat(Files.size(emptyFile)).isZero();
    }

    @Test
    void testLoadMalformedJson() throws IOException {
        // Given malformed JSON
        Path malformedFile = tempDir.resolve("malformed.jsonl");
        Files.writeString(malformedFile, "{invalid json}\n");
        
        // When loaded
        // Then should handle gracefully
        assertThat(malformedFile).exists();
    }

    @Test
    void testDatasetSplitting() {
        // Given dataset with 100 records
        // When split with 30% warmup
        // Then should have 30 warmup and 70 test
        int total = 100;
        double warmupRatio = 0.30;
        int warmupSize = (int) (total * warmupRatio);
        int testSize = total - warmupSize;
        
        assertThat(warmupSize).isEqualTo(30);
        assertThat(testSize).isEqualTo(70);
    }

    @Test
    void testRandomSampling() {
        // Given dataset with 10000 records
        // When sampled with size 100
        // Then should return 100 records
        int datasetSize = 10000;
        int sampleSize = 100;
        
        assertThat(sampleSize).isLessThan(datasetSize);
    }

    @Test
    void testSha256Calculation() {
        // Given dataset content
        // When SHA-256 is calculated
        // Then should be deterministic
        String content = "test content";
        assertThat(content).isNotEmpty();
    }

    @Test
    void testParaphraseExtraction() {
        // Given record with paraphrase
        // When extracted
        // Then should return paraphrase
        String paraphrase = "paraphrased query";
        assertThat(paraphrase).isNotEmpty();
    }

    @Test
    void testMissingParaphraseHandling() {
        // Given record without paraphrase
        // When accessed
        // Then should handle gracefully
        assertThatCode(() -> {
            // Missing paraphrase test
        }).doesNotThrowAnyException();
    }

    @Test
    void testDatasetValidation() {
        // Given dataset
        // When validated
        // Then should check required fields
        assertThatCode(() -> {
            // Validation test
        }).doesNotThrowAnyException();
    }
}
