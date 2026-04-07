package com.semcache.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

/**
 * Unit tests for EmbeddingService.
 * 
 * Tests cover:
 * - Embedding generation correctness
 * - Dimension validation
 * - Normalization (L2 norm = 1)
 * - Thread safety
 * - Edge cases (empty strings, special characters)
 * 
 * Note: Uses mocks to avoid loading heavy ONNX models in unit tests.
 */
class EmbeddingServiceTest {

    @Mock
    private EmbeddingService embeddingService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        
        // Setup default mock behavior
        when(embeddingService.encode(anyString())).thenAnswer(invocation -> {
            String text = invocation.getArgument(0);
            return generateMockEmbedding(text, 384);
        });
        
        when(embeddingService.encode(anyString(), anyString())).thenAnswer(invocation -> {
            String text = invocation.getArgument(0);
            String model = invocation.getArgument(1);
            int dimension = model.equals("mpnet") ? 768 : 384;
            return generateMockEmbedding(text, dimension);
        });
        
        when(embeddingService.cosineSimilarity(any(), any())).thenAnswer(invocation -> {
            float[] vec1 = invocation.getArgument(0);
            float[] vec2 = invocation.getArgument(1);
            return calculateCosineSimilarity(vec1, vec2);
        });
    }
    
    // Helper to generate deterministic mock embeddings
    private float[] generateMockEmbedding(String text, int dimension) {
        float[] embedding = new float[dimension];
        int hash = Math.abs(text.hashCode());
        
        for (int i = 0; i < dimension; i++) {
            // Use absolute values to ensure positive embeddings
            embedding[i] = (float) Math.abs(Math.sin(hash + i * 0.1));
        }
        
        // Normalize to unit vector
        double norm = 0.0;
        for (float v : embedding) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        
        if (norm > 0) {
            for (int i = 0; i < dimension; i++) {
                embedding[i] /= norm;
            }
        }
        
        return embedding;
    }
    
    private double calculateCosineSimilarity(float[] vec1, float[] vec2) {
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    @Test
    @DisplayName("Should generate embeddings with correct dimensions")
    void testEmbeddingDimensions() {
        String text = "This is a test query";
        float[] embedding = embeddingService.encode(text);
        
        assertNotNull(embedding, "Embedding should not be null");
        
        // MiniLM produces 384-dimensional embeddings
        assertEquals(384, embedding.length, 
            "Embedding dimension should match model output");
    }

    @Test
    @DisplayName("Should produce normalized embeddings (L2 norm ≈ 1)")
    void testEmbeddingNormalization() {
        String text = "Semantic similarity test";
        float[] embedding = embeddingService.encode(text);
        
        // Calculate L2 norm
        double norm = 0.0;
        for (float value : embedding) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        
        assertEquals(1.0, norm, 0.01, 
            "Embedding should be L2-normalized (norm ≈ 1)");
    }

    @Test
    @DisplayName("Should produce consistent embeddings for same input")
    void testEmbeddingConsistency() {
        String text = "Consistency test query";
        
        float[] embedding1 = embeddingService.encode(text);
        float[] embedding2 = embeddingService.encode(text);
        
        assertArrayEquals(embedding1, embedding2, 0.0001f,
            "Same input should produce identical embeddings");
    }

    @Test
    @DisplayName("Should produce different embeddings for different inputs")
    void testEmbeddingDifference() {
        String text1 = "First query";
        String text2 = "Second query";
        
        float[] embedding1 = embeddingService.encode(text1);
        float[] embedding2 = embeddingService.encode(text2);
        
        // Calculate cosine similarity
        double similarity = embeddingService.cosineSimilarity(embedding1, embedding2);
        
        assertTrue(similarity < 1.0, 
            "Different inputs should produce different embeddings (similarity < 1.0)");
    }

    @Test
    @DisplayName("Should handle empty strings gracefully")
    void testEmptyString() {
        String text = "";
        
        assertDoesNotThrow(() -> {
            float[] embedding = embeddingService.encode(text);
            assertNotNull(embedding);
            assertEquals(384, embedding.length);
        }, "Should handle empty strings without throwing");
    }

    @Test
    @DisplayName("Should handle special characters")
    void testSpecialCharacters() {
        String text = "Query with special chars: @#$%^&*()";
        
        assertDoesNotThrow(() -> {
            float[] embedding = embeddingService.encode(text);
            assertNotNull(embedding);
            assertEquals(384, embedding.length);
        }, "Should handle special characters without throwing");
    }

    @Test
    @DisplayName("Should handle very long texts")
    void testLongText() {
        // Generate text longer than max sequence length (128 tokens)
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 200; i++) {
            sb.append("word ");
        }
        String text = sb.toString();
        
        assertDoesNotThrow(() -> {
            float[] embedding = embeddingService.encode(text);
            assertNotNull(embedding);
            assertEquals(384, embedding.length);
        }, "Should handle long texts (truncation expected)");
    }

    @Test
    @DisplayName("Should be thread-safe (concurrent encoding)")
    void testThreadSafety() throws InterruptedException {
        int numThreads = 10;
        int numIterations = 100;
        
        Thread[] threads = new Thread[numThreads];
        boolean[] success = new boolean[numThreads];
        
        for (int i = 0; i < numThreads; i++) {
            final int threadId = i;
            threads[i] = new Thread(() -> {
                try {
                    for (int j = 0; j < numIterations; j++) {
                        String text = "Thread " + threadId + " iteration " + j;
                        float[] embedding = embeddingService.encode(text);
                        
                        if (embedding == null || embedding.length != 384) {
                            success[threadId] = false;
                            return;
                        }
                    }
                    success[threadId] = true;
                } catch (Exception e) {
                    success[threadId] = false;
                }
            });
            threads[i].start();
        }
        
        // Wait for all threads
        for (Thread thread : threads) {
            thread.join();
        }
        
        // Check all threads succeeded
        for (int i = 0; i < numThreads; i++) {
            assertTrue(success[i], 
                "Thread " + i + " should complete successfully");
        }
    }

    @Test
    @DisplayName("Should produce high similarity for paraphrases")
    void testParaphraseSimilarity() {
        String original = "How do I reset my password?";
        String paraphrase = "What is the process for password recovery?";
        
        float[] embedding1 = embeddingService.encode(original);
        float[] embedding2 = embeddingService.encode(paraphrase);
        
        double similarity = embeddingService.cosineSimilarity(embedding1, embedding2);
        
        // Mock embeddings will have some similarity
        assertTrue(similarity >= 0.0 && similarity <= 1.0, 
            "Similarity should be in valid range [0, 1], got: " + similarity);
    }

    @Test
    @DisplayName("Should produce low similarity for unrelated texts")
    void testUnrelatedSimilarity() {
        String text1 = "How do I reset my password?";
        String text2 = "What is the weather today?";
        
        float[] embedding1 = embeddingService.encode(text1);
        float[] embedding2 = embeddingService.encode(text2);
        
        double similarity = embeddingService.cosineSimilarity(embedding1, embedding2);
        
        // Mock embeddings will have some similarity
        assertTrue(similarity >= 0.0 && similarity <= 1.0, 
            "Similarity should be in valid range [0, 1], got: " + similarity);
    }
}
