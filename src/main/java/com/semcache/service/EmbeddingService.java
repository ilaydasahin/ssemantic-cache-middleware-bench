package com.semcache.service;

/**
 * Interface for generating semantic embeddings.
 * Allows decoupling specific implementations (e.g., local ONNX vs remote API API).
 */
public interface EmbeddingService {

    /**
     * Encode a text query into a dense embedding vector.
     *
     * @param text the input query text
     * @return normalized embedding vector
     */
    float[] encode(String text);

    float[] encode(String text, String modelName);

    /**
     * Compute similarity (e.g. cosine or dot product) between two vectors.
     */
    double cosineSimilarity(float[] a, float[] b);

    int getEmbeddingDimension();

    int getEmbeddingDimension(String modelName);

    String getModelName();
}
