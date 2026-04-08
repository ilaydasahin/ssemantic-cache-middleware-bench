package com.semcache.model;

/**
 * Enumeration of supported ONNX embedding models.
 * Replaces magic string constants throughout the project.
 */
public enum EmbeddingModelType {
    
    /** 384-dimensional model optimized for short sequence representations. */
    MINILM("all-MiniLM-L6-v2", 384, 128),
    
    /** 768-dimensional model for generalized, high-quality representations. */
    MPNET("all-mpnet-base-v2", 768, 384),
    
    /** 312-dimensional, extremely fast model for constrained environments. */
    TINYBERT("paraphrase-TinyBERT-L6-v2", 312, 128),
    
    /** 384-dimensional multilingual model supporting 50+ languages (Q1: Multi-language support). */
    MULTILINGUAL_MINILM("paraphrase-multilingual-MiniLM-L12-v2", 384, 128);

    private final String directoryName;
    private final int dimension;
    private final int maxSequenceLength;

    EmbeddingModelType(String dir, int dim, int maxLen) {
        this.directoryName = dir;
        this.dimension = dim;
        this.maxSequenceLength = maxLen;
    }

    public String directoryName() { return directoryName; }
    public int dimension() { return dimension; }
    public int maxSequenceLength() { return maxSequenceLength; }

    public static EmbeddingModelType fromString(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Model name cannot be blank");
        }
        try {
            return valueOf(name.toUpperCase().replace("-", ""));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unknown embedding model: " + name + 
                    ". Valid values: " + java.util.Arrays.toString(values()));
        }
    }
}
