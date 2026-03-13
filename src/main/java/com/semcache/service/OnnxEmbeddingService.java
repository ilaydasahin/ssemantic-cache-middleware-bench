package com.semcache.service;

import ai.onnxruntime.*;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.LongBuffer;
import java.util.*;
import java.io.File;

/**
 * Enhanced Embedding Service — Supports multiple concurrent ONNX models.
 * 
 * Used for Hybrid Caching (§5.7): fast MiniLM for recall + MPNet for
 * verification.
 */
@Service
public class OnnxEmbeddingService implements EmbeddingService {

    private static final Logger log = LoggerFactory.getLogger(OnnxEmbeddingService.class);

    @Value("${embedding.model-name:minilm}")
    private String primaryModelName;

    @Value("${embedding.max-length:256}")
    private int maxLength;

    private final MeterRegistry meterRegistry;
    private OrtEnvironment env;

    private static class ModelContext {
        OrtSession session;
        SimpleWordPieceTokenizer tokenizer;
        int dimension;
        int maxLength; // T2: per-model max sequence length
        Timer timer;
        String name;
    }

    private final Map<String, ModelContext> modelRegistry = new HashMap<>();

    public OnnxEmbeddingService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    @PostConstruct
    public void init() {
        log.info("OnnxEmbeddingService initializing in multi-model mode...");
        try {
            env = OrtEnvironment.getEnvironment();

            // Auto-load available models from the filesystem
            String[] commonModels = { "minilm", "mpnet", "tinybert" };
            for (String m : commonModels) {
                tryLoadModel(m);
            }

            log.info("OnnxEmbeddingService ready with {} models: {}",
                    modelRegistry.size(), modelRegistry.keySet());
        } catch (Exception e) {
            log.error("Failed to initialize ONNX environment: {}", e.getMessage());
        }
    }

    private void tryLoadModel(String name) {
        String fullModelDir = switch (name) {
            case "mpnet" -> "all-mpnet-base-v2";
            case "tinybert" -> "paraphrase-TinyBERT-L6-v2";
            default -> "all-MiniLM-L6-v2";
        };

        int dim = switch (name) {
            case "mpnet" -> 768;
            case "tinybert" -> 312;
            default -> 384;
        };

        // T2 fix: per-model max-sequence-length (matches actual model architecture)
        // MiniLM/TinyBERT: 128 is their practical optimum; MPNet: supports up to 384
        int modelMaxLength = switch (name) {
            case "mpnet" -> 384;
            case "tinybert" -> 128;
            default -> 128; // minilm
        };

        String modelPath = "models/" + fullModelDir + "/model.onnx";
        String vocabPath = "models/" + fullModelDir + "/vocab.txt";

        File modelFile = new File(modelPath);
        if (modelFile.exists()) {
            try {
                ModelContext ctx = new ModelContext();
                ctx.name = name;
                ctx.dimension = dim;
                ctx.maxLength = modelMaxLength; // T2: per-model length
                ctx.session = env.createSession(modelPath, new OrtSession.SessionOptions());
                ctx.tokenizer = new SimpleWordPieceTokenizer(vocabPath);
                ctx.timer = Timer.builder("embedding.latency")
                        .tag("model", name)
                        .description("Time for " + name + " encoding")
                        .register(meterRegistry);

                modelRegistry.put(name, ctx);
                log.info("Loaded model context: {} ({}d, maxLen={})", name, dim, modelMaxLength);
            } catch (Exception e) {
                log.warn("Failed to load model {}: {}", name, e.getMessage());
            }
        }
    }

    @Override
    public float[] encode(String text) {
        return encode(text, primaryModelName);
    }

    @Override
    public float[] encode(String text, String modelName) {
        ModelContext ctx = modelRegistry.get(modelName.toLowerCase());
        if (ctx == null) {
            log.error("Model not found in registry: {}. Defaulting to first available.", modelName);
            ctx = modelRegistry.values().iterator().next();
        }

        ModelContext finalCtx = ctx;
        return ctx.timer.record(() -> performInference(text, finalCtx));
    }

    @Override
    public double cosineSimilarity(float[] a, float[] b) {
        // Assuming pre-normalized vectors as per our pooling logic
        double dot = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    @Override
    public int getEmbeddingDimension() {
        return getEmbeddingDimension(primaryModelName);
    }

    @Override
    public int getEmbeddingDimension(String modelName) {
        ModelContext ctx = modelRegistry.get(modelName.toLowerCase());
        return ctx != null ? ctx.dimension : 384;
    }

    @Override
    public String getModelName() {
        return primaryModelName;
    }

    private float[] performInference(String text, ModelContext ctx) {
        try {
            // T2: use per-model maxLength, fallback to global maxLength
            int seqLen = (ctx.maxLength > 0) ? ctx.maxLength : maxLength;
            List<Integer> tokenIds = ctx.tokenizer.tokenize(text, seqLen);
            long[] inputIds = tokenIds.stream().mapToLong(i -> i).toArray();
            long[] attentionMask = new long[seqLen];

            for (int i = 0; i < seqLen; i++) {
                attentionMask[i] = (i < tokenIds.size() && tokenIds.get(i) != 0) ? 1L : 0L;
            }

            long[] shape = { 1, seqLen };
            try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape);
                    OnnxTensor maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape)) {

                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", idsTensor);
                inputs.put("attention_mask", maskTensor);

                if (ctx.session.getInputNames().contains("token_type_ids")) {
                    long[] ttids = new long[seqLen];
                    inputs.put("token_type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(ttids), shape));
                }

                try (OrtSession.Result results = ctx.session.run(inputs)) {
                    float[][][] outputData = (float[][][]) results.get(0).getValue();
                    return meanPooling(outputData[0], attentionMask, ctx.dimension);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Inference failed for " + ctx.name, e);
        }
    }

    private float[] meanPooling(float[][] hiddenState, long[] mask, int dim) {
        // Fix #7: guard against model output being narrower than configured dim
        int actualDim = (hiddenState.length > 0) ? Math.min(dim, hiddenState[0].length) : dim;
        float[] pooled = new float[actualDim];
        int count = 0;
        for (int i = 0; i < hiddenState.length; i++) {
            if (mask[i] == 1) {
                for (int j = 0; j < actualDim; j++)
                    pooled[j] += hiddenState[i][j];
                count++;
            }
        }
        if (count > 0) {
            for (int j = 0; j < actualDim; j++)
                pooled[j] /= count;
        }
        normalize(pooled);
        return pooled;
    }

    private void normalize(float[] v) {
        double n = 0;
        for (float x : v)
            n += x * x;
        n = Math.sqrt(n);
        if (n > 0)
            for (int i = 0; i < v.length; i++)
                v[i] /= (float) n;
    }
}
