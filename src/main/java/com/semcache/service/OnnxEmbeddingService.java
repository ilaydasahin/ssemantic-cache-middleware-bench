package com.semcache.service;

import ai.onnxruntime.*;
import com.semcache.model.EmbeddingModelType;
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
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

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
        BlockingQueue<OrtSession> sessionPool;
        SimpleWordPieceTokenizer tokenizer;
        EmbeddingModelType modelType;
        Timer timer;
        int poolSize;
    }

    private final Map<String, ModelContext> modelRegistry = new HashMap<>();

    public OnnxEmbeddingService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    @PostConstruct
    public void init() {
        log.info("OnnxEmbeddingService initializing in multi-model mode...");
        
        // Validate configuration
        if (primaryModelName == null || primaryModelName.isEmpty()) {
            throw new IllegalStateException("Primary model name not configured");
        }
        if (maxLength <= 0 || maxLength > 512) {
            throw new IllegalStateException(
                    "Invalid max length: " + maxLength + " (must be in (0, 512])");
        }
        
        try {
            env = OrtEnvironment.getEnvironment();

            // Auto-load available models from the filesystem
            for (EmbeddingModelType type : EmbeddingModelType.values()) {
                tryLoadModel(type);
            }
            
            // Validate that at least one model was loaded
            if (modelRegistry.isEmpty()) {
                throw new IllegalStateException(
                        "No ONNX models loaded. Check that model files exist under models/. " +
                        "Run: bash scripts/fetch_embedding_assets.sh");
            }
            
            // Validate that primary model was loaded
            if (!modelRegistry.containsKey(primaryModelName.toLowerCase())) {
                log.warn("Primary model '{}' not found. Available models: {}", 
                        primaryModelName, modelRegistry.keySet());
                // Use first available model as fallback
                String fallback = modelRegistry.keySet().iterator().next();
                log.warn("Falling back to model: {}", fallback);
                primaryModelName = fallback;
            }

            log.info("OnnxEmbeddingService ready with {} models: {}",
                    modelRegistry.size(), modelRegistry.keySet());
            log.info("✅ Primary model: {} ({}d)", primaryModelName, 
                    modelRegistry.get(primaryModelName.toLowerCase()).modelType.dimension());
        } catch (Exception e) {
            log.error("Failed to initialize ONNX environment: {}", e.getMessage());
            throw new IllegalStateException("ONNX initialization failed", e);
        }
    }

    private void tryLoadModel(EmbeddingModelType type) {
        String name = type.name().toLowerCase();
        String modelPath = "models/" + type.directoryName() + "/model.onnx";
        String vocabPath = "models/" + type.directoryName() + "/vocab.txt";

        File modelFile = new File(modelPath);
        if (modelFile.exists()) {
            try {
                ModelContext ctx = new ModelContext();
                ctx.modelType = type;
                ctx.tokenizer = new SimpleWordPieceTokenizer(vocabPath);
                ctx.timer = Timer.builder("embedding.latency")
                        .tag("model", name)
                        .description("Time for " + name + " encoding")
                        .register(meterRegistry);

                // Thread-safe session pool: size = available processors
                ctx.poolSize = Math.max(2, Runtime.getRuntime().availableProcessors());
                ctx.sessionPool = new ArrayBlockingQueue<>(ctx.poolSize);

                OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
                for (int i = 0; i < ctx.poolSize; i++) {
                    ctx.sessionPool.offer(env.createSession(modelPath, sessionOptions));
                }

                modelRegistry.put(name, ctx);
                log.info("Loaded model context: {} ({}d, maxLen={}, poolSize={})", 
                        name, type.dimension(), type.maxSequenceLength(), ctx.poolSize);
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
            if (modelRegistry.isEmpty()) {
                throw new RuntimeException(
                        "No ONNX models loaded. Check that model files exist under models/. Requested: " + modelName);
            }
            log.error("Model not found in registry: {}. Defaulting to first available.", modelName);
            ctx = modelRegistry.values().iterator().next();
        }

        ModelContext finalCtx = ctx;
        return ctx.timer.record(() -> {
            OrtSession session = null;
            try {
                // Acquire session from pool with 60s timeout (increased for heavy load scenarios)
                session = finalCtx.sessionPool.poll(60, java.util.concurrent.TimeUnit.SECONDS);
                if (session == null) {
                    throw new RuntimeException("Failed to acquire ONNX session from pool (timeout after 60s). Pool size: " + finalCtx.poolSize);
                }
                return performInferenceWithSession(text, finalCtx, session);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while waiting for ONNX session", e);
            } finally {
                // Always return session to pool
                if (session != null) {
                    finalCtx.sessionPool.offer(session);
                }
            }
        });
    }
    
    /**
     * Cleanup ONNX resources on shutdown to prevent memory leaks.
     */
    @jakarta.annotation.PreDestroy
    public void shutdown() {
        log.info("Shutting down OnnxEmbeddingService...");
        for (Map.Entry<String, ModelContext> entry : modelRegistry.entrySet()) {
            String modelName = entry.getKey();
            ModelContext ctx = entry.getValue();
            try {
                // Drain and close all sessions in the pool
                OrtSession session;
                int closedCount = 0;
                while ((session = ctx.sessionPool.poll()) != null) {
                    session.close();
                    closedCount++;
                }
                log.info("Closed {} ONNX sessions for model: {}", closedCount, modelName);
            } catch (Exception e) {
                log.error("Failed to close ONNX sessions for model {}: {}", modelName, e.getMessage());
            }
        }
        log.info("OnnxEmbeddingService shutdown complete");
    }
    
    private float[] performInferenceWithSession(String text, ModelContext ctx, OrtSession session) {
        try {
            return performInference(text, ctx, session);
        } catch (Exception e) {
            log.error("ONNX inference failed: {}", e.getMessage());
            throw new RuntimeException("ONNX inference error", e);
        }
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
        return ctx != null ? ctx.modelType.dimension() : 384;
    }

    @Override
    public String getModelName() {
        return primaryModelName;
    }

    private float[] performInference(String text, ModelContext ctx, OrtSession session) {
        try {
            // T2: use per-model maxLength, fallback to global maxLength
            int seqLen = (ctx.modelType.maxSequenceLength() > 0) ? ctx.modelType.maxSequenceLength() : maxLength;
            List<Integer> tokenIds = ctx.tokenizer.tokenize(text, seqLen);
            long[] inputIds = tokenIds.stream().mapToLong(i -> i).toArray();
            long[] attentionMask = new long[seqLen];
            int tokenCount = tokenIds.size();
            for (int i = 0; i < seqLen; i++) {
                attentionMask[i] = (i < tokenCount && tokenIds.get(i) != 0) ? 1L : 0L;
            }

            long[] shape = { 1, seqLen };
            try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape);
                    OnnxTensor maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape)) {

                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", idsTensor);
                inputs.put("attention_mask", maskTensor);

                OnnxTensor ttidsTensor = null;
                try {
                    if (session.getInputNames().contains("token_type_ids")) {
                        long[] ttids = new long[seqLen];
                        ttidsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(ttids), shape);
                        inputs.put("token_type_ids", ttidsTensor);
                    }
                    try (OrtSession.Result results = session.run(inputs)) {
                        float[][][] outputData = (float[][][]) results.get(0).getValue();
                        return meanPooling(outputData[0], attentionMask, ctx.modelType.dimension());
                    }
                } finally {
                    if (ttidsTensor != null) {
                        ttidsTensor.close();
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Inference failed for " + ctx.modelType.name(), e);
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
                v[i] = (float)(v[i] / n); // divide in double precision, then cast
    }
}
