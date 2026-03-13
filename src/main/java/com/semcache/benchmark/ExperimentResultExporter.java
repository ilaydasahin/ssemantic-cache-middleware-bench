package com.semcache.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.*;

/**
 * Exports experiment results to structured, self-describing output files.
 *
 * <p>
 * <b>Scientific Purpose:</b> Every result file is self-contained — it embeds
 * the
 * full {@link ExperimentConfig} next to the metrics so that a reader of the
 * file alone can reconstruct the experimental conditions without consulting
 * shell scripts or source code (Reproducibility Level 2, M.8).
 *
 * <p>
 * <b>Output format:</b> Pretty-printed JSON containing:
 * <ol>
 * <li>{@code config} — the complete {@link ExperimentConfig} for this run</li>
 * <li>{@code metrics} — {@link MetricsCollector.AggregateMetrics} from the test
 * phase</li>
 * <li>{@code queryLogs} — per-query observations, consumed by
 * {@code analyze_results.py} to compute SBERT / ROUGE-L post-hoc</li>
 * </ol>
 */
@Component
public class ExperimentResultExporter {

    private static final Logger log = LoggerFactory.getLogger(ExperimentResultExporter.class);

    private final ObjectMapper prettyMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private final ObjectMapper simpleMapper = new ObjectMapper();
    private final io.micrometer.core.instrument.MeterRegistry meterRegistry;

    public ExperimentResultExporter(io.micrometer.core.instrument.MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Writes a complete experiment result to a JSON file.
     *
     * <p>
     * The output file is structured as a flat JSON object. The
     * {@code analyze_results.py} script reads these files to produce
     * publication-ready tables (Table 4) and Pareto plots (Figure 3).
     *
     * @param config    Experimental configuration that produced these results
     * @param metrics   Aggregate metrics from {@link MetricsCollector#compute()}
     * @param queryLogs Per-query observations; may be empty for throughput runs
     * @throws ExportException if the target directory cannot be created or the
     *                         file cannot be written
     */
    public void export(ExperimentConfig config,
            MetricsCollector.AggregateMetrics metrics,
            List<QueryLog> queryLogs) {

        if (config.outputFilePath() == null || config.outputFilePath().isBlank()) {
            log.warn("outputFilePath is null — skipping result export");
            return;
        }

        File outputFile = new File(config.outputFilePath());
        ensureParentDirectoryExists(outputFile);

        // Build the combined result envelope
        Map<String, Object> envelope = buildResultEnvelope(config, metrics);

        try {
            prettyMapper.writeValue(outputFile, envelope);
            log.info("Result written to: {}", outputFile.getAbsolutePath());
        } catch (Exception e) {
            throw new ExportException("Failed to write result to " + config.outputFilePath(), e);
        }

        // Write per-query logs to a companion .logs.jsonl file
        // (consumed by analyze_results.py for SBERT / ROUGE-L computation)
        if (!queryLogs.isEmpty()) {
            exportQueryLogs(config.outputFilePath(), queryLogs);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Builds the flat JSON envelope that represents one complete experimental run.
     *
     * <p>
     * The top-level keys mirror the fields expected by {@code analyze_results.py}.
     * Embedding the full config as a nested object allows the file to serve as
     * its own experimental notebook entry.
     */
    private Map<String, Object> buildResultEnvelope(
            ExperimentConfig config,
            MetricsCollector.AggregateMetrics m) {

        // Use a LinkedHashMap to preserve insertion order in the JSON output
        // — more readable for manual inspection
        Map<String, Object> envelope = new LinkedHashMap<>();

        // ── Identification ────────────────────────────────────────────────────
        envelope.put("experimentId", config.experimentId());
        envelope.put("runTimestamp", config.runTimestamp());

        // ── Experimental parameters (mirrors BenchmarkResult fields for backward
        // compat) ─
        envelope.put("dataset", config.datasetName());
        envelope.put("embeddingModel", config.embeddingModelName());
        envelope.put("threshold", config.similarityThreshold());
        envelope.put("warmupStrategy", config.warmupStrategy());
        envelope.put("seed", config.randomSeed());
        envelope.put("sampleSize", config.sampleSize());
        envelope.put("hnswEnabled", config.hnswEnabled());
        envelope.put("cacheStrategy", config.cacheStrategy());

        // ── Aggregate metrics ─────────────────────────────────────────────────
        envelope.put("totalQueries", m.totalQueries());
        envelope.put("cacheHits", m.cacheHits());
        envelope.put("cacheMisses", m.cacheMisses());
        envelope.put("hitRate", round2(m.hitRate()));
        envelope.put("costSavingsPercent", round2(m.costSavingsPercent()));
        envelope.put("p50LatencyMs", round2(m.p50LatencyMs()));
        envelope.put("p95LatencyMs", round2(m.p95LatencyMs()));
        envelope.put("p99LatencyMs", round2(m.p99LatencyMs()));
        envelope.put("avgEmbeddingLatencyMs", round2(m.avgEmbeddingLatencyMs()));
        envelope.put("avgLlmLatencyMs", round2(m.avgLlmLatencyMs()));
        envelope.put("memoryUsageMb", round2(m.memoryUsageMb()));

        // Post-hoc metrics placeholder (populated by analyze_results.py from
        // .logs.jsonl)
        envelope.put("avgBertScore", 0.0);
        envelope.put("avgRougeL", 0.0);

        // ── Full config sub-object (for reproducibility) ──────────────────────
        envelope.put("config", config);

        // ── Infrastructure Monitoring (M.6) ──────────────────────────────────
        Map<String, Object> infra = new LinkedHashMap<>();
        infra.put("lettucePoolWaitMaxMs", getMeterValue("lettuce.connection.acquire_wait", "max"));
        infra.put("lettucePoolWaitAvgMs", getMeterValue("lettuce.connection.acquire_wait", "avg"));
        envelope.put("infrastructureMetrics", infra);

        return envelope;
    }

    private double getMeterValue(String name, String statistic) {
        return meterRegistry.find(name).timer() != null
                ? (statistic.equals("max")
                        ? meterRegistry.find(name).timer().max(java.util.concurrent.TimeUnit.MILLISECONDS)
                        : meterRegistry.find(name).timer().mean(java.util.concurrent.TimeUnit.MILLISECONDS))
                : 0.0;
    }

    /**
     * Writes per-query observations to a companion JSONL file.
     *
     * <p>
     * Each line is a JSON object with {@code groundTruth} and
     * {@code generatedResponse} fields consumed by {@code analyze_results.py}
     * to compute SBERT cosine similarity and ROUGE-L scores post-hoc.
     *
     * @param resultFilePath Path of the primary result JSON; the logs file is
     *                       written alongside it with the {@code .logs.jsonl}
     *                       suffix
     * @param queryLogs      Per-query observations
     */
    private void exportQueryLogs(String resultFilePath, List<QueryLog> queryLogs) {
        String logsPath = resultFilePath.replace(".json", ".logs.jsonl");
        try (java.io.PrintWriter writer = new java.io.PrintWriter(
                new java.io.FileWriter(logsPath, false))) {
            for (QueryLog ql : queryLogs) {
                writer.println(simpleMapper.writeValueAsString(ql));
            }
            log.info("Query logs written ({} records): {}", queryLogs.size(), logsPath);
        } catch (Exception e) {
            log.warn("Could not write query logs to {}: {}", logsPath, e.getMessage());
        }
    }

    private void ensureParentDirectoryExists(File outputFile) {
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new ExportException("Cannot create output directory: " + parent);
        }
    }

    private double round2(double value) {
        return Math.round(value * 100.0) / 100.0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Value types
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Per-query observation suitable for post-hoc evaluation.
     *
     * @param query              Original (not paraphrase) query text
     * @param groundTruth        Reference answer from the dataset
     * @param generatedResponse  Response returned to the caller (from cache or LLM)
     * @param hit                Whether the response came from cache
     * @param similarityScore    Best cosine similarity found during lookup
     * @param totalLatencyMs     End-to-end wall-clock duration (ms)
     * @param embeddingLatencyMs ONNX encoding time (ms)
     * @param llmLatencyMs       LLM API wait time (0 on cache hit) (ms)
     */
    public record QueryLog(
            String query,
            String groundTruth,
            String generatedResponse,
            boolean hit,
            double similarityScore,
            long totalLatencyMs,
            long embeddingLatencyMs,
            long llmLatencyMs) {
    }

    /** Signals an unrecoverable failure during result export. */
    public static class ExportException extends RuntimeException {
        public ExportException(String message) {
            super(message);
        }

        public ExportException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
