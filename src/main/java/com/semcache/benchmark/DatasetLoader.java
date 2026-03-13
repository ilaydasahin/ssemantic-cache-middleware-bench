package com.semcache.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.*;

/**
 * Responsible for loading, validating, and sampling benchmark datasets.
 *
 * <p>
 * <b>Scientific Purpose:</b> Centralises all data-ingestion logic so that
 * data-loading decisions are explicit, auditable, and decoupled from the
 * experimental pipeline. Every call logs the dataset checksum (SHA-256),
 * enabling reviewers to verify that results were obtained from the correct
 * dataset version.
 *
 * <p>
 * <b>Supported dataset format:</b> Newline-delimited JSON (JSONL) where
 * each record contains at minimum {@code "query"} (or {@code "question"}) and
 * {@code "answer"} (or {@code "response"}) keys. An optional
 * {@code "paraphrase"}
 * key supplies a semantic near-duplicate for the query, used to stress-test the
 * semantic lookup path.
 *
 * <p>
 * <b>Reproducibility contract:</b> Given the same {@code datasetPath},
 * {@code sampleSize}, and {@code randomSeed}, this class always returns the
 * same ordered list of {@link DatasetRecord} objects.
 */
@Component
public class DatasetLoader {

    private static final Logger log = LoggerFactory.getLogger(DatasetLoader.class);

    private final ObjectMapper objectMapper;

    public DatasetLoader(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Loads the full dataset from a JSONL file and validates its integrity.
     *
     * <p>
     * Logs the dataset path, record count, and SHA-256 fingerprint so that
     * every experiment log contains sufficient information to verify the data
     * version used.
     *
     * @param datasetPath Filesystem path to the JSONL dataset file
     * @return Ordered, unmodifiable list of {@link DatasetRecord} objects
     * @throws DatasetLoadException if the file is missing, empty, or malformed
     */
    public List<DatasetRecord> load(String datasetPath) {
        Path filePath = Paths.get(datasetPath);
        validateFileExists(filePath, datasetPath);

        List<DatasetRecord> records = parseJsonlRecords(filePath);
        String sha256 = computeSha256(filePath);

        validateNonEmpty(records, datasetPath);

        log.info("Dataset loaded: path='{}', records={}, SHA-256={}",
                datasetPath, records.size(), sha256);

        return Collections.unmodifiableList(records);
    }

    /**
     * Applies deterministic shuffling and optional pilot-study sampling to a
     * dataset.
     *
     * <p>
     * <b>Experiment Design Note:</b> Shuffling with a fixed seed ensures that
     * warmup / test splits are identical across independent runs, which is a
     * prerequisite for valid multi-seed statistical comparison (Wilcoxon
     * signed-rank test, §5.3).
     *
     * @param dataset    Full dataset returned by {@link #load(String)}
     * @param randomSeed Reproducibility seed — must match the value logged in
     *                   {@link ExperimentConfig}
     * @param sampleSize Optional upper bound on the number of records returned.
     *                   {@code null} or a value ≥ {@code dataset.size()} returns
     *                   the full (shuffled) dataset.
     * @return A new, mutable {@code ArrayList} containing the sampled records
     *         in shuffled order
     */
    public List<DatasetRecord> shuffleAndSample(List<DatasetRecord> dataset,
            long randomSeed,
            Integer sampleSize) {
        // Step 1: Copy to mutable list — do not mutate the caller's list
        List<DatasetRecord> mutableCopy = new ArrayList<>(dataset);

        // Step 2: Deterministic shuffle — same seed → same order across runs
        Collections.shuffle(mutableCopy, new Random(randomSeed));

        // Step 3: Optional pilot-study sampling
        // If sampleSize is set, truncate to that many records.
        // subList is wrapped in new ArrayList to avoid view-mutation risks.
        if (sampleSize != null && sampleSize > 0 && sampleSize < mutableCopy.size()) {
            mutableCopy = new ArrayList<>(mutableCopy.subList(0, sampleSize));
            log.info("Pilot sampling applied: {} of {} records selected (seed={})",
                    sampleSize, dataset.size(), randomSeed);
        } else {
            log.info("Full dataset used: {} records (seed={})", mutableCopy.size(), randomSeed);
        }

        return mutableCopy;
    }

    /**
     * Partitions a shuffled dataset into warmup and test splits.
     *
     * <p>
     * <b>Experiment Design Note:</b> The warmup set is used to pre-populate
     * the semantic cache before measurement begins. The test set drives all
     * reported metrics. Only ORIGINALS are placed in the warmup set; paraphrases
     * are reserved for the test phase to ensure the test exercises the semantic
     * lookup path rather than the O(1) exact-match path (Threat to Validity §5.2).
     *
     * @param shuffledDataset The output of {@link #shuffleAndSample}
     * @param warmupRatio     Fraction of records assigned to warmup; must be in (0,
     *                        1)
     * @return A {@link DatasetSplit} containing both non-overlapping subsets
     * @throws IllegalArgumentException if {@code warmupRatio} is out of (0, 1)
     */
    public DatasetSplit split(List<DatasetRecord> shuffledDataset, double warmupRatio) {
        validateWarmupRatio(warmupRatio);

        int warmupSize = (int) Math.round(shuffledDataset.size() * warmupRatio);
        // Guard: ensure test set is never empty
        warmupSize = Math.min(warmupSize, shuffledDataset.size() - 1);

        List<DatasetRecord> warmupSet = new ArrayList<>(shuffledDataset.subList(0, warmupSize));
        List<DatasetRecord> testSet = new ArrayList<>(shuffledDataset.subList(warmupSize, shuffledDataset.size()));

        log.info("Dataset split: warmup={} ({:.0f}%), test={} ({:.0f}%)",
                warmupSet.size(), warmupRatio * 100,
                testSet.size(), (1 - warmupRatio) * 100);

        return new DatasetSplit(warmupSet, testSet);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    private List<DatasetRecord> parseJsonlRecords(Path filePath) {
        List<DatasetRecord> records = new ArrayList<>();
        int lineNumber = 0;

        try (BufferedReader reader = Files.newBufferedReader(filePath)) {
            String line;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                line = line.trim();
                if (line.isEmpty())
                    continue;

                try {
                    Map<String, Object> entry = objectMapper.readValue(
                            line, new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {
                            });

                    String query = extractStringField(entry, "query", "question");
                    String answer = extractStringField(entry, "answer", "response");

                    if (query.isEmpty() || query.equals("null")) {
                        log.warn("Skipping record at line {}: missing or null query field", lineNumber);
                        continue;
                    }

                    Object paraphraseRaw = entry.get("paraphrase");
                    String paraphrase = (paraphraseRaw != null && !paraphraseRaw.toString().isEmpty())
                            ? paraphraseRaw.toString()
                            : null;

                    records.add(new DatasetRecord(query, answer, paraphrase));

                } catch (Exception e) {
                    log.warn("Skipping malformed JSON at line {}: {}", lineNumber, e.getMessage());
                }
            }
        } catch (Exception e) {
            throw new DatasetLoadException("Failed to read dataset file: " + filePath, e);
        }

        return records;
    }

    private String extractStringField(Map<String, Object> entry, String... keys) {
        for (String key : keys) {
            Object val = entry.get(key);
            if (val != null && !val.toString().isEmpty())
                return val.toString();
        }
        return "";
    }

    /**
     * Computes the SHA-256 fingerprint of the raw dataset file.
     *
     * <p>
     * Purpose: enables reviewers to verify that the exact same data file
     * was used across different experimental runs or between the authors and
     * independent replicators.
     */
    private String computeSha256(Path filePath) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = Files.readAllBytes(filePath);
            byte[] hash = digest.digest(bytes);
            StringBuilder hex = new StringBuilder();
            for (byte b : hash)
                hex.append(String.format("%02x", b));
            return hex.substring(0, 16) + "…"; // Truncated for log brevity
        } catch (Exception e) {
            log.warn("Could not compute SHA-256 for {}: {}", filePath, e.getMessage());
            return "unavailable";
        }
    }

    private void validateFileExists(Path filePath, String datasetPath) {
        if (!Files.exists(filePath)) {
            throw new DatasetLoadException("Dataset file not found: " + datasetPath);
        }
    }

    private void validateNonEmpty(List<DatasetRecord> records, String datasetPath) {
        if (records.isEmpty()) {
            throw new DatasetLoadException("Dataset is empty or fully malformed: " + datasetPath);
        }
    }

    private void validateWarmupRatio(double warmupRatio) {
        if (warmupRatio <= 0.0 || warmupRatio >= 1.0) {
            throw new IllegalArgumentException(
                    "warmupRatio must be in (0, 1); got: " + warmupRatio);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Value types
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Immutable representation of a single benchmark query pair.
     *
     * @param query      Original query text (used for cache warming and LLM
     *                   fallback)
     * @param answer     Ground-truth reference answer (used for SBERT / ROUGE-L
     *                   evaluation)
     * @param paraphrase Optional semantic near-duplicate of {@code query};
     *                   when present, used as the test-time query to stress the
     *                   semantic retrieval path
     */
    public record DatasetRecord(String query, String answer, String paraphrase) {
        /** Convenience constructor for records without a paraphrase. */
        public DatasetRecord(String query, String answer) {
            this(query, answer, null);
        }

        /** @return {@code true} if a paraphrase is available for this record */
        public boolean hasParaphrase() {
            return paraphrase != null && !paraphrase.isBlank();
        }
    }

    /**
     * Non-overlapping warmup / test partition of a shuffled dataset.
     *
     * @param warmupSet Records used to pre-populate the cache before measurement
     * @param testSet   Records used for all reported experimental measurements
     */
    public record DatasetSplit(List<DatasetRecord> warmupSet, List<DatasetRecord> testSet) {
        /** @return Total number of records across both splits */
        public int totalSize() {
            return warmupSet.size() + testSet.size();
        }
    }

    /**
     * Checked exception for data loading failures.
     * Separates data errors from algorithmic errors in stack traces.
     */
    public static class DatasetLoadException extends RuntimeException {
        public DatasetLoadException(String message) {
            super(message);
        }

        public DatasetLoadException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
