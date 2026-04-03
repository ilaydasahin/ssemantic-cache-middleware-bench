package com.semcache.model;

import java.util.Arrays;

/**
 * Enumeration of supported cache lookup strategies.
 *
 * <p>Replaces magic-string comparisons scattered across the codebase.
 * Each constant maps to exactly one {@link com.semcache.service.CacheLookupStrategy}
 * implementation.
 */
public enum CacheStrategy {

    /** Embedding-based cosine similarity lookup (Algorithm 1, §3.1). */
    SEMANTIC,

    /** Two-tier cascaded lookup: MiniLM recall + MPNet verification (§5.7). */
    HYBRID,

    /** O(1) normalised-string exact match — baseline comparator. */
    EXACT_MATCH,

    /** Simulates middleware overhead (15 ms penalty) atop semantic lookup. */
    MIDDLEWARE_BASELINE,

    /** Cache disabled — all queries go to LLM. Control baseline. */
    NONE;

    /**
     * Parses a strategy name from configuration or CLI input.
     *
     * <p>Accepts case-insensitive input and converts hyphens to underscores
     * so that both {@code "EXACT_MATCH"} and {@code "exact-match"} are valid.
     *
     * @param value the raw configuration string (may be {@code null})
     * @return the matching strategy; defaults to {@link #SEMANTIC} if {@code value} is blank
     * @throws IllegalArgumentException if {@code value} is non-blank but unrecognised
     */
    public static CacheStrategy fromString(String value) {
        if (value == null || value.isBlank()) {
            return SEMANTIC;
        }
        try {
            return valueOf(value.toUpperCase().replace("-", "_"));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unknown cache strategy: '" + value +
                    "'. Valid values: " + Arrays.toString(values()));
        }
    }
}
