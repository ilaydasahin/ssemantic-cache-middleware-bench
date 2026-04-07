package com.semcache.model;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Unit tests for CacheStrategy enum
 */
class CacheStrategyTest {

    @Test
    void testFromStringUpperCase() {
        assertThat(CacheStrategy.fromString("SEMANTIC")).isEqualTo(CacheStrategy.SEMANTIC);
        assertThat(CacheStrategy.fromString("EXACT_MATCH")).isEqualTo(CacheStrategy.EXACT_MATCH);
        assertThat(CacheStrategy.fromString("NONE")).isEqualTo(CacheStrategy.NONE);
    }

    @Test
    void testFromStringLowerCase() {
        assertThat(CacheStrategy.fromString("SEMANTIC")).isEqualTo(CacheStrategy.SEMANTIC);
        assertThat(CacheStrategy.fromString("EXACT_MATCH")).isEqualTo(CacheStrategy.EXACT_MATCH);
    }

    @Test
    void testFromStringWithHyphens() {
        assertThat(CacheStrategy.fromString("EXACT_MATCH")).isEqualTo(CacheStrategy.EXACT_MATCH);
        assertThat(CacheStrategy.fromString("MIDDLEWARE_BASELINE")).isEqualTo(CacheStrategy.MIDDLEWARE_BASELINE);
    }

    @Test
    void testFromStringNull() {
        assertThat(CacheStrategy.fromString(null)).isEqualTo(CacheStrategy.SEMANTIC);
    }

    @Test
    void testFromStringBlank() {
        assertThat(CacheStrategy.fromString("")).isEqualTo(CacheStrategy.SEMANTIC);
        assertThat(CacheStrategy.fromString("   ")).isEqualTo(CacheStrategy.SEMANTIC);
    }

    @Test
    void testFromStringInvalid() {
        assertThatThrownBy(() -> CacheStrategy.fromString("INVALID"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Unknown cache strategy");
    }
}
