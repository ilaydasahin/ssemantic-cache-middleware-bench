package com.semcache.benchmark;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class NoiseGeneratorTest {

    private final NoiseGenerator generator = new NoiseGenerator();

    @Test
    public void testZeroProbabilityDoesNotChangeText() {
        String original = "This is a test query for semantic caching.";
        String result = generator.injectNoise(original, 0.0, 42L);
        assertEquals(original, result);
    }

    @Test
    public void testHighProbabilityInjectsNoise() {
        String original = "This is a very long text that must be mutated by the noise generator because the probability is very high.";
        String result = generator.injectNoise(original, 1.0, 42L);
        assertNotEquals(original, result);
    }

    @Test
    public void testReproducibilityWithSameSeed() {
        String original = "Consistency is key in scientific experiments.";
        long seed = 12345L;
        
        String result1 = generator.injectNoise(original, 1.0, seed);
        String result2 = generator.injectNoise(original, 1.0, seed);
        
        assertEquals(result1, result2, "Same seed and probability should produce identical noise patterns");
    }

    @Test
    public void testDifferentSeedsProduceDifferentResults() {
        String original = "Consistency is key in scientific experiments.";
        
        String result1 = generator.injectNoise(original, 1.0, 42L);
        String result2 = generator.injectNoise(original, 1.0, 99L);
        
        // At 1.0 probability, they should both be mutated, but differently due to different seeds
        assertNotEquals(result1, result2);
    }
}
