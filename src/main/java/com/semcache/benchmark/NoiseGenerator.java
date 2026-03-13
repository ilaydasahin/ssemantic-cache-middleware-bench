package com.semcache.benchmark;

import org.springframework.stereotype.Component;
import java.util.Random;

/**
 * Generates synthetic noise (typos, swaps) to test adversarial robustness.
 * M.6 Gold Standard: Proves system resilience to near-duplicate lexical
 * variants.
 */
@Component
public class NoiseGenerator {

    /**
     * Injects noise into the input text with a given probability.
     */
    public String injectNoise(String text, double probability, long seed) {
        if (probability <= 0.0 || text == null || text.length() < 5) {
            return text;
        }

        Random seededRandom = new Random(seed);
        if (seededRandom.nextDouble() > probability) {
            return text;
        }

        // Apply one of 3 noise types randomly
        int noiseType = seededRandom.nextInt(3);
        return switch (noiseType) {
            case 0 -> injectTypo(text, seededRandom); // Character substitution
            case 1 -> injectSwap(text, seededRandom); // Adjacent Word swap
            case 2 -> injectDuplication(text, seededRandom); // Word duplication
            default -> text;
        };
    }

    private String injectTypo(String text, Random rand) {
        int pos = rand.nextInt(text.length());
        char original = text.charAt(pos);
        if (Character.isWhitespace(original))
            return text;

        char typo = (char) ('a' + rand.nextInt(26));
        return text.substring(0, pos) + typo + text.substring(pos + 1);
    }

    private String injectSwap(String text, Random rand) {
        String[] words = text.split("\\s+");
        if (words.length < 3)
            return text;

        int pos = rand.nextInt(words.length - 1);
        String temp = words[pos];
        words[pos] = words[pos + 1];
        words[pos + 1] = temp;

        return String.join(" ", words);
    }

    private String injectDuplication(String text, Random rand) {
        String[] words = text.split("\\s+");
        int pos = rand.nextInt(words.length);

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < words.length; i++) {
            sb.append(words[i]).append(" ");
            if (i == pos) {
                sb.append(words[i]).append(" ");
            }
        }
        return sb.toString().trim();
    }
}
