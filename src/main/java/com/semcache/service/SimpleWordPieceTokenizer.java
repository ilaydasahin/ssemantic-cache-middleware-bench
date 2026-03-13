package com.semcache.service;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * A lightweight WordPiece tokenizer implementation for Java.
 * Used to tokenize input text for transformer models (MiniLM, MPNet, TinyBERT).
 */
public class SimpleWordPieceTokenizer {

    private final Map<String, Integer> vocab = new HashMap<>();
    private final int unkTokenId;
    private final int clsTokenId;
    private final int sepTokenId;
    private final int padTokenId;

    public SimpleWordPieceTokenizer(String vocabPath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(vocabPath))) {
            String line;
            int id = 0;
            while ((line = reader.readLine()) != null) {
                vocab.put(line.trim(), id++);
            }
        }
        this.unkTokenId = vocab.getOrDefault("[UNK]", 100);
        this.clsTokenId = vocab.getOrDefault("[CLS]", 101);
        this.sepTokenId = vocab.getOrDefault("[SEP]", 102);
        this.padTokenId = vocab.getOrDefault("[PAD]", 0);
    }

    public List<Integer> tokenize(String text, int maxLength) {
        List<Integer> ids = new ArrayList<>();
        ids.add(clsTokenId);

        // T1 fix: split on whitespace AND punctuation so "hello,world" becomes "hello" + "world"
        String[] words = text.toLowerCase().split("[\\s\\p{Punct}]+");
        for (String word : words) {
            ids.addAll(wordPieceTokenize(word));
            if (ids.size() >= maxLength - 1)
                break;
        }

        if (ids.size() > maxLength - 1) {
            ids = ids.subList(0, maxLength - 1);
        }
        ids.add(sepTokenId);

        // Padding
        while (ids.size() < maxLength) {
            ids.add(padTokenId);
        }

        return ids;
    }

    private List<Integer> wordPieceTokenize(String word) {
        List<Integer> result = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            String curSubstr = null;
            while (start < end) {
                String substr = (start == 0) ? word.substring(start, end) : "##" + word.substring(start, end);
                if (vocab.containsKey(substr)) {
                    curSubstr = substr;
                    break;
                }
                end--;
            }

            if (curSubstr == null) {
                // T3 Fix: Character-level fallback instead of whole-word UNK
                // Prevents data loss on out-of-vocabulary or typo characters
                result.add(unkTokenId);
                start++;
                continue;
            }

            result.add(vocab.get(curSubstr));
            start = end;
        }
        return result;
    }
}
