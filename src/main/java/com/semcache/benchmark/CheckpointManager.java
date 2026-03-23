package com.semcache.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Checkpoint Manager - Saves progress and enables resume after quota exhaustion
 */
@Component
public class CheckpointManager {
    
    private static final Logger log = LoggerFactory.getLogger(CheckpointManager.class);
    private static final String CHECKPOINT_DIR = "checkpoints";
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    public static class Checkpoint {
        public String experimentId;
        public String dataset;
        public long seed;
        public double threshold;
        public Set<Integer> completedQueryIndices = new HashSet<>();
        public long lastUpdateTime;
        public int totalQueries;
        
        public Checkpoint() {}
        
        public Checkpoint(String experimentId, String dataset, long seed, double threshold, int totalQueries) {
            this.experimentId = experimentId;
            this.dataset = dataset;
            this.seed = seed;
            this.threshold = threshold;
            this.totalQueries = totalQueries;
            this.lastUpdateTime = System.currentTimeMillis();
        }
    }
    
    public CheckpointManager() {
        try {
            Files.createDirectories(Paths.get(CHECKPOINT_DIR));
            // Clean up old checkpoints (older than 7 days) to prevent disk space issues
            cleanupOldCheckpoints();
        } catch (IOException e) {
            log.error("Failed to create checkpoint directory", e);
        }
    }
    
    /**
     * Deletes checkpoint files older than 7 days to prevent disk space accumulation.
     * With 450K queries and frequent checkpoints, old files can accumulate to 2GB+.
     */
    private void cleanupOldCheckpoints() {
        try {
            File checkpointDir = new File(CHECKPOINT_DIR);
            if (!checkpointDir.exists()) {
                return;
            }
            
            long sevenDaysAgo = System.currentTimeMillis() - (7L * 24 * 60 * 60 * 1000);
            File[] files = checkpointDir.listFiles((dir, name) -> name.endsWith(".json"));
            
            if (files != null) {
                int deletedCount = 0;
                long freedBytes = 0;
                
                for (File file : files) {
                    if (file.lastModified() < sevenDaysAgo) {
                        long fileSize = file.length();
                        if (file.delete()) {
                            deletedCount++;
                            freedBytes += fileSize;
                        }
                    }
                }
                
                if (deletedCount > 0) {
                    log.info("Cleaned up {} old checkpoints, freed {} MB", 
                            deletedCount, freedBytes / (1024 * 1024));
                }
            }
        } catch (Exception e) {
            log.warn("Failed to cleanup old checkpoints: {}", e.getMessage());
        }
    }
    
    public void saveCheckpoint(Checkpoint checkpoint) {
        checkpoint.lastUpdateTime = System.currentTimeMillis();
        String filename = getCheckpointFilename(checkpoint.experimentId);
        try {
            // Periodic cleanup: every 100 saves, clean up old checkpoints
            if (checkpoint.completedQueryIndices.size() % 5000 == 0) {
                cleanupOldCheckpoints();
            }
            
            // Check disk space before save (require 500MB free)
            File checkpointDir = new File(CHECKPOINT_DIR);
            long freeSpaceMB = checkpointDir.getFreeSpace() / (1024 * 1024);
            if (freeSpaceMB < 500) {
                log.error("Insufficient disk space for checkpoint: {} MB free (require 500 MB)", freeSpaceMB);
                // Try emergency cleanup
                cleanupOldCheckpoints();
                freeSpaceMB = checkpointDir.getFreeSpace() / (1024 * 1024);
                if (freeSpaceMB < 500) {
                    throw new IOException("Disk space critically low: " + freeSpaceMB + " MB");
                }
            }
            
            // Atomic write: temp file + rename
            String tempFilename = filename + ".tmp";
            objectMapper.writerWithDefaultPrettyPrinter()
                    .writeValue(new File(tempFilename), checkpoint);
            
            // Backup previous checkpoint before overwriting
            File targetFile = new File(filename);
            if (targetFile.exists()) {
                File backupFile = new File(filename + ".backup");
                if (!targetFile.renameTo(backupFile)) {
                    log.warn("Failed to backup previous checkpoint");
                }
            }
            
            // Atomic rename
            File tempFile = new File(tempFilename);
            if (!tempFile.renameTo(targetFile)) {
                log.error("Failed to rename checkpoint temp file to {}", filename);
                tempFile.delete();
            } else {
                // Delete backup after successful save
                new File(filename + ".backup").delete();
                
                log.debug("Checkpoint saved: {} ({}/{} queries completed)", 
                        checkpoint.experimentId, 
                        checkpoint.completedQueryIndices.size(), 
                        checkpoint.totalQueries);
            }
        } catch (IOException e) {
            log.error("Failed to save checkpoint: {}", checkpoint.experimentId, e);
        }
    }
    
    public Checkpoint loadCheckpoint(String experimentId) {
        String filename = getCheckpointFilename(experimentId);
        File file = new File(filename);
        if (!file.exists()) {
            return null;
        }
        
        try {
            Checkpoint checkpoint = objectMapper.readValue(file, Checkpoint.class);
            log.info("✅ Checkpoint loaded: {} ({}/{} queries already completed)", 
                    experimentId, 
                    checkpoint.completedQueryIndices.size(), 
                    checkpoint.totalQueries);
            return checkpoint;
        } catch (IOException e) {
            log.error("Failed to load checkpoint: {}", experimentId, e);
            return null;
        }
    }
    
    public void deleteCheckpoint(String experimentId) {
        String filename = getCheckpointFilename(experimentId);
        try {
            Files.deleteIfExists(Paths.get(filename));
            log.info("Checkpoint deleted: {}", experimentId);
        } catch (IOException e) {
            log.error("Failed to delete checkpoint: {}", experimentId, e);
        }
    }
    
    public boolean hasCheckpoint(String experimentId) {
        return new File(getCheckpointFilename(experimentId)).exists();
    }
    
    private String getCheckpointFilename(String experimentId) {
        return CHECKPOINT_DIR + "/" + experimentId + ".json";
    }
    
    public static String generateExperimentId(String dataset, long seed, double threshold) {
        return String.format("%s_seed%d_t%.2f", dataset, seed, threshold);
    }
}
