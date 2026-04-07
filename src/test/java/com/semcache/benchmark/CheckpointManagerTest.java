package com.semcache.benchmark;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.nio.file.Path;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for CheckpointManager
 */
class CheckpointManagerTest {

    @TempDir
    Path tempDir;

    private CheckpointManager manager;

    @BeforeEach
    void setUp() {
        manager = new CheckpointManager();
    }

    @Test
    void testSaveAndLoadCheckpoint() {
        String experimentId = "test_exp_1";
        CheckpointManager.Checkpoint checkpoint = new CheckpointManager.Checkpoint(
                experimentId, "msmarco", 42, 0.85, 1000);
        checkpoint.completedQueryIndices.add(0);
        checkpoint.completedQueryIndices.add(1);
        checkpoint.completedQueryIndices.add(2);

        manager.saveCheckpoint(checkpoint);

        CheckpointManager.Checkpoint loaded = manager.loadCheckpoint(experimentId);
        assertThat(loaded).isNotNull();
        assertThat(loaded.experimentId).isEqualTo(experimentId);
        assertThat(loaded.dataset).isEqualTo("msmarco");
        assertThat(loaded.seed).isEqualTo(42);
        assertThat(loaded.threshold).isEqualTo(0.85);
        assertThat(loaded.totalQueries).isEqualTo(1000);
        assertThat(loaded.completedQueryIndices).containsExactlyInAnyOrder(0, 1, 2);

        manager.deleteCheckpoint(experimentId);
    }

    @Test
    void testHasCheckpoint() {
        String experimentId = "test_exp_2";
        assertThat(manager.hasCheckpoint(experimentId)).isFalse();

        CheckpointManager.Checkpoint checkpoint = new CheckpointManager.Checkpoint(
                experimentId, "nq", 123, 0.90, 500);
        manager.saveCheckpoint(checkpoint);

        assertThat(manager.hasCheckpoint(experimentId)).isTrue();

        manager.deleteCheckpoint(experimentId);
        assertThat(manager.hasCheckpoint(experimentId)).isFalse();
    }

    @Test
    void testGenerateExperimentId() {
        String id = CheckpointManager.generateExperimentId("msmarco", 42, 0.85);
        // Note: Locale-dependent formatting may use comma instead of dot
        assertThat(id).matches("msmarco_seed42_t0[.,]85");
    }

    @Test
    void testLoadNonExistentCheckpoint() {
        CheckpointManager.Checkpoint loaded = manager.loadCheckpoint("non_existent");
        assertThat(loaded).isNull();
    }
}
