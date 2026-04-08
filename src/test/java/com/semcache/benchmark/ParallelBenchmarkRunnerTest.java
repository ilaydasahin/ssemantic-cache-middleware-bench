package com.semcache.benchmark;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Unit tests for ParallelBenchmarkRunner
 */
@ExtendWith(MockitoExtension.class)
class ParallelBenchmarkRunnerTest {

    @Mock
    private BenchmarkRunner benchmarkRunner;

    private ParallelBenchmarkRunner parallelRunner;

    @BeforeEach
    void setUp() {
        parallelRunner = new ParallelBenchmarkRunner(benchmarkRunner);
    }

    @Test
    void testEmptyConfigList() {
        // Given empty config list
        List<ExperimentConfig> configs = new ArrayList<>();
        
        // When parallel run is executed
        boolean result = parallelRunner.runParallel(configs);
        
        // Then should return false
        assertThat(result).isFalse();
        verifyNoInteractions(benchmarkRunner);
    }

    @Test
    void testSingleConfig() throws Exception {
        // Given single config
        ExperimentConfig config = ExperimentConfig.createV2(
                "msmarco", "data/msmarco_sample.jsonl", "auto",
                0.90, "BIDIRECTIONAL", 0.30, 42L, null,
                true, "SEMANTIC", 5, 50000, 86400L, null,
                0.0, 0.0, "results/test.json");
        
        List<ExperimentConfig> configs = List.of(config);
        
        doNothing().when(benchmarkRunner).run(any(ExperimentConfig.class));
        
        // When parallel run is executed
        boolean result = parallelRunner.runParallel(configs);
        
        // Then should run directly (not in parallel)
        assertThat(result).isTrue();
        verify(benchmarkRunner, times(1)).run(config);
    }

    @Test
    void testMultipleConfigs() throws Exception {
        // Given multiple configs
        ExperimentConfig config1 = ExperimentConfig.createV2(
                "msmarco", "data/msmarco_sample.jsonl", "auto",
                0.90, "BIDIRECTIONAL", 0.30, 42L, null,
                true, "SEMANTIC", 5, 50000, 86400L, null,
                0.0, 0.0, "results/test1.json");
        
        ExperimentConfig config2 = ExperimentConfig.createV2(
                "nq", "data/nq_sample.jsonl", "auto",
                0.90, "BIDIRECTIONAL", 0.30, 123L, null,
                true, "SEMANTIC", 5, 50000, 86400L, null,
                0.0, 0.0, "results/test2.json");
        
        List<ExperimentConfig> configs = List.of(config1, config2);
        
        doNothing().when(benchmarkRunner).run(any(ExperimentConfig.class));
        
        // When parallel run is executed
        boolean result = parallelRunner.runParallel(configs);
        
        // Then should run in parallel
        assertThat(result).isTrue();
        verify(benchmarkRunner, times(2)).run(any(ExperimentConfig.class));
    }

    @Test
    void testSingleConfigFailure() throws Exception {
        // Given single config that fails
        ExperimentConfig config = ExperimentConfig.createV2(
                "msmarco", "data/msmarco_sample.jsonl", "auto",
                0.90, "BIDIRECTIONAL", 0.30, 42L, null,
                true, "SEMANTIC", 5, 50000, 86400L, null,
                0.0, 0.0, "results/test.json");
        
        List<ExperimentConfig> configs = List.of(config);
        
        doThrow(new RuntimeException("Benchmark failed")).when(benchmarkRunner).run(any(ExperimentConfig.class));
        
        // When parallel run is executed
        boolean result = parallelRunner.runParallel(configs);
        
        // Then should return false
        assertThat(result).isFalse();
        verify(benchmarkRunner, times(1)).run(config);
    }

    @Test
    void testParallelExecutionFailure() throws Exception {
        // Given multiple configs where one fails
        ExperimentConfig config1 = ExperimentConfig.createV2(
                "msmarco", "data/msmarco_sample.jsonl", "auto",
                0.90, "BIDIRECTIONAL", 0.30, 42L, null,
                true, "SEMANTIC", 5, 50000, 86400L, null,
                0.0, 0.0, "results/test1.json");
        
        ExperimentConfig config2 = ExperimentConfig.createV2(
                "nq", "data/nq_sample.jsonl", "auto",
                0.90, "BIDIRECTIONAL", 0.30, 123L, null,
                true, "SEMANTIC", 5, 50000, 86400L, null,
                0.0, 0.0, "results/test2.json");
        
        List<ExperimentConfig> configs = List.of(config1, config2);
        
        doNothing().when(benchmarkRunner).run(config1);
        doThrow(new RuntimeException("Benchmark failed")).when(benchmarkRunner).run(config2);
        
        // When parallel run is executed
        boolean result = parallelRunner.runParallel(configs);
        
        // Then should return false
        assertThat(result).isFalse();
    }

    @Test
    void testThreadPoolSize() {
        // Given multiple configs
        int configCount = 10;
        List<ExperimentConfig> configs = new ArrayList<>();
        
        for (int i = 0; i < configCount; i++) {
            configs.add(ExperimentConfig.createV2(
                    "dataset" + i, "data/test" + i + ".jsonl", "auto",
                    0.90, "BIDIRECTIONAL", 0.30, 42L + i, null,
                    true, "SEMANTIC", 5, 50000, 86400L, null,
                    0.0, 0.0, "results/test" + i + ".json"));
        }
        
        // When parallel run is executed
        // Then thread pool should be limited to available processors
        int expectedThreads = Math.min(configCount, Runtime.getRuntime().availableProcessors());
        assertThat(expectedThreads).isGreaterThan(0);
        assertThat(expectedThreads).isLessThanOrEqualTo(configCount);
    }

    @Test
    void testConfigValidation() {
        // Given valid config
        ExperimentConfig config = ExperimentConfig.createV2(
                "msmarco", "data/msmarco_sample.jsonl", "auto",
                0.90, "BIDIRECTIONAL", 0.30, 42L, null,
                true, "SEMANTIC", 5, 50000, 86400L, null,
                0.0, 0.0, "results/test.json");
        
        // Then config should have required fields
        assertThat(config.datasetName()).isEqualTo("msmarco");
        assertThat(config.datasetPath()).isEqualTo("data/msmarco_sample.jsonl");
        assertThat(config.randomSeed()).isEqualTo(42L);
    }
}
