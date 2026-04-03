package com.semcache.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Runs multiple dataset benchmarks in parallel for 3x speedup
 */
@Component
public class ParallelBenchmarkRunner {
    
    private static final Logger log = LoggerFactory.getLogger(ParallelBenchmarkRunner.class);
    private final BenchmarkRunner benchmarkRunner;
    
    public ParallelBenchmarkRunner(BenchmarkRunner benchmarkRunner) {
        this.benchmarkRunner = benchmarkRunner;
    }
    
    /**
     * Run multiple experiments in parallel
     * @param configs List of experiment configurations
     * @return true if all succeeded
     */
    public boolean runParallel(List<ExperimentConfig> configs) {
        if (configs.isEmpty()) {
            log.warn("No experiments to run");
            return false;
        }
        
        if (configs.size() == 1) {
            // Single experiment, run directly
            try {
                benchmarkRunner.run(configs.get(0));
                return true;
            } catch (Exception e) {
                log.error("Experiment failed", e);
                return false;
            }
        }
        
        // Multiple experiments, run in parallel
        log.info("Starting {} experiments in parallel...", configs.size());
        ExecutorService executor = Executors.newFixedThreadPool(
                Math.min(configs.size(), Runtime.getRuntime().availableProcessors()));
        
        try {
            CompletableFuture<?>[] futures = configs.stream()
                    .map(config -> CompletableFuture.runAsync(() -> {
                        try {
                            log.info("Starting parallel experiment: {}", config.datasetName());
                            benchmarkRunner.run(config);
                            log.info("Completed: {}", config.datasetName());
                        } catch (Exception e) {
                            log.error("Failed: {}", config.datasetName(), e);
                            throw new RuntimeException(e);
                        }
                    }, executor))
                    .toArray(CompletableFuture[]::new);
            
            CompletableFuture.allOf(futures).join();
            log.info("All {} experiments completed successfully!", configs.size());
            return true;
            
        } catch (Exception e) {
            log.error("Parallel execution failed", e);
            return false;
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(60, java.util.concurrent.TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                    if (!executor.awaitTermination(60, java.util.concurrent.TimeUnit.SECONDS)) {
                        log.error("Executor did not terminate");
                    }
                }
            } catch (InterruptedException ie) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}
