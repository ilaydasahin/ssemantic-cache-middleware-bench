package com.semcache.config;

import io.lettuce.core.resource.ClientResources;
import io.lettuce.core.metrics.MicrometerCommandLatencyRecorder;
import io.lettuce.core.metrics.MicrometerOptions;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Infrastructure Monitoring: Enables Lettuce Command Latency recording via Micrometer.
 * M.6 Gold Standard: Proves that P99 spikes are not caused by Redis connection pool delays.
 */
@Configuration
public class LettuceMetricsConfig {

    @Bean(destroyMethod = "shutdown")
    public ClientResources clientResources(MeterRegistry meterRegistry) {
        MicrometerOptions options = MicrometerOptions.create();

        return ClientResources.builder()
                .commandLatencyRecorder(new MicrometerCommandLatencyRecorder(meterRegistry, options))
                .build();
    }
}
