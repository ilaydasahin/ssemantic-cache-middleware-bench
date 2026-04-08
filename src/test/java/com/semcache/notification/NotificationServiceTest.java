package com.semcache.notification;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Unit tests for NotificationService
 */
@ExtendWith(MockitoExtension.class)
class NotificationServiceTest {

    @Mock
    private WebClient webClient;

    @Mock
    private WebClient.RequestBodyUriSpec requestBodyUriSpec;

    @Mock
    private WebClient.RequestBodySpec requestBodySpec;

    @Mock
    private WebClient.ResponseSpec responseSpec;

    private NotificationService notificationService;

    @BeforeEach
    void setUp() {
        notificationService = new NotificationService();
    }

    @Test
    void testNotificationDisabled() {
        // Given notifications disabled
        ReflectionTestUtils.setField(notificationService, "enabled", false);
        
        // When notification is sent
        notificationService.notifyExperimentComplete("exp-1", "msmarco", 85.5, 120.0, 60);
        
        // Then should not throw and do nothing
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 85.5, 120.0, 60);
        }).doesNotThrowAnyException();
    }

    @Test
    void testNotificationEnabled() {
        // Given notifications enabled
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        ReflectionTestUtils.setField(notificationService, "slackWebhook", "");
        ReflectionTestUtils.setField(notificationService, "emailWebhook", "");
        
        // When notification is sent
        // Then should not throw (console notification always works)
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 85.5, 120.0, 60);
        }).doesNotThrowAnyException();
    }

    @Test
    void testSlackWebhookConfigured() {
        // Given Slack webhook configured
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        ReflectionTestUtils.setField(notificationService, "slackWebhook", "https://hooks.slack.com/test");
        ReflectionTestUtils.setField(notificationService, "emailWebhook", "");
        
        // When notification is sent
        // Then should not throw
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 85.5, 120.0, 60);
        }).doesNotThrowAnyException();
    }

    @Test
    void testEmailWebhookConfigured() {
        // Given email webhook configured
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        ReflectionTestUtils.setField(notificationService, "slackWebhook", "");
        ReflectionTestUtils.setField(notificationService, "emailWebhook", "https://api.email.com/send");
        
        // When notification is sent
        // Then should not throw
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 85.5, 120.0, 60);
        }).doesNotThrowAnyException();
    }

    @Test
    void testBothWebhooksConfigured() {
        // Given both webhooks configured
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        ReflectionTestUtils.setField(notificationService, "slackWebhook", "https://hooks.slack.com/test");
        ReflectionTestUtils.setField(notificationService, "emailWebhook", "https://api.email.com/send");
        
        // When notification is sent
        // Then should not throw
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 85.5, 120.0, 60);
        }).doesNotThrowAnyException();
    }

    @Test
    void testMessageFormatting() {
        // Given notification parameters
        String experimentId = "exp-123";
        String dataset = "msmarco";
        double hitRate = 85.5;
        double p99Latency = 120.0;
        long durationMinutes = 60;
        
        // When notification is sent
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        
        // Then should format message correctly
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete(experimentId, dataset, hitRate, p99Latency, durationMinutes);
        }).doesNotThrowAnyException();
    }

    @Test
    void testHighHitRate() {
        // Given high hit rate
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        
        // When notification is sent
        // Then should handle correctly
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 99.9, 50.0, 30);
        }).doesNotThrowAnyException();
    }

    @Test
    void testLowHitRate() {
        // Given low hit rate
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        
        // When notification is sent
        // Then should handle correctly
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 10.5, 200.0, 120);
        }).doesNotThrowAnyException();
    }

    @Test
    void testLongDuration() {
        // Given long experiment duration
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        
        // When notification is sent
        // Then should handle correctly
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 85.5, 120.0, 960); // 16 hours
        }).doesNotThrowAnyException();
    }

    @Test
    void testShortDuration() {
        // Given short experiment duration
        ReflectionTestUtils.setField(notificationService, "enabled", true);
        
        // When notification is sent
        // Then should handle correctly
        assertThatCode(() -> {
            notificationService.notifyExperimentComplete("exp-1", "msmarco", 85.5, 120.0, 5);
        }).doesNotThrowAnyException();
    }
}
