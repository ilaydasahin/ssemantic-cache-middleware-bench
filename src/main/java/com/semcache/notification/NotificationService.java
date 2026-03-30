package com.semcache.notification;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

/**
 * Sends notifications when experiments complete
 * Supports: Slack, Email (via webhook), Console
 */
@Service
public class NotificationService {
    
    private static final Logger log = LoggerFactory.getLogger(NotificationService.class);
    
    @Value("${notification.slack.webhook:}")
    private String slackWebhook;
    
    @Value("${notification.email.webhook:}")
    private String emailWebhook;
    
    @Value("${notification.enabled:false}")
    private boolean enabled;
    
    private final WebClient webClient;
    
    public NotificationService() {
        // Configure WebClient with timeouts for notification webhooks
        io.netty.channel.ChannelOption<Integer> channelOption = io.netty.channel.ChannelOption.CONNECT_TIMEOUT_MILLIS;
        reactor.netty.http.client.HttpClient httpClient = reactor.netty.http.client.HttpClient.create()
                .option(channelOption, 10000) // 10s connection timeout
                .responseTimeout(java.time.Duration.ofSeconds(30)); // 30s response timeout
        
        this.webClient = WebClient.builder()
                .clientConnector(new org.springframework.http.client.reactive.ReactorClientHttpConnector(httpClient))
                .build();
    }
    
    public void notifyExperimentComplete(String experimentId, String dataset, 
                                        double hitRate, double p99Latency, 
                                        long durationMinutes) {
        if (!enabled) {
            log.debug("Notifications disabled");
            return;
        }
        
        String message = formatMessage(experimentId, dataset, hitRate, p99Latency, durationMinutes);
        
        // Console notification (always)
        log.info("📧 {}", message);
        
        // Slack notification
        if (slackWebhook != null && !slackWebhook.isEmpty()) {
            sendSlackNotification(message);
        }
        
        // Email notification
        if (emailWebhook != null && !emailWebhook.isEmpty()) {
            sendEmailNotification(message);
        }
    }
    
    private String formatMessage(String experimentId, String dataset, 
                                 double hitRate, double p99Latency, 
                                 long durationMinutes) {
        return String.format(
                "✅ Experiment Complete!\n" +
                "ID: %s\n" +
                "Dataset: %s\n" +
                "Hit Rate: %.1f%%\n" +
                "P99 Latency: %.0fms\n" +
                "Duration: %dmin",
                experimentId, dataset, hitRate, p99Latency, durationMinutes
        );
    }
    
    private void sendSlackNotification(String message) {
        try {
            Map<String, String> payload = Map.of("text", message);
            webClient.post()
                    .uri(slackWebhook)
                    .bodyValue(payload)
                    .retrieve()
                    .bodyToMono(String.class)
                    .subscribe(
                            response -> log.info("Slack notification sent"),
                            error -> log.error("Failed to send Slack notification", error)
                    );
        } catch (Exception e) {
            log.error("Slack notification error", e);
        }
    }
    
    private void sendEmailNotification(String message) {
        try {
            Map<String, String> payload = Map.of(
                    "subject", "Experiment Complete",
                    "body", message
            );
            webClient.post()
                    .uri(emailWebhook)
                    .bodyValue(payload)
                    .retrieve()
                    .bodyToMono(String.class)
                    .subscribe(
                            response -> log.info("Email notification sent"),
                            error -> log.error("Failed to send email notification", error)
                    );
        } catch (Exception e) {
            log.error("Email notification error", e);
        }
    }
}
