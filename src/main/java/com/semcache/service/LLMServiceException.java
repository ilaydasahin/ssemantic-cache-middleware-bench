package com.semcache.service;

/**
 * Specific exception for LLM service failures.
 * 
 * Allows differentiation between:
 * - Rate limit errors (429)
 * - Authentication errors (401, 403)
 * - Service unavailable (503)
 * - Circuit breaker open
 * - Network timeouts
 */
public class LLMServiceException extends RuntimeException {
    
    private final ErrorType errorType;
    private final boolean retryable;
    
    public enum ErrorType {
        RATE_LIMIT(true),
        AUTHENTICATION(false),
        SERVICE_UNAVAILABLE(true),
        CIRCUIT_BREAKER_OPEN(false),
        NETWORK_TIMEOUT(true),
        INVALID_REQUEST(false),
        UNKNOWN(true);
        
        private final boolean retryable;
        
        ErrorType(boolean retryable) {
            this.retryable = retryable;
        }
        
        public boolean isRetryable() {
            return retryable;
        }
    }
    
    public LLMServiceException(String message) {
        this(message, ErrorType.UNKNOWN);
    }
    
    public LLMServiceException(String message, Throwable cause) {
        this(message, ErrorType.UNKNOWN, cause);
    }
    
    public LLMServiceException(String message, ErrorType errorType) {
        super(message);
        this.errorType = errorType;
        this.retryable = errorType.isRetryable();
    }
    
    public LLMServiceException(String message, ErrorType errorType, Throwable cause) {
        super(message, cause);
        this.errorType = errorType;
        this.retryable = errorType.isRetryable();
    }
    
    public ErrorType getErrorType() {
        return errorType;
    }
    
    public boolean isRetryable() {
        return retryable;
    }
    
    /**
     * Creates exception from HTTP status code.
     */
    public static LLMServiceException fromStatusCode(int statusCode, String message) {
        ErrorType type = switch (statusCode) {
            case 429 -> ErrorType.RATE_LIMIT;
            case 401, 403 -> ErrorType.AUTHENTICATION;
            case 503 -> ErrorType.SERVICE_UNAVAILABLE;
            case 400 -> ErrorType.INVALID_REQUEST;
            default -> ErrorType.UNKNOWN;
        };
        return new LLMServiceException(message, type);
    }
}
