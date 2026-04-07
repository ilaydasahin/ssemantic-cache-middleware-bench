# Multi-stage build for semantic cache benchmark
# Optimized for reproducibility and minimal image size

# Stage 1: Build stage
FROM maven:3.9-eclipse-temurin-21-alpine AS builder

WORKDIR /build

# Copy dependency files first (better layer caching)
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source code
COPY src ./src
COPY models ./models

# Build application (skip tests in Docker build, run separately)
RUN mvn clean package -DskipTests -B

# Stage 2: Runtime stage
FROM eclipse-temurin:21-jre-alpine

# Install Python for analysis scripts
RUN apk add --no-cache python3 py3-pip bash curl

# Create non-root user for security
RUN addgroup -g 1000 semcache && \
    adduser -D -u 1000 -G semcache semcache

WORKDIR /app

# Copy built JAR from builder stage
COPY --from=builder /build/target/*.jar app.jar

# Copy models and scripts
COPY --chown=semcache:semcache models ./models
COPY --chown=semcache:semcache scripts ./scripts
COPY --chown=semcache:semcache data ./data

# Install Python dependencies
COPY scripts/requirements.txt ./scripts/
RUN pip3 install --no-cache-dir -r scripts/requirements.txt

# Create results directory
RUN mkdir -p /app/results /app/logs && \
    chown -R semcache:semcache /app

# Switch to non-root user
USER semcache

# Environment variables (override with docker run -e)
ENV JAVA_OPTS="-Xmx8g -Xms4g" \
    SPRING_PROFILES_ACTIVE="benchmark" \
    BENCHMARK_CURRENT_DATASET="msmarco" \
    BENCHMARK_CURRENT_SEED="42"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/actuator/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
