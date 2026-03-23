# Multi-stage build for semantic cache benchmark
FROM maven:3.9-eclipse-temurin-21-alpine AS build

WORKDIR /app

# Copy pom.xml and download dependencies (cached layer)
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source and build
COPY src ./src
COPY models ./models
COPY data ./data
RUN mvn clean package -DskipTests -q

# Runtime stage
FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

# Copy built jar and resources
COPY --from=build /app/target/*.jar app.jar
COPY --from=build /app/models ./models
COPY --from=build /app/data ./data

# Create directories for results and logs
RUN mkdir -p results logs checkpoints

# Environment variables (override with -e)
ENV GEMINI_API_KEYS=""
ENV SPRING_PROFILES_ACTIVE=benchmark
ENV BENCHMARK_SAMPLE_SIZE=1000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD ps aux | grep java || exit 1

# Run benchmark
ENTRYPOINT ["java", "-Xmx2g", "-jar", "app.jar"]

# Usage:
# docker build -t semantic-cache-benchmark .
# docker run -e GEMINI_API_KEYS="key1,key2,..." semantic-cache-benchmark
