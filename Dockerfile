# Multi-stage build for a lightweight production image
FROM maven:3.9.6-eclipse-temurin-21-alpine AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn clean package -DskipTests

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app

# Non-root user for security
RUN addgroup -S semcache && adduser -S semcache -G semcache
USER semcache

COPY --from=build /app/target/*.jar app.jar
COPY models/ ./models/
COPY data/ ./data/

# Default environment variables
ENV SPRING_PROFILES_ACTIVE=prod
ENV GEMINI_API_KEY=""

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
