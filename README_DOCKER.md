# Docker Deployment Guide

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build all services
docker-compose build

# Start all services (Redis, Ollama, Semantic Cache, Prometheus, Grafana)
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f semcache
```

### 2. Setup Secrets

```bash
# Copy example secrets file
cp secrets/application-secrets.yml.example secrets/application-secrets.yml

# Edit with your API keys (or use Ollama for no keys)
nano secrets/application-secrets.yml
```

### 3. Download Ollama Model

```bash
# Enter Ollama container
docker-compose exec ollama bash

# Download model
ollama pull llama3.2

# Exit container
exit
```

### 4. Run Benchmark

```bash
# Run quick test
docker-compose exec semcache bash -c "mvn spring-boot:run -Dspring-boot.run.profiles=benchmark"

# Or use pre-built JAR
docker-compose exec semcache java -jar app.jar
```

## Service URLs

- **Semantic Cache API**: http://localhost:8080
- **Prometheus Metrics**: http://localhost:9090/actuator/prometheus
- **Prometheus UI**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
- **Redis**: localhost:6379
- **RedisInsight**: http://localhost:8001
- **Ollama API**: http://localhost:11434

## Resource Limits

Default limits (adjust in docker-compose.yml):
- **CPU**: 4 cores (2 reserved)
- **Memory**: 12 GB (8 GB reserved)
- **Disk**: Unlimited (monitor with `df -h`)

## Reproducibility

### Hardware Profiling

```bash
# Profile hardware inside container
docker-compose exec semcache python3 scripts/hardware_profiler.py --output /app/results/hardware_specs.json

# Copy to host
docker cp semcache:/app/results/hardware_specs.json ./results/
```

### Dependency Locking

```bash
# Generate dependency tree
docker-compose exec semcache mvn dependency:tree > dependency-tree.txt

# Generate effective POM (with resolved versions)
docker-compose exec semcache mvn help:effective-pom > effective-pom.xml
```

### Image Digest Pinning

```bash
# Get image digests for reproducibility
docker images --digests | grep semcache

# Pin in docker-compose.yml:
# image: redis/redis-stack@sha256:abc123...
```

## Troubleshooting

### Out of Memory

```bash
# Increase heap size
docker-compose exec semcache bash
export JAVA_OPTS="-Xmx12g -Xms8g"
java $JAVA_OPTS -jar app.jar
```

### Redis Connection Failed

```bash
# Check Redis health
docker-compose exec redis redis-cli ping

# Restart Redis
docker-compose restart redis
```

### Ollama Model Not Found

```bash
# List available models
docker-compose exec ollama ollama list

# Download missing model
docker-compose exec ollama ollama pull llama3.2
```

## Production Deployment

### Security Hardening

1. **Use secrets manager** (not files):
   ```yaml
   secrets:
     gemini_api_key:
       external: true
   ```

2. **Enable TLS**:
   ```yaml
   environment:
     - SERVER_SSL_ENABLED=true
     - SERVER_SSL_KEY_STORE=/app/certs/keystore.p12
   ```

3. **Run as non-root** (already configured):
   ```dockerfile
   USER semcache
   ```

### Monitoring

```bash
# View Prometheus metrics
curl http://localhost:9090/actuator/prometheus

# Import Grafana dashboard
# Go to http://localhost:3000
# Import dashboard from monitoring/grafana-dashboards/
```

### Backup and Restore

```bash
# Backup Redis data
docker-compose exec redis redis-cli SAVE
docker cp semcache-redis:/data/dump.rdb ./backups/

# Backup results
docker cp semcache:/app/results ./backups/results-$(date +%Y%m%d)

# Restore
docker cp ./backups/dump.rdb semcache-redis:/data/
docker-compose restart redis
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build and Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker-compose build
      
      - name: Run tests
        run: docker-compose run semcache mvn test
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./target/site/jacoco/jacoco.xml
```

## Performance Tuning

### JVM Options

```bash
# G1GC with optimized pause times
JAVA_OPTS="-Xmx8g -Xms4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+ParallelRefProcEnabled"

# ZGC for ultra-low latency
JAVA_OPTS="-Xmx8g -Xms4g -XX:+UseZGC -XX:ZCollectionInterval=5"
```

### Redis Tuning

```bash
# Increase max memory
docker-compose exec redis redis-cli CONFIG SET maxmemory 4gb
docker-compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Full cleanup
docker system prune -a --volumes
```

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Verify health: `docker-compose ps`
3. Open GitHub issue with logs attached
