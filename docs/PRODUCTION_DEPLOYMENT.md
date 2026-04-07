# Production Deployment Guide

Q1 dergileri için production-ready deployment kanıtı.

## Docker Deployment

### 1. Build Docker Image

```bash
# Build image
docker build -t semantic-cache:latest .

# Test image
docker run --rm semantic-cache:latest --version
```

### 2. Docker Compose Full Stack

```bash
# Start full stack (Redis + App)
docker-compose up -d

# Check logs
docker-compose logs -f app

# Stop
docker-compose down
```

### 3. Production Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: always
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  app:
    image: semantic-cache:latest
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=production
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - LLM_PROVIDER=ollama
      - LLM_OLLAMA_URL=http://host.docker.internal:11434
    depends_on:
      redis:
        condition: service_healthy
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/actuator/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: always

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: always

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace semantic-cache
```

### 2. Deploy Redis

```yaml
# k8s/redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: semantic-cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis/redis-stack:latest
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: semantic-cache
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

### 3. Deploy Application

```yaml
# k8s/app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-cache
  namespace: semantic-cache
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-cache
  template:
    metadata:
      labels:
        app: semantic-cache
    spec:
      containers:
      - name: app
        image: semantic-cache:latest
        ports:
        - containerPort: 8080
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: "production"
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /actuator/health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /actuator/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: semantic-cache
  namespace: semantic-cache
spec:
  type: LoadBalancer
  selector:
    app: semantic-cache
  ports:
  - port: 80
    targetPort: 8080
```

### 4. Deploy

```bash
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/app-deployment.yaml

# Check status
kubectl get pods -n semantic-cache
kubectl get svc -n semantic-cache
```

## Load Testing

### 1. Install K6

```bash
# macOS
brew install k6

# Linux
sudo apt-get install k6
```

### 2. Load Test Script

```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 500 },  // Ramp up to 500 users
    { duration: '5m', target: 500 },  // Stay at 500 users
    { duration: '2m', target: 1000 }, // Ramp up to 1000 users
    { duration: '5m', target: 1000 }, // Stay at 1000 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'], // 95% < 500ms, 99% < 1s
    http_req_failed: ['rate<0.01'],                  // Error rate < 1%
  },
};

const queries = [
  "What is machine learning?",
  "How does semantic caching work?",
  "Explain neural networks",
  "What is the difference between AI and ML?",
  "How to optimize database queries?",
];

export default function () {
  const query = queries[Math.floor(Math.random() * queries.length)];
  
  const payload = JSON.stringify({
    query: query,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post('http://localhost:8080/api/query', payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

### 3. Run Load Test

```bash
# Run load test
k6 run load-test.js

# Run with custom VUs
k6 run --vus 1000 --duration 10m load-test.js

# Run with output to InfluxDB
k6 run --out influxdb=http://localhost:8086/k6 load-test.js
```

## Monitoring

### 1. Prometheus Metrics

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'semantic-cache'
    metrics_path: '/actuator/prometheus'
    static_configs:
      - targets: ['app:8080']
```

### 2. Grafana Dashboard

Import dashboard from `monitoring/grafana-dashboard.json`

Key metrics:
- Cache hit rate
- P50/P95/P99 latency
- Throughput (RPS)
- Memory usage
- Redis operations

### 3. Alerts

```yaml
# monitoring/alerts.yml
groups:
  - name: semantic-cache
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P99 latency detected"
          description: "P99 latency is {{ $value }}s"

      - alert: LowHitRate
        expr: cache_hit_rate < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Hit rate is {{ $value }}"
```

## Performance Benchmarks

### Expected Performance (Production)

| Metric | Value |
|--------|-------|
| Throughput | 10,000+ RPS |
| P50 Latency | < 10ms |
| P95 Latency | < 50ms |
| P99 Latency | < 100ms |
| Hit Rate | 85-90% |
| Memory Usage | 4-8 GB |
| CPU Usage | 50-70% |

### Scaling Guidelines

- **< 1,000 RPS**: Single instance (4 GB RAM, 2 CPU)
- **1,000 - 10,000 RPS**: 3 instances (8 GB RAM, 4 CPU each)
- **10,000+ RPS**: 5+ instances + Redis cluster

## Security

### 1. API Authentication

```yaml
# application-production.yml
security:
  api-key:
    enabled: true
    header-name: X-API-Key
```

### 2. Rate Limiting

```yaml
rate-limit:
  enabled: true
  requests-per-second: 100
  burst: 200
```

### 3. TLS/SSL

```yaml
server:
  ssl:
    enabled: true
    key-store: classpath:keystore.p12
    key-store-password: ${SSL_KEYSTORE_PASSWORD}
    key-store-type: PKCS12
```

## Troubleshooting

### High Memory Usage

```bash
# Check heap usage
jmap -heap <pid>

# Dump heap
jmap -dump:format=b,file=heap.bin <pid>

# Analyze with VisualVM
visualvm heap.bin
```

### Redis Connection Issues

```bash
# Check Redis connectivity
redis-cli -h redis -p 6379 ping

# Check Redis memory
redis-cli INFO memory

# Clear cache
redis-cli FLUSHALL
```

### Performance Degradation

```bash
# Check thread dumps
jstack <pid> > thread-dump.txt

# Check GC logs
java -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -jar app.jar

# Profile with async-profiler
./profiler.sh -d 60 -f flamegraph.html <pid>
```

## Conclusion

This deployment guide provides production-ready configurations for:
- ✅ Docker containerization
- ✅ Kubernetes orchestration
- ✅ Load testing (K6)
- ✅ Monitoring (Prometheus + Grafana)
- ✅ Security best practices

For Q1 publication, include:
- Load test results (1000+ RPS)
- Latency percentiles under load
- Resource utilization metrics
- Scalability analysis
