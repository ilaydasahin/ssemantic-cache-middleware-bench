# 65+ Kritik Hata Noktası - Deney Yarıda Kalma Riskleri

## 🚨 EN KRİTİK 10 RİSK (Acil Düzeltme Gerekli)

### 1. ❌ QUOTA EXHAUSTION (Gün 4)
**Risk:** 450K sorgu / 111,650 günlük = 4.03 gün → Gün 4'te tüm keyler tükenir
**Etki:** Deney durur, tamamlanamaz
**Çözüm:** 
- 77 key yeterli DEĞİL, 78+ key gerekli
- VEYA query sayısını 445K'ya düşür
- VEYA quota reset mekanizmasını doğrula

### 2. ❌ MEMORY LEAK (Gün 2-3)
**Risk:** 450K × 500 byte/log = 225MB + cache + metrics = 500MB+
**Etki:** OutOfMemoryError, process crash
**Çözüm:**
- Heap size 8GB'a çıkar (şu an default 512MB)
- Query logs'u periodic flush et
- Metrics'i periodic export et

### 3. ❌ DISK FULL (Gün 3-4)
**Risk:** 9,000 checkpoint × 2MB = 18GB + results = 20GB+
**Etki:** Checkpoint save fail, resume impossible
**Çözüm:**
- Checkpoint cleanup DURING experiment (not just startup)
- Disk space check BEFORE each checkpoint save
- Compress old checkpoints

### 4. ❌ REDIS CONNECTION POOL EXHAUSTION
**Risk:** Max 64 connections, 450K queries → pool exhausted
**Etki:** Timeout, fallback to brute-force, latency spike
**Çözüm:**
- Connection pool 256'ya çıkar
- Connection timeout 5s'ye düşür
- Circuit breaker ekle

### 5. ❌ THREAD POOL EXHAUSTION
**Risk:** Custom ForkJoinPool max 32 threads, leak riski
**Etki:** RejectedExecutionException, crash
**Çözüm:**
- Thread monitoring ekle
- Leak detection ekle
- Graceful degradation

### 6. ❌ ONNX SESSION TIMEOUT
**Risk:** Session pool timeout 30s, if all sessions stuck
**Etki:** Cascading failures, all queries timeout
**Çözüm:**
- Session timeout 10s'ye düşür
- Session health check ekle
- Fallback to simple embedding

### 7. ❌ CHECKPOINT CORRUPTION
**Risk:** Partial write if interrupted, JSON parse fail
**Etki:** Cannot resume, must restart from beginning
**Çözüm:**
- Atomic write (DONE)
- Backup previous checkpoint
- Validate checkpoint after write

### 8. ❌ QUERY RETRY SKIP
**Risk:** Max 100 retries, if fail → query skipped
**Etki:** 10% fail = 45K queries skipped, results invalid
**Çözüm:**
- Log skipped queries
- Fail experiment if >1% skipped
- Retry with different strategy

### 9. ❌ PARALLEL STREAM HANG
**Risk:** If one query hangs, entire benchmark hangs
**Etki:** Experiment never completes
**Çözüm:**
- Per-query timeout (5 minutes max)
- Watchdog thread
- Force-kill hung queries

### 10. ❌ NO HEALTH CHECK
**Risk:** No periodic verification system is healthy
**Etki:** Silent failures, experiment continues with bad data
**Çözüm:**
- Health check every 100 queries
- Check: memory, disk, threads, Redis, API keys
- Auto-pause if unhealthy

---

## 📊 Tüm 65+ Hata Noktası Kategorileri

### A. Resource Exhaustion (15 nokta)
1. ONNX session pool leak
2. Embedding vector accumulation
3. Query logs accumulation (225MB)
4. Metrics collector array (45MB)
5. Checkpoint file accumulation (18GB)
6. ForkJoinPool thread leak
7. Eviction scheduler backlog
8. Redis connection pool (64 max)
9. Disk space (20GB+ needed)
10. Result file size (225MB)
11. Query logs JSONL (225MB)
12. Heap size (default 512MB)
13. Redis max memory
14. Native memory (ONNX)
15. File descriptors

### B. Unhandled Exceptions (20 nokta)
16. Model file missing
17. Tokenization failure
18. Tensor creation failure
19. Mean pooling dimension mismatch
20. Null pointer in semantic strategy
21. Expired entry handling
22. Brute-force concurrent modification
23. No API keys
24. All keys exhausted
25. Rate limit not respected
26. Retry loop infinite
27. WebClient timeout
28. Redis connection lost
29. VADD failure
30. VSIM response parsing
31. Connection pool exhaustion
32. DNS resolution failure
33. Connection timeout
34. Network partition
35. Dataset file missing

### C. External Dependencies (10 nokta)
36. Malformed JSONL
37. Dataset too small
38. Quota exhaustion
39. Per-minute rate limit
40. Key rotation failure
41. Invalid similarity threshold
42. Invalid strategy
43. Invalid model name
44. TTL = 0
45. Heap size too small

### D. Race Conditions (10 nokta)
46. CacheStore concurrent modification
47. QueryIndex inconsistency
48. Checkpoint race condition
49. Metrics collector concurrent modification
50. Write lock contention
51. Eviction lock contention
52. Eviction scheduler deadlock
53. Session pool timeout
54. Synchronous LLM call blocking
55. Parallel stream blocking

### E. Data Corruption (5 nokta)
56. Checkpoint partial write
57. Atomic rename failure
58. Result file partial JSON
59. Concurrent writes
60. Redis/local mismatch

### F. Process Termination (5 nokta)
61. OutOfMemoryError
62. StackOverflowError
63. SIGTERM
64. SIGKILL
65. JVM crash (segfault)

---

## 🔧 ACİL DÜZELTMELER (Öncelik Sırası)

### Öncelik 1: CRITICAL (Deney kesinlikle başarısız olur)
1. ✅ Heap size 8GB'a çıkar
2. ✅ Redis connection pool 256'ya çıkar
3. ✅ Checkpoint cleanup during experiment
4. ✅ Health check loop ekle
5. ✅ Per-query timeout ekle

### Öncelik 2: HIGH (Deney muhtemelen başarısız olur)
6. ✅ Query logs periodic flush
7. ✅ Metrics periodic export
8. ✅ Disk space check before checkpoint
9. ✅ Session timeout 10s'ye düşür
10. ✅ Skipped query tracking

### Öncelik 3: MEDIUM (Deney yavaşlar ama tamamlanabilir)
11. ✅ Circuit breaker for Redis
12. ✅ Backpressure mechanism
13. ✅ Graceful degradation
14. ✅ Partial result export
15. ✅ Resource monitoring

### Öncelik 4: LOW (Performans optimizasyonu)
16. Connection timeout optimization
17. Thread pool tuning
18. Lock contention reduction
19. Eviction optimization
20. Logging optimization

---

## 📋 DÜZELTME PLANI

### Adım 1: JVM & Resource Configuration
```bash
# MAVEN_OPTS ekle
export MAVEN_OPTS="-Xmx8g -Xms4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# Disk space check
df -h | grep "/$" | awk '{if ($4 < 25) exit 1}'
```

### Adım 2: Application Configuration
```yaml
# application.yml
spring:
  data:
    redis:
      lettuce:
        pool:
          max-active: 256  # 64 → 256
          max-wait: 5000ms  # 1000ms → 5000ms
```

### Adım 3: Code Changes
- [ ] CheckpointManager: Periodic cleanup during experiment
- [ ] BenchmarkRunner: Health check loop every 100 queries
- [ ] BenchmarkRunner: Per-query timeout (5 minutes)
- [ ] MetricsCollector: Periodic export to disk
- [ ] ExperimentResultExporter: Disk space check before write
- [ ] OnnxEmbeddingService: Session timeout 10s
- [ ] RedisSearchService: Circuit breaker pattern
- [ ] GeminiService: Backpressure mechanism

### Adım 4: Monitoring & Alerts
- [ ] Memory usage monitoring
- [ ] Disk space monitoring
- [ ] Thread count monitoring
- [ ] Redis connection monitoring
- [ ] API key quota monitoring
- [ ] Query success rate monitoring

### Adım 5: Testing
- [ ] 1K query test (validate all fixes)
- [ ] 10K query test (stress test)
- [ ] 50K query test (endurance test)
- [ ] Full 450K query test

---

## 🎯 BAŞARI KRİTERLERİ

Deney başarılı sayılır eğer:
1. ✅ 450,000 sorgunun TAMAMINI işler
2. ✅ Hiçbir sorgu skip edilmez (<%0.1 acceptable)
3. ✅ Hiçbir crash olmaz
4. ✅ Tüm checkpoint'ler valid
5. ✅ Tüm result dosyaları complete
6. ✅ 4-5 gün içinde tamamlanır
7. ✅ Memory leak yok
8. ✅ Disk full olmaz
9. ✅ Thread leak yok
10. ✅ Redis connection leak yok

---

**SONUÇ:** 65+ potansiyel hata noktası tespit edildi. En kritik 10 tanesi acil düzeltme gerektiriyor.

