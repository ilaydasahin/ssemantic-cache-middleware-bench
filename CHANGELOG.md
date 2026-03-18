# Changelog

All notable changes to the Semantic Cache Benchmark project.

## [2.1.0] - 2025-01-XX - Q1 Journal Hardening

### Architecture Improvements

#### Strategy Pattern Refactoring
- **Introduced `CacheLookupStrategy` interface** for pluggable lookup algorithms
- **Created strategy implementations**:
  - `SemanticStrategy`: L1 exact-match → HNSW → brute-force fallback
  - `HybridCascadeStrategy`: MiniLM (recall) → MPNet (precision)
  - `ExactMatchStrategy`: O(1) hash-based lookup
  - `MiddlewareBaselineStrategy`: 15ms overhead simulation
- **Eliminated dual-store inconsistency**: Removed `LocalVectorIndex` duplication
- **Single authoritative data store**: `cacheStore` + `queryIndex` only

### Performance Optimizations

#### Thread Safety & Concurrency
- **ONNX Session Pooling**: `ArrayBlockingQueue<OrtSession>` prevents inference contention
  - Pool size = CPU core count
  - Eliminates native memory leaks from shared session access
- **ConcurrentHashMap**: Replaced `HashMap` in `SimpleWordPieceTokenizer` for thread-safe vocab lookups
- **Primitive Arrays**: Replaced `ConcurrentLinkedQueue<Long>` with `long[]` in throughput benchmarks
  - Eliminates boxing/unboxing overhead
  - Reduces GC pressure

#### Memory Efficiency
- **Streaming SHA-256**: 8KB buffer prevents OOM on large datasets
  - Replaced `Files.readAllBytes()` with streaming digest
- **ONNX Tensor Leak Fix**: `token_type_ids` now properly closed in try-with-resources

### Error Handling & Observability

#### Production-Grade Logging
- **Redis Error Metrics**: Added `redis.store.errors` and `redis.search.errors` counters
- **Upgraded log levels**: `log.debug` → `log.warn` for Redis failures
- **Full stack traces**: Eviction failures now log complete context
- **Graceful degradation**: Redis unavailable → automatic brute-force fallback

#### Configuration Management
- **Externalized TTL**: Moved hardcoded `86400L` to `benchmark.ttl-seconds` property
- **Platform-specific dependencies**: MacOS netty resolver now profile-activated

### Code Quality

#### Cleanup
- **Removed deprecated code**: `BenchmarkRunner.runSingleBenchmark()` deleted
- **Simplified logging**: `ExperimentConfig.toLogSummary()` uses platform-agnostic `%n`
- **Lombok removal**: Manual getters/setters in `CacheProperties` for annotation processor compatibility

#### Java Version
- **Downgraded to Java 21 LTS**: From experimental Java 25
  - Full Spring Boot 3.3.5 compatibility
  - Production-ready LTS support

### Documentation

#### Comprehensive README
- Architecture diagrams
- Quick start guide
- Configuration reference
- Troubleshooting section
- Reproducibility guarantees

#### Code Comments
- Clarified `LocalVectorIndex` deprecation rationale
- Added javadoc for all strategy implementations
- Documented thread-safety guarantees

### Testing

#### Validation
- All existing tests pass with Java 21
- Compilation warnings eliminated
- No hardcoded credentials or debug statements

---

## [1.0.0] - Initial Release

### Core Features
- Semantic cache with ONNX embeddings (MiniLM, MPNet, TinyBERT)
- Redis 8 vectorset integration (VADD/VSIM)
- Background LFU eviction with fine-grained locking
- Comprehensive benchmark suite
- Deterministic experiment orchestration
- Multi-seed statistical validation

### Datasets
- MS MARCO (10K samples)
- Natural Questions (10K samples)
- Quora Question Pairs (10K samples)
- Paraphrase generation for semantic evaluation

### Metrics
- Hit rate, latency percentiles (p50, p95, p99)
- Cost savings estimation
- Memory usage tracking
- SBERT and ROUGE-L post-hoc evaluation
