# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-07

### Added - Q1 Publication Readiness

#### Security
- **CRITICAL**: Removed 77 exposed API keys from version control
- Added secrets management with `secrets/` directory (gitignored)
- Created `application-secrets.yml.example` template
- Added security scanning in CI/CD pipeline (Trivy)
- Implemented Docker secrets mounting

#### Reproducibility
- Docker container with locked dependencies
- Hardware profiler script (`hardware_profiler.py`)
- Dependency version locking in `pom.xml`
- Dataset checksum verification
- Git commit tracking in experiment metadata
- `REPRODUCIBILITY.md` with complete instructions
- Result comparison tool (`compare_results.py`)

#### Testing
- Increased test coverage target to 80%
- Added `HybridCascadeStrategyTest` (unit tests)
- Added `EndToEndBenchmarkTest` (integration tests)
- JaCoCo code coverage reporting
- CI/CD pipeline with automated testing

#### Documentation
- `ETHICS.md` with dataset licenses and carbon footprint
- `README_DOCKER.md` for Docker deployment
- `CHANGELOG.md` (this file)
- Enhanced Javadoc comments
- API documentation

#### Infrastructure
- Docker Compose with Redis, Ollama, Prometheus, Grafana
- GitHub Actions CI/CD pipeline
- Prometheus metrics collection
- Grafana dashboards (TBD)

#### Statistical Analysis
- FDR correction for multiple comparisons (Benjamini-Hochberg)
- Effect size confidence intervals
- Power analysis validation
- Bias analysis (query length, dataset, temporal)
- Reproducibility score calculation

### Changed

#### Dependencies
- Locked ONNX Runtime to 1.24.3
- Locked Jedis to 5.2.0
- Locked Jackson to 2.18.2
- Locked Micrometer to 1.14.2
- Added Maven Enforcer plugin for version validation

#### Configuration
- Moved API keys to external secrets file
- Added environment variable support
- Improved YAML configuration structure

#### Code Quality
- Refactored error handling (specific exceptions)
- Improved null safety (Optional usage)
- Enhanced logging (structured format)
- Fixed memory leaks (streaming writes)

### Fixed

#### Critical Bugs
- Cost savings calculation (E5 fix - hitRate already percentage)
- Null pointer in `processSingleQuery` (Ö-4 fix)
- Race condition in cache eviction
- Memory leak in query log collection

#### Statistical Issues
- FDR correction implementation
- Effect size calculation (Cohen's d)
- Confidence interval computation
- Multiple testing correction

### Security

#### Vulnerabilities Fixed
- **CVE-2024-XXXX**: API keys in version control (CRITICAL)
- Dependency vulnerabilities (via `mvn dependency:check`)
- Docker image vulnerabilities (via Trivy scan)

### Deprecated
- `LocalVectorIndex` (removed - redundant with `cacheStore`)
- Direct API key configuration (use secrets file)

### Removed
- Hardcoded API keys from `application-local.yml`
- Duplicate data structures
- Unused dependencies

## [0.9.0] - 2026-03-15

### Added
- Initial implementation of semantic cache
- ONNX embedding service
- Redis HNSW integration
- Benchmark runner
- Dataset preparation scripts
- Statistical analysis scripts

### Known Issues
- Test coverage below 50%
- API keys exposed in config files
- No Docker support
- Limited documentation

## [0.1.0] - 2026-02-01

### Added
- Project scaffolding
- Basic cache implementation
- Proof of concept

---

## Upgrade Guide

### From 0.9.0 to 1.0.0

#### 1. Update API Key Configuration

**Old** (application-local.yml):
```yaml
llm:
  api-keys: "AIzaSy..."
```

**New** (secrets/application-secrets.yml):
```yaml
llm:
  api-keys: "${GEMINI_API_KEYS:your-key-here}"
```

#### 2. Update Docker Deployment

```bash
# Old
java -jar target/semantic-cache-benchmark-1.0.0.jar

# New
docker-compose up -d
```

#### 3. Update Test Execution

```bash
# Old
mvn test

# New (with coverage)
mvn test jacoco:report
```

#### 4. Update Result Analysis

```bash
# Old
python3 scripts/analyze_results.py results/

# New (with reproducibility check)
python3 scripts/analyze_results.py results/
python3 scripts/compare_results.py --your-results results/ --published-results published/
```

## Migration Checklist

- [ ] Move API keys to `secrets/application-secrets.yml`
- [ ] Update `.gitignore` to exclude secrets
- [ ] Rebuild Docker images
- [ ] Run hardware profiler
- [ ] Verify test coverage ≥80%
- [ ] Run reproducibility verification
- [ ] Update documentation

## Support

For migration help:
- GitHub Issues: [repository-url]/issues
- Email: [your-email@institution.edu]
