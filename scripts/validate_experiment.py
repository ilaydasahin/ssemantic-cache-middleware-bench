"""
Pre-flight Validation Script for Semantic Cache Benchmark.

Checks all prerequisites before running experiments to prevent wasted compute time.

Usage: python3 validate_experiment.py

Exit codes:
  0 - All checks passed
  1 - Critical failure (cannot proceed)
  2 - Warning (can proceed with caution)
"""

import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path


class ValidationError(Exception):
    """Critical validation failure."""
    pass


class ValidationWarning(Exception):
    """Non-critical validation issue."""
    pass


def check_java_version():
    """Verify Java 17+ is installed."""
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        version_line = result.stderr.split('\n')[0]
        
        # Extract version number (handles both old and new format)
        import re
        version_match = re.search(r'version "?(\d+)', version_line)
        if version_match:
            major_version = int(version_match.group(1))
            if major_version < 17:
                raise ValidationError(f"Java 17+ required, found version {major_version}")
        else:
            raise ValidationError(f"Cannot parse Java version: {version_line}")
        
        print(f"✅ Java version: {version_line}")
    except FileNotFoundError:
        raise ValidationError("Java not found in PATH")


def check_maven():
    """Verify Maven is installed."""
    try:
        result = subprocess.run(['mvn', '-version'], capture_output=True, text=True)
        version_line = result.stdout.split('\n')[0]
        print(f"✅ Maven: {version_line}")
    except FileNotFoundError:
        raise ValidationError("Maven not found in PATH")


def check_python_dependencies():
    """Verify all Python dependencies are installed."""
    required = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
        'sentence_transformers', 'torch', 'rouge_score', 'datasets', 'statsmodels'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        raise ValidationError(f"Missing Python packages: {', '.join(missing)}\n"
                            f"Run: pip install -r scripts/requirements.txt")
    
    print(f"✅ Python dependencies: All {len(required)} packages installed")


def check_datasets():
    """Verify datasets are prepared and valid."""
    data_dir = Path('data')
    required_files = [
        'msmarco_sample_with_paraphrases.jsonl',
        'nq_sample_with_paraphrases.jsonl',
        'qqp_sample_with_paraphrases.jsonl'
    ]
    
    missing = []
    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        raise ValidationError(f"Missing datasets: {', '.join(missing)}\n"
                            f"Run: cd scripts && python prepare_datasets.py")
    
    # Validate dataset integrity
    for filename in required_files:
        filepath = data_dir / filename
        line_count = sum(1 for _ in open(filepath))
        
        if line_count < 100:
            raise ValidationWarning(f"{filename} has only {line_count} lines (expected >1000)")
        
        # Check first line is valid JSON
        with open(filepath) as f:
            first_line = f.readline()
            try:
                record = json.loads(first_line)
                if 'query' not in record or 'answer' not in record:
                    raise ValidationError(f"{filename} missing required fields")
            except json.JSONDecodeError:
                raise ValidationError(f"{filename} contains invalid JSON")
    
    print(f"✅ Datasets: All {len(required_files)} files validated")


def check_embedding_models():
    """Verify ONNX embedding models are present."""
    models_dir = Path('models')
    required_models = [
        'all-MiniLM-L6-v2/model.onnx',
        'all-mpnet-base-v2/model.onnx',
        'paraphrase-TinyBERT-L6-v2/model.onnx'
    ]
    
    missing = []
    for model_path in required_models:
        full_path = models_dir / model_path
        if not full_path.exists():
            missing.append(model_path)
    
    if missing:
        raise ValidationError(f"Missing ONNX models: {', '.join(missing)}\n"
                            f"Run: bash scripts/fetch_embedding_assets.sh")
    
    print(f"✅ Embedding models: All {len(required_models)} ONNX files present")


def check_disk_space():
    """Verify sufficient disk space for results."""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    stat = os.statvfs(results_dir)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    
    if free_gb < 1:
        raise ValidationError(f"Insufficient disk space: {free_gb:.1f} GB free (need 1+ GB)")
    elif free_gb < 5:
        raise ValidationWarning(f"Low disk space: {free_gb:.1f} GB free (recommend 5+ GB)")
    
    print(f"✅ Disk space: {free_gb:.1f} GB available")


def check_memory():
    """Verify sufficient RAM."""
    try:
        if sys.platform == 'darwin':
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            mem_bytes = int(result.stdout.split(':')[1].strip())
            mem_gb = mem_bytes / (1024**3)
        elif sys.platform.startswith('linux'):
            with open('/proc/meminfo') as f:
                mem_kb = int(f.readline().split()[1])
                mem_gb = mem_kb / (1024**2)
        else:
            raise ValidationWarning("Cannot detect RAM on this platform")
        
        if mem_gb < 8:
            raise ValidationError(f"Insufficient RAM: {mem_gb:.1f} GB (need 8+ GB)")
        elif mem_gb < 16:
            raise ValidationWarning(f"Limited RAM: {mem_gb:.1f} GB (recommend 16+ GB)")
        
        print(f"✅ Memory: {mem_gb:.1f} GB available")
    except Exception as e:
        raise ValidationWarning(f"Cannot verify RAM: {e}")


def check_ollama():
    """Verify Ollama is running (optional)."""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama: Running with {len(models)} models")
        else:
            raise ValidationWarning("Ollama API returned non-200 status")
    except Exception:
        print("⚠️  Ollama: Not running (optional for mock experiments)")


def check_redis():
    """Verify Redis is running (optional)."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2)
        r.ping()
        info = r.info()
        version = info.get('redis_version', 'unknown')
        print(f"✅ Redis: Running version {version}")
    except Exception:
        print("⚠️  Redis: Not running (HNSW experiments will be skipped)")


def check_git_status():
    """Verify repository is clean (for reproducibility)."""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            raise ValidationWarning("Git repository has uncommitted changes\n"
                                  "Commit changes for full reproducibility")
        
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
        print(f"✅ Git: Clean repository at commit {commit}")
    except Exception:
        print("⚠️  Git: Not a git repository (reproducibility may be limited)")


def main():
    print("=== Semantic Cache Benchmark - Pre-flight Validation ===\n")
    
    warnings = []
    
    try:
        # Critical checks (must pass)
        check_java_version()
        check_maven()
        check_python_dependencies()
        check_datasets()
        check_embedding_models()
        check_disk_space()
        check_memory()
        
        # Optional checks (warnings only)
        try:
            check_ollama()
        except ValidationWarning as w:
            warnings.append(str(w))
        
        try:
            check_redis()
        except ValidationWarning as w:
            warnings.append(str(w))
        
        try:
            check_git_status()
        except ValidationWarning as w:
            warnings.append(str(w))
        
    except ValidationError as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("\nCannot proceed with experiments. Fix the above issues and retry.")
        sys.exit(1)
    
    except ValidationWarning as w:
        warnings.append(str(w))
    
    # Summary
    print("\n" + "="*60)
    if warnings:
        print(f"⚠️  {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"  • {w}")
        print("\nYou can proceed, but results may be affected.")
        sys.exit(2)
    else:
        print("✅ ALL CHECKS PASSED - Ready to run experiments!")
        sys.exit(0)


if __name__ == "__main__":
    main()
