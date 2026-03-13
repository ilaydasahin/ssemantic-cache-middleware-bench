#!/bin/bash
# ============================================
# Download ONNX embedding models for CPU inference
# ============================================
# Downloads and converts 3 sentence-transformer models to ONNX format.
# 
# Models:
#   1. all-MiniLM-L6-v2      (384-dim, ~80MB)
#   2. all-mpnet-base-v2     (768-dim, ~420MB)
#   3. paraphrase-TinyBERT-L6-v2 (312-dim, ~60MB)
#
# Usage: ./download_models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"

echo "=== Downloading Embedding Models ==="
mkdir -p "${MODEL_DIR}"

# Check Python + dependencies
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

echo "--- Installing Python dependencies ---"
pip3 install --quiet sentence-transformers onnx onnxruntime optimum[onnxruntime] transformers torch

# Function to download and export a model
download_and_export() {
    local model_name=$1
    local short_name=$2
    local output_dir="${MODEL_DIR}/${short_name}"

    if [ -d "${output_dir}" ] && [ -f "${output_dir}/model.onnx" ]; then
        echo "  ✅ ${short_name} already exists, skipping"
        return
    fi

    echo "  📥 Downloading ${model_name}..."
    
    python3 << PYEOF
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import os

model_name = "${model_name}"
output_dir = "${output_dir}"

print(f"    Exporting {model_name} to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"    ✅ Saved to {output_dir}")

# Verify
import onnxruntime as ort
session = ort.InferenceSession(os.path.join(output_dir, "model.onnx"))
print(f"    ✅ ONNX model verified: inputs={[i.name for i in session.get_inputs()]}")
PYEOF
}

echo ""
echo "--- Model 1/3: all-MiniLM-L6-v2 (384d, recommended) ---"
download_and_export "sentence-transformers/all-MiniLM-L6-v2" "all-MiniLM-L6-v2"

echo ""
echo "--- Model 2/3: all-mpnet-base-v2 (768d, high quality) ---"
download_and_export "sentence-transformers/all-mpnet-base-v2" "all-mpnet-base-v2"

echo ""
echo "--- Model 3/3: paraphrase-TinyBERT-L6-v2 (312d, fastest) ---"
download_and_export "sentence-transformers/paraphrase-TinyBERT-L6-v2" "paraphrase-TinyBERT-L6-v2"

echo ""
echo "=== All models downloaded ==="
echo "Directory: ${MODEL_DIR}"
ls -la "${MODEL_DIR}"
echo ""

# Quick benchmark: embedding speed comparison
echo "--- Quick Speed Test ---"
python3 << 'PYEOF'
import time
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import os

models_dir = os.environ.get("MODEL_DIR", "models")
test_text = "How do I reset my password in the web application?"

for model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-TinyBERT-L6-v2"]:
    model_path = os.path.join(models_dir, model_name, "model.onnx")
    if not os.path.exists(model_path):
        continue
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(models_dir, model_name))
    session = ort.InferenceSession(model_path)
    
    inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True, max_length=256)
    
    # Warmup
    for _ in range(3):
        session.run(None, dict(inputs))
    
    # Benchmark (10 runs)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        outputs = session.run(None, dict(inputs))
        times.append((time.perf_counter() - start) * 1000)
    
    embedding = outputs[0][0].mean(axis=0)  # Mean pooling
    avg_ms = np.mean(times)
    print(f"  {model_name:40s} dim={len(embedding):4d}  avg={avg_ms:.1f}ms")

PYEOF

echo "=== Done ==="
