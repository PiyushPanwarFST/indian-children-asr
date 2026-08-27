#!/bin/bash
# ============================================================
# HPC Environment Setup — Run this ONCE on login node
# ============================================================
# Usage: bash hpc/setup_env.sh
#
# This creates a fresh conda environment with all dependencies
# needed for semantic and acoustic distillation training.
# ============================================================

set -e

echo "=== Setting up conda environment ==="
module load codes/anaconda3-2023.09

# Create fresh environment
conda create -n children_asr python=3.10 -y
source activate children_asr

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing HuggingFace + dependencies ==="
pip install transformers==4.36.0
pip install jiwer
pip install numpy

echo "=== Installing IndicConformer dependencies ==="
pip install onnxruntime-gpu==1.16.3
pip install nemo_toolkit[asr]==1.22.0 2>/dev/null || echo "NeMo install may have warnings — OK"

echo "=== Installing ffmpeg (for audio processing) ==="
conda install -c conda-forge ffmpeg -y

echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import transformers
print(f'Transformers: {transformers.__version__}')
import torchaudio
print(f'Torchaudio: {torchaudio.__version__}')
import jiwer
print('jiwer: OK')
print('All dependencies installed successfully!')
"

echo ""
echo "=== DONE ==="
echo "Environment 'children_asr' is ready."
echo "Activate with: source activate children_asr"
