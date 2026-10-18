#!/bin/bash

set -euo pipefail

# -------------------------------------------------------
# HITL WiSe26 Course Environment Setup
# Common baseline for the semester projects
# -------------------------------------------------------

ENV_NAME="hitl-course"
PYTHON_VERSION="3.10"

# Optional: set USE_CUDA=1 before running the script if you
# explicitly want the CUDA 11.8 PyTorch build.
# Example:
#   USE_CUDA=1 bash install_env_revised.sh
USE_CUDA="${USE_CUDA:-0}"

# -------------------------------------------------------
# 1. Make conda available
# -------------------------------------------------------

# HSBI / YourAI cluster setup
if type module >/dev/null 2>&1; then
    module load miniconda3 || true
fi

# If conda is still unavailable, stop with a useful message.
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found."
    echo "Please load/install Miniconda or Anaconda first, then rerun this script."
    exit 1
fi

# Required for 'conda activate' inside a shell script.
eval "$(conda shell.bash hook)"

# -------------------------------------------------------
# 2. Create or reuse the course environment
# -------------------------------------------------------

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists; reusing it."
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" pip --yes
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# -------------------------------------------------------
# 3. Install PyTorch
# -------------------------------------------------------

if [ "$USE_CUDA" = "1" ]; then
    echo "Installing PyTorch with CUDA 11.8 support..."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing CPU PyTorch (default)..."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# -------------------------------------------------------
# 4. Install common course libraries
# -------------------------------------------------------

python -m pip install \
    numpy scipy pandas scikit-learn matplotlib tqdm \
    jupyterlab ipykernel \
    transformers "datasets==3.6.0" sentence-transformers \
    pillow opencv-python-headless

# -------------------------------------------------------
# 5. Register the Jupyter kernel
# -------------------------------------------------------

python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "HITL Course"

# -------------------------------------------------------
# 6. Optional packages for later course arcs
# -------------------------------------------------------

# Interpretability / explanation methods (install when needed):
# python -m pip install shap lime

# NLP utilities (install only if a project needs them):
# python -m pip install nltk spacy
# python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm

# OCR support (only if students need local OCR rather than provided OCR):
# python -m pip install pytesseract
# Note: pytesseract also requires the Tesseract system package.

# -------------------------------------------------------
# 7. Quick environment check
# -------------------------------------------------------

python - <<'PY'
import sys
import numpy
import pandas
import sklearn
import torch
import torchvision
import transformers
import datasets
import sentence_transformers
import PIL
import cv2

print("\nEnvironment check passed.")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo ""
echo "HITL Course environment setup complete!"
echo "Activate it later with: conda activate $ENV_NAME"
echo "Jupyter kernel: HITL Course"