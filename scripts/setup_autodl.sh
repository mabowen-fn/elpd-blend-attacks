#!/usr/bin/env bash
# =============================================================================
# setup_autodl.sh — One-shot environment setup for AutoDL GPU instances
#
# Run this ONCE after cloning the repo on AutoDL:
#   bash setup_autodl.sh
#
# It will:
#   1. Install Python dependencies (CUDA-enabled torch from PyTorch index)
#   2. Log in to Weights & Biases
#   3. Verify the GPU is visible to PyTorch
#   4. Validate the ImageNet data directory structure
#   5. Run a 3-image smoke test to confirm everything works end-to-end
# =============================================================================

set -e   # exit immediately on any error

# ── Colours for readability ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Python version check ───────────────────────────────────────────────────
info "Checking Python version..."
PY=$(python3 --version 2>&1)
info "Found: $PY"
python3 -c "import sys; assert sys.version_info >= (3, 9), 'Python 3.9+ required'" \
    || error "Python 3.9+ is required"

# ── 2. Install dependencies ───────────────────────────────────────────────────
info "Installing Python dependencies..."
# AutoDL base images come with CUDA PyTorch pre-installed.
# Install everything EXCEPT torch/torchvision to avoid downgrading the CUDA build.
pip install --quiet \
    wandb \
    pandas \
    matplotlib \
    numpy \
    Pillow \
    PyYAML

# Verify torch is CUDA-enabled (should already be installed on AutoDL)
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check AutoDL instance type'
print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" || error "PyTorch CUDA check failed"

info "Dependencies installed successfully."

# ── 3. Weights & Biases login ─────────────────────────────────────────────────
info "Logging in to Weights & Biases..."
wandb login || warn "W&B login failed — run manually: wandb login"

# ── 4. Verify ImageNet data ───────────────────────────────────────────────────
info "Checking ImageNet data directory..."
IMAGENET_VAL="data/val"
if [ ! -d "$IMAGENET_VAL" ]; then
    warn "ImageNet val dir not found at: $IMAGENET_VAL"
    warn "Expected structure: data/val/<class_name>/<image>.JPEG"
    warn "On AutoDL: symlink your ImageNet path → data/"
    warn "  e.g.: ln -s /root/autodl-tmp/imagenet/val data/val"
    warn "Continuing — smoke test will use CIFAR-10 instead."
    USE_CIFAR=true
else
    N_CLASSES=$(ls "$IMAGENET_VAL" | wc -l)
    N_IMAGES=$(find "$IMAGENET_VAL" -name "*.JPEG" | wc -l)
    info "ImageNet val: $N_CLASSES classes, $N_IMAGES images"
    USE_CIFAR=false
fi

# ── 5. Smoke test ─────────────────────────────────────────────────────────────
info "Running 3-image smoke test..."
if [ "$USE_CIFAR" = true ]; then
    python3 scripts/run_experiment.py --n_images 3 --steps 5 --no_wandb \
        || error "Smoke test failed"
else
    python3 scripts/run_experiment.py --config configs/config_autodl.yaml \
        --n_images 3 --steps 5 --no_wandb \
        || error "Smoke test failed"
fi
info "Smoke test passed."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Setup complete. Run the full experiment with:${NC}"
echo ""
echo -e "  ${YELLOW}python scripts/run_experiment.py --config configs/config_autodl.yaml${NC}"
echo ""
echo -e "  To resume a crashed run:"
echo -e "  ${YELLOW}python scripts/run_experiment.py --config configs/config_autodl.yaml --resume${NC}"
echo -e "${GREEN}============================================================${NC}"
