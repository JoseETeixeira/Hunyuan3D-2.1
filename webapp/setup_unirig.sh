#!/usr/bin/env bash
# One-time UniRig (https://github.com/VAST-AI-Research/UniRig) setup in an ISOLATED conda env so its
# heavy/incompatible deps (torch, flash-attn, spconv, numpy==1.26.4) never perturb the Hunyuan env.
# Run inside the container:
#   docker compose exec hunyuan3d-studio bash webapp/setup_unirig.sh
#
# Mirrors setup_mvadapter.sh: clones to a path on the persisted `unirig` volume and creates the env
# by PREFIX so it survives container recreation and never shadows the baked-in hunyuan3d21 env.
# Model weights pull lazily from HuggingFace (VAST-AI/UniRig) on the first rig job (cached in hf-cache).
#
# Wheel pins below target the container's CUDA 12.8 / torch 2.7 (RTX 50-series, sm_120). spconv,
# torch_scatter/torch_cluster, and flash-attn ship CUDA-version-specific wheels — if your GPU/CUDA
# differs, adjust TORCH_WHL_CU / the spconv package / the PyG find-links URL accordingly.
set -euo pipefail

UNIRIG_DIR="${UNIRIG_DIR:-/opt/unirig/UniRig}"
PREFIX="${UNIRIG_CONDA_PREFIX:-/opt/unirig/env}"
TORCH_WHL_CU="${UNIRIG_TORCH_CU:-cu128}"   # match the host torch build (cu128 = Blackwell/sm_120)
TORCH_VER="${UNIRIG_TORCH_VER:-2.7.0}"
SPCONV_PKG="${UNIRIG_SPCONV:-spconv-cu126}"  # no cu128 wheel yet; cu126 runs on a 12.8 runtime

source /workspace/miniconda3/etc/profile.d/conda.sh
run() { conda run --no-capture-output -p "$PREFIX" "$@"; }

echo "[unirig] 1/5 clone -> $UNIRIG_DIR"
mkdir -p "$(dirname "$UNIRIG_DIR")"
if [ ! -d "$UNIRIG_DIR/.git" ]; then
  git clone https://github.com/VAST-AI-Research/UniRig "$UNIRIG_DIR"
fi
cd "$UNIRIG_DIR"

echo "[unirig] 2/5 conda env @ $PREFIX (python 3.11)"
if [ ! -x "$PREFIX/bin/python" ]; then
  conda create -y -p "$PREFIX" python=3.11
fi

echo "[unirig] 3/5 torch $TORCH_VER/$TORCH_WHL_CU + numpy pin"
run pip install "torch==${TORCH_VER}" torchvision --index-url "https://download.pytorch.org/whl/${TORCH_WHL_CU}"
run pip install "numpy==1.26.4"

echo "[unirig] 4/5 UniRig requirements + spconv + torch-geometric ext + flash-attn"
# UniRig's own deps (transformers, lightning, trimesh, etc.).
[ -f requirements.txt ] && run pip install -r requirements.txt || echo "[unirig] no requirements.txt in repo (skipping)"
run pip install "$SPCONV_PKG" || echo "[unirig] WARN: $SPCONV_PKG failed; install the spconv wheel matching your CUDA"
# torch_scatter / torch_cluster: prebuilt against the exact torch+cu build.
run pip install torch_scatter torch_cluster \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${TORCH_WHL_CU}.html" \
  || echo "[unirig] WARN: torch_scatter/cluster wheels not found for ${TORCH_VER}+${TORCH_WHL_CU}; adjust the find-links URL"
# flash-attn: build against the env's torch; slow first build, JITs for sm_120.
TORCH_CUDA_ARCH_LIST="12.0" run pip install flash-attn --no-build-isolation \
  || echo "[unirig] WARN: flash-attn build failed; install a wheel matching torch/CUDA or disable it in UniRig config"

echo "[unirig] 5/5 done."
echo "[unirig] Set in the studio env:  UNIRIG_DIR=$UNIRIG_DIR  UNIRIG_PYTHON=$PREFIX/bin/python"
echo "[unirig] Weights (VAST-AI/UniRig) download lazily from HuggingFace on the first rig job."
