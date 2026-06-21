#!/usr/bin/env bash
# One-time MV-Adapter SDXL setup in an ISOLATED conda env so it never perturbs the
# Hunyuan paint environment. Run inside the running container (compose SERVICE is `hunyuan3d`, not
# the container_name `hunyuan3d-studio`):
#   docker compose exec hunyuan3d bash webapp/setup_mvadapter.sh
#   # or, by container name:  docker exec -it hunyuan3d-studio bash webapp/setup_mvadapter.sh
#
# Downloads: SDXL base (~6.9G) + MV-Adapter ig2mv adapter (~3.6G) + VAE + BiRefNet
# pull lazily from HuggingFace on the first texture job (cached in the hf-cache
# volume). This script installs the env + the small bake-stage checkpoints.
#
# NOTE: the conda env (/workspace/miniconda3/envs/mvadapter) and the checkout
# (MVADAPTER_DIR) live inside the container and are lost if the container is
# recreated. To persist, bake them into the image or mount a volume — see the
# README note added alongside this script.
set -euo pipefail

MVADAPTER_DIR="${MVADAPTER_DIR:-/opt/mvadapter/MV-Adapter}"
PREFIX="${MVADAPTER_CONDA_PREFIX:-/opt/mvadapter/env}"
VARIANT="${MVADAPTER_VARIANT:-sd21}"

source /workspace/miniconda3/etc/profile.d/conda.sh
# Address the env by PREFIX path so it lives on the persisted volume and never
# shadows the baked-in hunyuan3d21 env.
run() { conda run --no-capture-output -p "$PREFIX" "$@"; }

echo "[mvadapter] 1/4 clone -> $MVADAPTER_DIR"
mkdir -p "$(dirname "$MVADAPTER_DIR")"
if [ ! -d "$MVADAPTER_DIR/.git" ]; then
  git clone https://github.com/huanngzh/MV-Adapter "$MVADAPTER_DIR"
fi
cd "$MVADAPTER_DIR"

echo "[mvadapter] 2/4 conda env @ $PREFIX (python 3.10 + torch cu128 for Blackwell)"
if [ ! -x "$PREFIX/bin/python" ]; then
  conda create -y -p "$PREFIX" python=3.10
fi
# Match the host's cu128 torch so sm_120 (RTX 50xx) kernels work. Do NOT pull cu118.
run pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "[mvadapter] 3/4 python deps (no torch; spandrel pinned per upstream; ninja for ext build)"
run pip install \
  ninja diffusers transformers peft accelerate huggingface_hub safetensors \
  controlnet_aux einops omegaconf kornia timm scikit-image opencv-python pillow \
  numpy sentencepiece trimesh open3d pytorch-lightning spandrel==0.4.1 \
  "pymeshlab==2022.2.post4" \
  jaxtyping typeguard gltflib "openai>=1.40" google-genai   # pymeshlab pinned (newer renamed Percentage->PercentageValue); jaxtyping/typeguard/gltflib missing from MV-Adapter's own requirements.txt; openai + google-genai for the GPT-refine step (gpt-image-2 -> Gemini nano-banana fallback)
# nvdiffrast: needs --no-build-isolation so its setup.py sees the env's torch (same
# reason the Dockerfile uses it for custom_rasterizer). CUDA kernels JIT-compile at
# runtime for sm_120 via the container's /usr/local/cuda (TORCH_CUDA_ARCH_LIST set).
TORCH_CUDA_ARCH_LIST="12.0" run pip install --no-build-isolation \
  git+https://github.com/NVlabs/nvdiffrast.git
# CV-CUDA is an optional fast path for the bake post-process; non-fatal if it fails.
run pip install cvcuda_cu12 \
  || echo "[mvadapter] WARN: cvcuda_cu12 wheel failed; bake post-process fast path disabled (non-fatal)"

echo "[mvadapter] 4/4 bake-stage checkpoints"
mkdir -p "$MVADAPTER_DIR/checkpoints"
[ -f checkpoints/RealESRGAN_x2plus.pth ] || wget -q \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
  -O checkpoints/RealESRGAN_x2plus.pth
[ -f checkpoints/big-lama.pt ] || wget -q \
  https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt \
  -O checkpoints/big-lama.pt

echo "[mvadapter] done. Pick MV-Adapter SDXL in the UI; SDXL + adapter weights"
echo "[mvadapter] (~11GB) download lazily on the first texture job."
echo "[mvadapter] On a 16GB GPU set MVADAPTER_VARIANT=sd21 if SDXL OOMs (variant=$VARIANT)."
