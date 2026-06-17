# Hunyuan3D-2.1 Studio image: builds the model + the meshy-style web app from the
# local source tree (not a fresh git clone) so our webapp/ is included.
# Based on the official docker/Dockerfile build steps.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL name="hunyuan3d21-studio" maintainer="hunyuan3d21-studio"

WORKDIR /workspace

# ── System dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip git-lfs
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev \
    libegl1-mesa-dev libgles2-mesa-dev cmake curl mesa-utils-extra libxrender1
RUN apt-get install -y libeigen3-dev python3-dev python3-setuptools libcgal-dev

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV PYOPENGL_PLATFORM=egl
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# GPU compute capability for the native CUDA extensions (custom_rasterizer, mesh painter).
# Default to just the RTX 50-series (Blackwell, sm_120 = "12.0") so nvcc doesn't compile for
# six architectures. Override for other GPUs, e.g. --build-arg TORCH_CUDA_ARCH_LIST="8.6;12.0".
ARG TORCH_CUDA_ARCH_LIST="12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ── Conda + Python 3.10 env ─────────────────────────────────────────────
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/workspace/miniconda3/bin:${PATH}"
RUN conda init bash
RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/r
# Force conda's classic solver. The new libmamba "sharded repodata" SQLite cache
# (conda 26.x) hits `sqlite3.OperationalError: database is locked` non-deterministically
# during docker build on WSL2. Classic solver bypasses conda_libmamba_solver entirely.
ENV CONDA_SOLVER=classic
RUN conda create -n hunyuan3d21 python=3.10 && echo "source activate hunyuan3d21" > ~/.bashrc
ENV PATH="/workspace/miniconda3/envs/hunyuan3d21/bin:${PATH}"
RUN conda config --set always_yes true
RUN conda install Ninja
RUN conda install cuda -c nvidia/label/cuda-12.8.1 -y
RUN conda install -c conda-forge libstdcxx-ng -y

# ── PyTorch (CUDA 12.4) ─────────────────────────────────────────────────
# torch cu128 / 2.7.x — required for RTX 50-series (Blackwell, sm_120) kernels.
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# pip network resilience: the requirements install pulls many large wheels; a dropped
# connection ("Response ended prematurely") should retry instead of failing the whole build.
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

# ── Python deps (cached on requirements.txt) ────────────────────────────
COPY requirements.txt /workspace/Hunyuan3D-2.1/requirements.txt
RUN pip install -r /workspace/Hunyuan3D-2.1/requirements.txt
RUN pip install python-multipart==0.0.12
# setuptools >=81 removed pkg_resources, which basicsr/realesrgan/pytorch-lightning
# still import at runtime (paint model load fails with ModuleNotFoundError: pkg_resources).
RUN pip install "setuptools<81"
# Image-edit SDKs for gpt/projection texturing: OpenAI (gpt-image-2) + Google GenAI
# (Gemini "nano banana" fallback).
RUN pip install "openai>=1.40" google-genai

# ── Native-extension sources ONLY (not the full tree) ───────────────────
# Copying just the two dirs the CUDA/C++ builds need keeps the expensive compile layers
# cached when webapp/ (or anything else) changes. The full tree lands much further down.
COPY hy3dpaint/custom_rasterizer /workspace/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
COPY hy3dpaint/DifferentiableRenderer /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
WORKDIR /workspace/Hunyuan3D-2.1

# ── Build native extensions ─────────────────────────────────────────────
# --no-build-isolation so setup.py can see the already-installed torch
# (PEP 517 isolated build env otherwise hides it -> ModuleNotFoundError: torch).
# TORCH_CUDA_ARCH_LIST inherited from the ENV above (default "12.0" / RTX 50-series).
RUN cd hy3dpaint/custom_rasterizer && \
    export CUDA_NVCC_FLAGS="-allow-unsupported-compiler" && \
    pip install -e . --no-build-isolation
RUN cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh

# ── RealESRGAN weights (baked into image; source-independent) ────────────
RUN cd hy3dpaint && mkdir -p ckpt && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ckpt

# ── Runtime apt libs (source-independent; kept above COPY . so edits don't rerun apt) ────
RUN apt-get install -y libxi6 libgconf-2-4 libxkbcommon-x11-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Blender (headless) for GLB -> FBX/.blend export AND the mvgpt camera-projection texture bake.
# Debian's apt `blender` is 3.0.1, which is too old: it can't compile CUDA kernels for recent
# GPUs, can't render Workbench/EEVEE headless, and lacks nodes/colorspaces the projection bake
# uses. Install a current LTS from the official tarball instead. Override with --build-arg.
# Source-independent, so kept above COPY . to stay cached across webapp/ edits.
ARG BLENDER_VERSION=4.2.3
ARG BLENDER_SERIES=4.2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6 xz-utils wget && \
    wget -q "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
        -O /tmp/blender.tar.xz && \
    mkdir -p /opt/blender && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && \
    ln -sf /opt/blender/blender /usr/local/bin/blender && rm /tmp/blender.tar.xz && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# blender_project.py / blender_convert.py stay version-robust (CPU-Cycles default + node/colorspace
# fallbacks) so they also work if an older Blender is present.

ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/hunyuan3d21/lib:${LD_LIBRARY_PATH}"
ENV HF_HOME=/root/.cache/huggingface

# ── Full project source (includes webapp/) ──────────────────────────────
# CACHE BOUNDARY: editing anything in the tree re-runs ONLY from here down — the COPY,
# the mkdir, and the sed path-fixes. The CUDA compile, Blender, apt, and weights above
# stay cached. (The COPY re-overlays the two native-ext dirs with identical source; the
# compiled .so files built above are not in the build context, so they survive untouched.)
COPY . /workspace/Hunyuan3D-2.1
RUN mkdir -p webapp/outputs

# ── Path fixes used by the official docker build (need the full source) ──
RUN sed -i 's/self\.multiview_cfg_path = "cfgs\/hunyuan-paint-pbr\.yaml"/self.multiview_cfg_path = "hy3dpaint\/cfgs\/hunyuan-paint-pbr.yaml"/' hy3dpaint/textureGenPipeline.py && \
    sed -i 's/custom_pipeline = config\.custom_pipeline/custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")/' hy3dpaint/utils/multiview_utils.py

EXPOSE 8080
CMD ["python", "-m", "webapp.server", "--host", "0.0.0.0", "--port", "8080"]
