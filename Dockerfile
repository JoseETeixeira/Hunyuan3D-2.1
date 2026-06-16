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
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0"

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

# ── Project source (local tree, includes webapp/) ───────────────────────
COPY . /workspace/Hunyuan3D-2.1
WORKDIR /workspace/Hunyuan3D-2.1

# ── Build native extensions ─────────────────────────────────────────────
# --no-build-isolation so setup.py can see the already-installed torch
# (PEP 517 isolated build env otherwise hides it -> ModuleNotFoundError: torch).
RUN cd hy3dpaint/custom_rasterizer && \
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0" && \
    export CUDA_NVCC_FLAGS="-allow-unsupported-compiler" && \
    pip install -e . --no-build-isolation
RUN cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh

# ── RealESRGAN weights (baked into image) ───────────────────────────────
RUN cd hy3dpaint && mkdir -p ckpt && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ckpt

# ── Path fixes used by the official docker build ────────────────────────
RUN sed -i 's/self\.multiview_cfg_path = "cfgs\/hunyuan-paint-pbr\.yaml"/self.multiview_cfg_path = "hy3dpaint\/cfgs\/hunyuan-paint-pbr.yaml"/' hy3dpaint/textureGenPipeline.py && \
    sed -i 's/custom_pipeline = config\.custom_pipeline/custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")/' hy3dpaint/utils/multiview_utils.py

ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/hunyuan3d21/lib:${LD_LIBRARY_PATH}"
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get install -y libxi6 libgconf-2-4 libxkbcommon-x11-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p webapp/outputs

# Blender (headless) for GLB -> FBX / .blend export in the web app.
RUN apt-get update && apt-get install -y --no-install-recommends blender && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 8080
CMD ["python", "-m", "webapp.server", "--host", "0.0.0.0", "--port", "8080"]
