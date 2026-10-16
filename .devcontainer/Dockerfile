FROM docker.io/pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

# opencv-python-headless does not need the X11 libraries, and rawpy wheels already bundle LibRaw,
# so only the essentials stay. build-essential covers packages that compile on install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Pinned so every rebuild gets the same uv.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /workspaces/zir-image-enhancer

# Dependencies are installed in the image (not in postCreateCommand) so the container starts
# ready, install errors are visible during the build, and the same image works on Vast.ai.
COPY pyproject.toml README.md ./

# The constraint file freezes the torch and torchvision already in the image, so uv can never
# replace the CUDA build with a different wheel from PyPI. If grep finds nothing, the build
# fails here on purpose: it means the base image changed.
RUN uv pip freeze --system | grep -iE '^(torch|torchvision)==' > /tmp/torch-constraints.txt \
    && uv pip install --system -r pyproject.toml --all-extras -c /tmp/torch-constraints.txt \
    && python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
