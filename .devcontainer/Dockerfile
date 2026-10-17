FROM docker.io/pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

# Only the essentials: the pipeline calls hosted models, and build-essential covers packages
# that compile on install.
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

# Install OpenCode CLI globally
RUN curl -fsSL https://opencode.ai/v2/install | bash \
    && mv /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && rm -rf /root/.opencode

# Pinned so every rebuild gets the same uv.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /workspaces/zir-image-enhancer

# Dependencies are installed in the image (not in postCreateCommand) so the container starts
# ready, install errors are visible during the build, and the same image works on Vast.ai.
# `tools/` is copied before the install because the frozen scorer is a path dependency.
COPY pyproject.toml README.md ./
COPY tools ./tools

# The constraint file freezes the torch and torchvision already in the image, so uv can never
# replace the CUDA build with a different wheel from PyPI. If grep finds nothing, the build
# fails here on purpose: it means the base image changed. `--all-extras` pulls the `scorer`
# group (the vendored `sartiq-score` from tools/scorer via [tool.uv.sources]), the `select`
# group (CLIP and AlexNet for the v4 selector) and `dev`.
RUN uv pip freeze --system | grep -iE '^(torch|torchvision)==' > /tmp/torch-constraints.txt \
    && uv pip install --system -r pyproject.toml --all-extras -c /tmp/torch-constraints.txt \
    && python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)" \
    && python -c "import sartiq_score; print('sartiq-score', sartiq_score.__version__)"
