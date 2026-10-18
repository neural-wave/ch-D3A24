# uv-native image: no conda, no system-python installs. uv builds the project
# environment once at /opt/venv, outside the workspace bind mount, so it survives at
# runtime and the same image works on Vast.ai. Pinned so every rebuild gets the same uv
# (reproducible with uv.lock via `uv sync --locked`).
FROM ghcr.io/astral-sh/uv:0.9.9-python3.11-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

# Runtime libraries the wheels expect; opencv-headless and rawpy bundles the rest.
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

WORKDIR /workspaces/zir-image-enhancer

# Dependencies first (tools/ before the sync because the scorer is a path dependency),
# then the source. Installing at build time keeps the container ready on start and makes
# install errors visible during the build rather than at first run. `--all-extras` pulls
# the `scorer` extra; the `dev` group is installed by default.
COPY pyproject.toml uv.lock README.md ./
COPY tools ./tools
RUN uv sync --all-extras --no-install-project --locked

COPY . .
RUN uv sync --all-extras --locked \
    && uv run python -c "import torch, sartiq_score, zir_image_enhancer; print('env ok')"
