# uv-native image:
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

# Only the essentials: the pipeline calls hosted models, build-essential covers packages
# that compile on install, and the rest are runtime libraries torch and Gradio expect.
# BuildKit cache mounts keep the apt indices and .deb cache across rebuilds without
# baking them into the image.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && git lfs install

# Install OpenCode CLI globally
RUN curl -fsSL https://opencode.ai/v2/install | bash \
    && mv /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && rm -rf /root/.opencode

WORKDIR /workspaces/zir-image-enhancer

# The heavy `scorer` extra (torch + CUDA/onnxruntime/open_clip/rembg) is opt-in: the app
# and CLI only call hosted models, so the default dev image stays lean. Build the eval
# image with `--build-arg SCORER=1`. Declared here so flipping it does not invalidate the
# apt/opencode layers above.
ARG SCORER=0

# Dependencies first (tools/ before the sync because the scorer is a path dependency),
# then the source. The uv cache mount makes rebuilds reuse already-downloaded wheels even
# though the runtime model-cache volume at /root/.cache is not mounted during the build.
COPY pyproject.toml uv.lock README.md ./
COPY tools ./tools
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eu; \
    if [ "$SCORER" = "1" ]; then EXTRAS="--all-extras"; else EXTRAS=""; fi; \
    uv sync $EXTRAS --no-install-project --locked

# Install the project itself once the source is present. Dependencies are already in
# place from the layer above, so this sync only wires up the package and the editable
# scorer.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eu; \
    if [ "$SCORER" = "1" ]; then EXTRAS="--all-extras"; else EXTRAS=""; fi; \
    uv sync $EXTRAS --locked; \
    if [ "$SCORER" = "1" ]; then \
        uv run python -c "import torch, sartiq_score, zir_image_enhancer; print('env ok (scorer)')"; \
    else \
        uv run python -c "import zir_image_enhancer; print('env ok')"; \
    fi
