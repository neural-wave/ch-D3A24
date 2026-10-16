# Zir Image Enhancer

A local AI image enhancer pipeline built for the [Neuralwave 2026 Hackathon](https://neuralwave.ch) by the `D3A24` team for [Sartiq AI](https://www.sartiq.ai/).

## Quickstart (local, VS Code Dev Container)

### Requirements

- **VS Code** with the **Dev Containers** extension (`ms-vscode-remote.remote-containers`).
- **Docker:**
  - **Windows:** Docker Desktop with the WSL2 backend. For GPU access, an up-to-date NVIDIA driver on Windows is enough; you do not install the NVIDIA Container Toolkit.
  - **Linux:** Docker Engine plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **GPU is optional.** Without an NVIDIA GPU the container still starts and runs on CPU (slow, but fine for UI and code work).

> **Windows tip:** clone the repo inside WSL2 (for example `~/code/zir-image-enhancer`), not under `C:\`. Files on the Windows drive are much slower inside the container.

### Steps

1. Clone this repository and open it in VS Code.
2. Click **"Reopen in Container"** (or run `Dev Containers: Reopen in Container` from the Command Palette). The first build takes a while because it installs every dependency into the image.
3. Inside the container, start the app with hot reloading:

   ```bash
   gradio src/zir_image_enhancer/app.py
   ```

   Or without hot reloading:

   ```bash
   zir-app
   ```

4. Open `http://localhost:7860`.

Model weights are cached in the `zir-model-cache` Docker volume, so they are downloaded only once and never end up in the repo.

## Running on Vast.ai

A Vast.ai instance is already a container, so the Dev Container itself is not used there. Use the same image instead:

1. Build and push the image once (from the repo root):

   ```bash
   docker build -f .devcontainer/Dockerfile -t ghcr.io/<org>/zir-image-enhancer:latest .
   ```

   ```bash
   docker push ghcr.io/<org>/zir-image-enhancer:latest
   ```

2. Create a Vast.ai template with that image, expose port `7860`, and pick an **on-demand** RTX 4090.
3. Connect over SSH, clone the repo into `/workspaces/zir-image-enhancer` and run:

   ```bash
   uv pip install --system --no-deps -e .
   ```

4. Start the app inside `tmux` so it survives SSH disconnects. Use `share=True` in `demo.launch()` (or a Cloudflare tunnel) to get the HTTPS link phones need for the camera.

## Optional dependency groups

All groups are installed in the image. If you install outside the container:

| Group | What for |
|---|---|
| `generative` | Optional creative mode (diffusers, accelerate) |
| `eval` | Metrics and plots (pyiqa, pandas, matplotlib) |
| `dev` | ruff, mypy, pytest, pre-commit |

## Development

```bash
ruff check . && ruff format .
```

```bash
mypy src
```

```bash
pytest
```
