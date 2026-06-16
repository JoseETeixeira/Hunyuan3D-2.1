# Hunyuan3D-2.1 Studio (meshy-style web app)

A web interface on top of Hunyuan3D-2.1: upload an image, generate a 3D **shape**,
preview it in the browser, then generate a **PBR texture** — packaged to run with a
single Docker command.

## What you get

- **Upload** any image (drag & drop or browse), with automatic background removal.
- **Two-step generation like meshy.ai**
  1. *Generate Model* → untextured mesh, previewed live in a rotatable 3D viewer.
  2. *Generate Texture* → PBR-textured GLB.
  - Optional **Auto-texture** runs both steps back to back.
- **Live 3D preview** via `<model-viewer>`, with shape/textured tabs and GLB download.
- Tunable params: inference steps, guidance scale, octree resolution, seed, texture face count.

## Run it (one command)

```bash
docker compose up --build
```

Then open <http://localhost:8080>.

> First launch builds the image (large — pulls CUDA, PyTorch, native extensions) and
> the first generation downloads the Hunyuan3D-2.1 weights from HuggingFace (tens of GB,
> cached in the `hf-cache` volume for next time).

### Requirements

- NVIDIA GPU (recommended ≥ 24 GB VRAM; lower works with `--low_vram_mode`).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so Docker can access the GPU.
- Linux host, or Windows via WSL2 with GPU passthrough.

### Lower-VRAM / faster options

Edit the `command:` in `docker-compose.yml`:

```yaml
command: ["python", "-m", "webapp.server", "--host", "0.0.0.0", "--port", "8080",
          "--preload", "--low_vram_mode", "--enable_flashvdm"]
```

## Architecture

```
Browser (static SPA)  ──HTTP──►  FastAPI  ──►  single GPU worker thread (queue)
  index.html / app.js            webapp/server.py     webapp/pipeline.py
  <model-viewer> preview         /api/generate        TextureWorker
                                 /api/jobs/{id}         ├─ generate_shape()  → {id}_shape.glb
                                 /api/jobs/{id}/texture ├─ generate_texture()→ {id}_textured.glb
                                 /api/files/{name}      └─ reuses hy3dshape + hy3dpaint pipelines
```

- One GPU → one sequential worker thread; jobs are queued, never run concurrently.
- The shape and texture pipelines are loaded once and reused for every job.
- Call signatures mirror the official `gradio_app.py` (`use_safetensors=False`,
  `output_type='mesh'`, `export_to_trimesh`, `FaceReducer`) so output matches the demo.

## HTTP API

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `POST` | `/api/generate` | multipart image + params → `{id}`; starts shape job |
| `POST` | `/api/jobs/{id}/texture` | queue the texture step for a shape-ready job |
| `GET`  | `/api/jobs/{id}` | job status, progress, `shape_url`, `textured_url` |
| `GET`  | `/api/files/{name}` | download a generated `.glb` |
| `GET`  | `/api/health` | model-ready + queue depth |

Job status flow: `queued → processing_shape → shape_ready → queued_texture → processing_texture → completed` (or `failed`).

## Run without Docker

From the repo root, inside the Hunyuan3D-2.1 conda env (see `docker/Dockerfile` for the
full setup, including the native extensions and RealESRGAN weights):

```bash
pip install python-multipart
python -m webapp.server --port 8080 --preload
```
