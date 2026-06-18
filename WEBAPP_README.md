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

## Per-Model Studio (current frontend)

The current frontend is a Next.js **per-model** studio (`webapp/studio.py` +
`webapp/reference_views.py`). A *model* is a durable, named aggregate (10 reference views, mesh,
texture) persisted at `outputs/models/{id}/model.json` — reusable across runs and process restarts.
Texturing is limited to **per-face AI paint (`hyface`)** and **`reface`**; the older modes
(`hunyuan`, `projection`, `gptproject`, `mvadapter`, `mvgpt`) were removed.

Flow: create a model → upload a seed → generate the 10 reference views with gpt-image-2 along the
dependency graph (front → cardinals → corners), approving/tweaking each → `texture/base` (mesh +
per-face AI paint over all approved refs; reaches `complete`) → optional per-face `reface`/`paint`
edits → download glb/fbx/blend.

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET/POST` | `/api/models` | list / create (multipart `name`, optional `seed_image`) |
| `GET/PATCH/DELETE` | `/api/models/{id}` | fetch / rename (`{name}`) / delete |
| `POST` | `/api/models/{id}/references/{view}/generate` | gpt-image-2 view (`{edit_prompt?}`) → Job |
| `POST` | `/api/models/{id}/references/{view}/upload` | upload a custom view (auto-approved) |
| `POST` | `/api/models/{id}/references/{view}/approve` | approve a generated view |
| `GET`  | `/api/models/{id}/references/{view}/image`, `/seed` | reference / seed PNG |
| `POST` | `/api/models/{id}/texture/base` | mesh + per-face AI paint (MeshConfig) → Job |
| `POST` | `/api/models/{id}/texture/reface/{view}` | depth-aware reface (`{edit_prompt?}`) → Job |
| `POST` | `/api/models/{id}/faces/{view}/edit` | `mode=paint\|reface`, `edit_prompt?`, `image?` → Job |
| `GET`  | `/api/jobs/{id}` | job status; completed jobs embed the full `Model` |
| `GET`  | `/api/models/{id}/download/{fmt}` | `glb` \| `fbx` \| `blend` |

Views: `front, back, left, right, top, bottom, front-left, front-right, back-left, back-right`.
`texture/base` requires all 10 references approved and runs on the GPU worker; reference generation
runs on a separate network lane (gpt-image-2 / Gemini; needs `OPENAI_API_KEY` and/or
`GEMINI_API_KEY`).

### Serving the Next.js frontend same-origin

```bash
# in the Next.js app:  next build  (next.config has output: 'export')  ->  out/
HY3D_WEBUI_DIR=/path/to/app/out python -m webapp.server --port 8080 --preload
```

FastAPI serves the static export at `/` (mounted after `/api/*`), so the app's relative `/api/*`
calls resolve same-origin. Set `NEXT_PUBLIC_USE_MOCK=false` for the built app. If `HY3D_WEBUI_DIR`
is unset, the bundled (legacy) static UI is served instead.

### Docker

`docker-compose.yml` already sets `HY3D_WEBUI_DIR=/workspace/Hunyuan3D-2.1/webapp/webui` (under the
live `./webapp` bind mount), so you only need to drop the built export there — no image rebuild:

```bash
# 1) build the export and copy it under the bind mount
cd /path/to/3-d-model-generation-workflow && NEXT_PUBLIC_USE_MOCK=false pnpm build
cp -r out/. /path/to/Hunyuan3D-2.1/webapp/webui/
# 2) run (keys in Hunyuan3D-2.1/.env: OPENAI_API_KEY / GEMINI_API_KEY / HF_TOKEN)
cd /path/to/Hunyuan3D-2.1 && docker compose up -d
# open http://localhost:8080  (logs: docker compose logs -f ; stop: docker compose down)
```

## Run without Docker

From the repo root, inside the Hunyuan3D-2.1 conda env (see `docker/Dockerfile` for the
full setup, including the native extensions and RealESRGAN weights):

```bash
pip install python-multipart
python -m webapp.server --port 8080 --preload
```
