# Tech Steering — Per-Model 3D Studio

## Backend

- **FastAPI** (`webapp/server.py`) + **uvicorn**, single process.
- **Single GPU worker thread** draining a `WORK` queue; jobs strictly sequential (one 16 GB GPU).
- **Hunyuan3D-2.1** shape (`hy3dshape` `Hunyuan3DDiTFlowMatchingPipeline`) + paint
  (`hy3dpaint`) via `webapp/pipeline.py:TextureWorker`.
- **Texturing primitives** (reused): `generate_shape`, `paint_faces` (hyface), `reface`,
  `fill_coverage_gaps`, `render_*`, `PROJECTION_CAMS`, `HYFACE_CORNER_CAMS`.
- **Image generation**: `webapp/image_edit.py:edit_image` → OpenAI `images.edit`
  (`OPENAI_IMAGE_MODEL`, default `gpt-image-2`) with Gemini fallback (`GEMINI_IMAGE_MODEL`,
  default `gemini-3-pro-image`). Mesh-free seed→views precedent: `webapp/elevations.py`.
- **Export**: `webapp/blender_convert.py` (headless Blender GLB→FBX/.blend); `BLENDER_BIN`.
- **Persistence today**: none beyond flat `{uid}_*.glb/png` under `HY3D_OUTPUT_DIR`
  (`webapp/outputs`) + a disk-glob `/api/gallery`. New per-model registry is net-new.

## Frontend

- **Next.js 16 / React 19**, SWR, `<model-viewer>` for preview. Talks to relative `/api/*`.
- Contract is defined in `lib/api.ts` (header), `lib/types.ts`, `lib/views.ts`,
  `lib/mock-backend.ts`; toggle real backend with `NEXT_PUBLIC_USE_MOCK=false`.

## Config / Env

- `OPENAI_API_KEY` / `GEMINI_API_KEY` (gate all generation), `OPENAI_IMAGE_MODEL`,
  `GEMINI_IMAGE_MODEL`, `HY3D_OUTPUT_DIR`, `BLENDER_BIN`, `HYFACE_*`, `REFACE_DEPTH_BAND`,
  `GAPFILL_*`. Loaded from repo-root `.env` via `_load_local_env`.

## Constraints

- Single GPU, sequential jobs; gpt-image calls are network-bound (not GPU).
- gpt-image drifts/reframes (~12%) and can mirror side views; orthographic fidelity is
  prompt-enforced only.
- Outputs are albedo-only matte (`_force_matte`).
