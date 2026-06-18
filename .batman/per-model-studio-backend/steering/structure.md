# Structure Steering — Per-Model 3D Studio

## Backend layout (`Hunyuan3D-2.1/webapp/`)

- `server.py` — FastAPI app, job queue, worker loop, texture-mode dispatch, all HTTP routes.
  The new per-model REST layer + model registry land here (or a new module mounted on the app).
- `pipeline.py` — `TextureWorker` (shape/paint/reface/gapfill/render). Reused; minimal change.
- `image_edit.py` — `edit_image` (OpenAI→Gemini). Reused by step-2 generation. KEEP.
- `elevations.py` — mesh-free seed→views precedent. Generalize or replace for step-2.
- `gen_transfer.py` — `restyle_to_references` (reface dep), `VIEW_SPEC`, `ADJ`. KEEP.
- `blender_convert.py` / `blender_project.py` — export + bake. KEEP.
- `mvadapter_*.py`, mvgpt helper cluster, `_run_projection`/`_run_gpt_projection` — REMOVE with
  their modes.
- `static/` — old vanilla-JS UI; retired by the new Next.js app.
- `outputs/` — generated artifacts (`{uid}_*.glb/png`). The new registry stores model metadata
  here too.

## Proposed new modules (Design will finalize)

- `webapp/models.py` — model registry: persistence (per-model metadata), CRUD, `Model` assembly,
  view→artifact mapping.
- `webapp/reference_views.py` — staged mesh-free reference generation + per-view prompts +
  dependency graph + tweak-prompt handling.

## Conventions

- Python: `snake_case` functions, `UPPER_SNAKE` env knobs read via `os.environ.get`, fail-fast on
  unknown variants. New behavior additive and reversible by config where it touches kept modes.
- HTTP responses match the frontend contract: camelCase JSON for the `Model`/`Job`/`ModelSummary`
  aggregates exactly as `lib/types.ts` declares; error bodies carry `detail` or `error`.
- View ids: `front, back, left, right, top, bottom, front-left, front-right, back-left,
  back-right` (frontend) ↔ backend angle tags `front/back/left/right/top/bottom/fl/fr/bl/br`.
- Albedo-only matte output preserved; gap-fill stays best-effort non-fatal.

## Spec artifacts

- `.batman/per-model-studio-backend/steering/` — understanding.md (approved), product/tech/
  structure/constitution.
- `.batman/per-model-studio-backend/spec/` — requirements.md, design.md, tasks.md.
