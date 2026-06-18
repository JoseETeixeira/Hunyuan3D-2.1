# Understanding: Per-Model 3D Studio — Wire the New Next.js Frontend to the Backend

## User Goal

The user re-built the frontend into a clean, **per-model** studio (a Next.js app at
`C:/Users/josee/Downloads/3-d-model-generation-workflow`) and wants the **backend fully
wired** so the whole flow works end-to-end against real models, not the in-memory mock.

The intended product flow per model:

1. **Per-model structure** — a named model that owns its reference images, mesh, and
   texture, all reusable across runs without re-uploading. The user can rename the model.
2. **Staged reference generation** (when no references exist yet) — upload one seed image
   (any view), then generate each of the 10 reference views with **gpt-image-2**, following
   an imperative dependency graph and per-view prompts, with individual approve / edit
   (tweak-prompt) per view.
3. **Texture** — build the mesh + base texture with **Per-face AI paint** using the approved
   **front** reference, then **reface** each remaining face with its own approved reference.
4. **Per-face editing** — once step 3 is complete, edit any individual face via reface or
   per-face AI paint.

Also: **remove every texture mode that is not per-face AI paint or reface**, and give each
view generation a **proper prompt** (e.g. "right = front rotated 90° CW, orthographic,
3D cartoonish; each view shows only the faces visible in that view") plus the **reference
images it needs** and a **tweak-ask prompt** the user can send from the frontend.

## Task Slug

`per-model-studio-backend`

## Current Behavior

There are two backends in `Hunyuan3D-2.1/`. The relevant one is the rich **webapp**
(`webapp/server.py` + `webapp/pipeline.py`); the root `api_server.py`/`model_worker.py` is
the thin Tencent demo port and is out of scope. The new frontend talks to the webapp's
`/api/*` surface.

### Workflow Summary

- **Server**: a single FastAPI app (`webapp/server.py:1428`). One in-memory `JOBS` dict
  (`server.py:352`, `JOBS_LOCK`) and one `WORK` queue (`server.py:354`) drained by a single
  daemon **GPU worker thread** `_worker_loop` (`server.py:1204`, started `server.py:1902`).
  Jobs run strictly sequentially — no concurrency, no priorities, no cancellation.
- **Job lifecycle**: `POST /api/generate` (multipart image(s) + params) → shape job →
  `_run_shape` → `{uid}_shape.glb`, status `shape_ready` → (auto or via
  `POST /api/jobs/{id}/texture`) texture job → `_run_texture` dispatch → `{uid}_textured.glb`,
  status `completed`. Status flow: `queued → processing_shape → shape_ready →
  queued_texture → processing_texture → completed | failed`.
- **Shape** is always Hunyuan `hy3dshape` (`Hunyuan3DDiTFlowMatchingPipeline`) via
  `TextureWorker.generate_shape` (`pipeline.py:276`). **TRELLIS.2** (sibling dir) is **not
  imported anywhere** in the webapp — dormant.
- **Texture modes** (`_run_texture`, `server.py:1139`) dispatch on `job["texture_mode"]`:
  `hunyuan` (default PBR fall-through, `server.py:1154`), `projection` (`476`),
  `gptproject` (`532`), `mvadapter` (`801`), `mvgpt` (`805`), **`hyface`** (`848`),
  **`reface`** (`1049`).
- **`hyface` = "Per-face AI paint"**: paints each face individually with one-view Hunyuan
  paint conditioned on that face's own reference, then bakes all faces into one shared UV
  texture (cosine-blend + inpaint). Front reference priority: explicit Front slot →
  shape-gen processed image → `source0`. Empty faces are **gpt-image-2-synthesized** from the
  face geometry + other refs. Corners `fl/fr/bl/br` supported. Albedo-only matte output.
  Core call `TextureWorker.paint_faces(uid, shape_glb, view_specs=[(PIL, elev, azim, weight)],
  ...)` (`pipeline.py:490`).
- **`reface` = depth-aware single-face re-texture** of an already-textured mesh: only the
  nearest depth band is repainted, the rest keeps its texture. Generates the face via
  `render_textured_view` + `restyle_to_references` (gpt-image-2/Gemini), geometry-locks it,
  bakes via `TextureWorker.reface(uid, textured_glb, elev, azim, view_image, depth_band, ...)`
  (`pipeline.py:577`). Cameras from `PROJECTION_CAMS` (cardinals) + `HYFACE_CORNER_CAMS`
  (corners).
- **Coverage gap-fill** (`fill_coverage_gaps`, `pipeline.py:706`) auto-paints oblique/recessed
  texels no fixed camera covers; runs after both hyface and reface, default-on, best-effort.
- **Reference channels** (two, different per mode): named per-side file fields
  `front/back/left/right/top/bottom/fl/fr/bl/br` → `{uid}_view_{angle}.png` →
  `job["view_paths"]` (used by **hyface**); and a parallel `reference[]` + `reference_side[]`
  list → `{uid}_reference{idx}.png` (used by gpt/mvgpt/mvadapter and **reface**).
- **Image generation** all funnels through `webapp/image_edit.py:edit_image(images, prompt,
  size, mask, prefer)` — tries OpenAI `images.edit` (model `OPENAI_IMAGE_MODEL`, default
  `gpt-image-2`) then Gemini (`GEMINI_IMAGE_MODEL`, default `gemini-3-pro-image`). `images[0]`
  is the structure to preserve, `images[1:]` are references; order is load-bearing
  ("Image 1", "Image 2"…).
- **Downloads**: `GET /api/jobs/{id}/download/{fmt}` serves `glb`, and converts to `fbx`/
  `blend` via Blender.

### Why This Evidence Answers The Question

- `webapp/server.py` `_run_texture` + `JOBS` dict + `view_paths`/`reference_paths`: this is
  the **only** place the texture modes, the job contract, and the reference routing live — it
  defines exactly which primitives the new per-model API can call and how uploads must be
  routed per mode. It answers "what already exists vs what must be built."
- `webapp/pipeline.py` `TextureWorker`: the concrete, **angle-generic** texturing primitives
  (`generate_shape`, `paint_faces`, `reface`, `fill_coverage_gaps`, `render_*`). It answers
  "can we implement the per-model flow by orchestration alone?" — yes for texturing.
- `webapp/elevations.py:generate_elevations` + `webapp/image_edit.py:edit_image`: the only
  **mesh-free** seed→multi-view image synthesis in the repo. It answers "how do we generate
  reference views in step 2 before any mesh exists?" — reuse `edit_image` with new per-view
  prompts; `generate_elevations` is the closest existing pattern but is building-specific and
  lacks bottom + the 4 corners.
- New frontend `lib/api.ts` header + `lib/types.ts` + `lib/mock-backend.ts`: the **exact
  target contract** (endpoints, request/response shapes, the `Model` state machine,
  `textureStage` transitions, `job.model` on completion). It answers "what must the backend
  return."
- `lib/views.ts:VIEW_INPUTS` / `STAGE_VIEWS`: the imperative dependency graph the backend
  must honor for staged generation.

### Process Distinctions And Terminology

- **`hyface` (per-face AI paint) vs `reface`**: hyface bakes a *fresh* full UV texture from
  per-face references on an **untextured** mesh (fresh UVs via `mesh_uv_wrap`); reface edits
  *one face* of an **already-textured** mesh, preserving existing UVs and only repainting the
  nearest depth band. The new frontend's `"paint"` mode → hyface; `"reface"` → reface.
- **`/texture/base` vs `/texture/reface/:view`** (USER-CONFIRMED 2026-06-17, overrides the
  mock): base does shape gen **and** a **full per-face AI paint over ALL approved references**
  (hyface `view_specs` = every approved view, not front-only). Afterward, a **reface pass over
  all approved references** refines each face individually (depth-aware single-face reface).
  This contradicts the frontend mock, where `textureBase` marks only `faces.front = {paint,
  done}` and leaves the others pending — the `Model`/`textureStage`/`faces` state machine must
  be reconciled in Requirements/Design (see Open Questions).
- **Named per-side fields vs `reference[]`+`reference_side[]`**: two distinct upload channels
  read by *different* modes. A naïve rewire that posts to the wrong channel silently drops
  references.
- **`view_paths` (hyface input refs) vs `{uid}_hyfaceref_*.png` (server-synthesized output
  refs)**: the `hyfaceref_*` names are *outputs* the server writes for gpt-synthesized missing
  faces, not request fields.
- **Reference-view generation, mesh-free vs mesh-conditioned**: `generate_elevations` is
  mesh-free; `gen_view_paths`/`transfer`/`restyle_to_references`/`gpt_colorize`/`gpt_depthmask`
  all require a grey geometry render or a textured-mesh render and therefore **cannot** run in
  step 2 before a mesh exists.

### Components Likely To Change And Why They Exist

- `webapp/server.py` — the HTTP + job + dispatch layer. The new per-model REST surface
  (`/api/models/*`, model persistence, staged reference generation, `job.model` on
  completion) must be added here (or in a new module mounted on the same app), and the
  removable modes deleted from dispatch/endpoints/UI.
- `webapp/pipeline.py:TextureWorker` — reused as-is for texturing; likely no change beyond
  possibly exposing helpers. `paint_faces` and `reface` already accept arbitrary references
  and cameras.
- `webapp/elevations.py` + `webapp/image_edit.py` — the seed→view generation seam. Step 2
  needs a new generic per-view prompt set (orthographic, cartoonish, "only faces in this
  view", rotation descriptions) plus bottom + the 4 corners and the staged feeding
  (front←seed; cardinals←front+seed; corners←front/back+left/right/top), built on `edit_image`.
- `webapp/gen_transfer.py` — keep (reface depends on `restyle_to_references`; only depends on
  `image_edit`). Used by mvgpt too, so do not delete when removing mvgpt — only the call sites.
- `webapp/static/*` — the **old** vanilla-JS UI; the new Next.js app replaces it. The old
  UI's exact wire contract (bare-angle file parts, parallel `reference`/`reference_side`,
  reface needs no fresh image) is the regression checklist.
- `webapp/mvadapter_runner.py`, `webapp/mvadapter_texture.py`, mvgpt helper cluster,
  `_run_projection`/`_run_gpt_projection` — removable with their modes.

### Execution Locations

- **Shape + per-face paint + reface bakes** run on the **single GPU worker thread** (CUDA,
  16 GB target). They must be serialized.
- **gpt-image-2 / Gemini reference generation** is **network/CPU-bound, not GPU**. Today it
  runs inline inside GPU job handlers (e.g. hyface synth, reface restyle). For step 2 (pure
  seed→views, no mesh) it does not need the GPU at all — an open design question is whether
  staged reference generation should block the GPU queue or run on a separate async path.
- **Persistence** today is just flat files in `OUTPUT_DIR` (`webapp/outputs`) keyed by a
  per-run uuid; `/api/gallery` rebuilds a list by globbing `*_shape.glb`/`*_textured.glb`.
  There is **no model identity, no names, no params, no DB** — this is the central gap.

## Likely Change Surface

### Files And Symbols

- [webapp/server.py](webapp/server.py) — add the per-model REST layer + model registry;
  map `/texture/base`→hyface(front), `/texture/reface/:view`→reface, `/faces/:view/edit`→
  paint|reface; make `GET /api/jobs/:id` (or a new jobs map) return the full `Model` on
  completion; remove `_run_projection` (`476`), `_run_gpt_projection` (`532`),
  `_run_mvadapter` (`801`), `_run_mvgpt`/`_mv_texture` (`805`/`583`), default `hunyuan`
  fall-through (`1154`), `POST /api/jobs/{id}/resume` (`1835`, mvgpt-only), and the health
  `mvadapter` flag (`1448`/`1461`). Keep `gpt_angles` field (hyface reuses it, `888`).
- [webapp/pipeline.py](webapp/pipeline.py) — `TextureWorker.generate_shape` (`276`),
  `paint_faces` (`490`), `reface` (`577`), `fill_coverage_gaps` (`706`),
  `PROJECTION_CAMS` (`374`); reused, likely unchanged.
- [webapp/elevations.py](webapp/elevations.py) — `generate_elevations` (`51`) +
  `_elev_prompt` (`33`): generalize / replace prompts for generic assets; add bottom + corners
  + staged feeding. Or add a new `webapp/reference_views.py` module.
- [webapp/image_edit.py](webapp/image_edit.py) — `edit_image` (`83`): the call to reuse for
  step-2 generation; `CARTOON_STYLE`/`CONSISTENCY_RULE` prompt fragments.
- [webapp/gen_transfer.py](webapp/gen_transfer.py) — `restyle_to_references` (`174`, keep),
  `VIEW_SPEC` (`25`), `ADJ` (`63`): reusable per-view descriptions + face-visibility map.
- [webapp/server.py:816](webapp/server.py) — `HYFACE_CORNER_CAMS`: corner view→(elev,azim).
- New frontend [lib/api.ts](../../../../../Downloads/3-d-model-generation-workflow/lib/api.ts)
  contract; flip `NEXT_PUBLIC_USE_MOCK=false` once endpoints exist.

### Tests

- `webapp/test_gapfill_logic.py` (existing pure unit test for gap-fill selection). No
  HTTP/endpoint tests exist for the webapp. New per-model API needs: endpoint/contract tests
  (model CRUD, staged-generation gating, job→model on completion), and the dependency-graph
  honoring. Frontend has no tests.

### Configuration And Infrastructure

- Env: `OPENAI_API_KEY` / `GEMINI_API_KEY` (gate all image generation), `OPENAI_IMAGE_MODEL`
  (default `gpt-image-2`), `GEMINI_IMAGE_MODEL` (default `gemini-3-pro-image`),
  `HY3D_OUTPUT_DIR`, `BLENDER_BIN`, `HYFACE_*`, `REFACE_DEPTH_BAND`, `GAPFILL_*`. Removable-mode
  envs (`MVADAPTER_*`, `MVGPT_*`) can go with their modes.
- Deployment seam: the new Next.js app issues **relative** `/api/*` fetches → it must be served
  same-origin as the API (FastAPI serving the built frontend, or a reverse proxy / Next
  rewrites). Open question.
- Single 16 GB GPU; sequential worker; `docker-compose.yml` runs `python -m webapp.server`.

### Documentation

- `webapp/CHANGELOG.md` (per-branch entries), `WEBAPP_README.md` (HTTP API table + flow),
  `Hunyuan3D-2.1/CLAUDE.md` (batman spec sync). All need updates when behavior changes.

## Evidence

- `server.py:352/1139/848/1049/1465/1568/1654/1835` — in-memory JOBS, `_run_texture` dispatch,
  hyface/reface handlers, generate/texture/gallery/resume endpoints; **no model persistence**.
- `pipeline.py:276/490/577/706` — `generate_shape`, `paint_faces`, `reface`,
  `fill_coverage_gaps` signatures (angle-generic, reusable for orchestration).
- `elevations.py:51` + `image_edit.py:83` — only mesh-free seed→views path; `edit_image` is the
  shared OpenAI→Gemini wrapper. `generate_elevations` skips bottom, produces no corners, and has
  building-specific prompts (key gap for generic assets).
- `gen_transfer.py:82/131/174` + `gpt_colorize.py` + `gpt_depthmask.py` — all mesh-conditioned;
  cannot generate step-2 references before a mesh exists.
- New frontend `lib/api.ts:1-54` (endpoint contract), `lib/types.ts` (`Model`, `textureStage`,
  `FaceMode="paint"|"reface"`), `lib/views.ts:62-73` (`VIEW_INPUTS`), `lib/use-job.ts:27-29`
  (**completed job must carry full `model`**), `lib/mock-backend.ts` (reference state machine).
- `static/app.js:275-326` — old wire contract: bare-angle file parts, parallel
  `reference`/`reference_side`, `mask`; reface needs no fresh image.
- `.batman/per-face-paint-mode` + `.batman/oblique-angle-coverage` specs — hyface/reface design
  intent, the locked constitution principles (additive, albedo-matte, reuse primitives, bounded
  cost, env-tunable, best-effort non-fatal), and the user's standing requirements (cartoonish
  flat albedo; each view shows only its faces; oblique coverage; per-face independence).

## Visual Recap

- Path: `.batman/per-model-studio-backend/steering/understanding.html` (to be generated with
  `visual-explainer` project-recap before user validation).
- Notes: will show the current webapp architecture, the new per-model contract, the gap map
  (persistence, step-2 mesh-free generation, mode removal), and the per-view dependency graph.

## Resolved Decisions (user-confirmed 2026-06-17)

- **Base flow**: `/texture/base` = `generate_shape` + hyface over **ALL approved references**
  (full per-face paint). The base texture is the final usable result; `textureStage` reaches
  **`complete`** right after base. Not front-only.
- **Reface is OPTIONAL** (user-confirmed 2026-06-18): per-face reface/paint are optional
  refinements available after base, not required to finish a model.
- **Reface reference source**: each face refaces **automatically using its own approved step-2
  reference** (no re-upload); the tweak/edit prompt refines it.
- **Mode removal**: **fully remove all 5** non-target modes (hunyuan, projection, gptproject,
  mvadapter, mvgpt) from dispatch + endpoints + old static UI; new default mode = `hyface`.
  Keep shared helpers (`gen_transfer.py`, `blender_project.py`, `image_edit.py`, the
  `gpt_angles` field).

## Open Questions

1. **Persistence mechanism** for the per-model registry (name → references + mesh + texture,
   reusable across runs): per-model JSON sidecar + index file (recommended), SQLite, or extend
   the gallery glob. None exists today.
2. **Where step-2 reference generation runs**: through the single GPU queue (blocks GPU) or a
   separate async/network path (recommended — gpt-image is not GPU work).
3. **Deployment**: how the Next.js app is served same-origin as `/api/*` (FastAPI static serve
   vs proxy vs Next rewrites).
4. **fbx/blend downloads** require Blender present at runtime; confirm it is installed in the
   target environment.

(Resolved understanding-level items: `meshUrl` and `texturedUrl` must both be populated at base
— the webapp already produces `{uid}_shape.glb` and `{uid}_textured.glb`, so both map directly.)

## Risks And Constraints

- **`job.model` contract**: the frontend only refreshes `activeModel` when a completed job
  returns the full `Model` (`use-job.ts:27-29`). The new jobs must embed the model snapshot, and
  the POST response must already carry `id` + numeric `progress`.
- **No model identity today**: building it is net-new; the `{uid}` disk prefix is the only stable
  key to map onto.
- **Two reference channels**: route uploads to the correct one per mode or refs are silently
  dropped.
- **Single sequential GPU worker**: 9 sequential reface jobs is the expected per-model load;
  long jobs block everything; no cancellation.
- **gpt-image fidelity**: orthographic / "only this view's faces" is prompt-enforced only; gpt
  drifts/reframes (~12%) and sometimes horizontally mirrors side views; back/bottom from a single
  seed are the weakest/most hallucinated. Need consistency feeding + possibly mirror checks.
- **API keys / model ids**: every generation needs `OPENAI_API_KEY` and/or `GEMINI_API_KEY` and
  the defaulted model ids must exist on the account.
- **Don't regress** the locked constitution principles (albedo-matte output, reuse bake
  primitives, best-effort non-fatal gap-fill, env-tunable, bounded cost).
- **Mode-removal pitfalls**: `gpt_angles` field is shared with hyface; health imports mvadapter
  `is_available`; resume hardcodes mvgpt; `gen_transfer.py`/`blender_project.py` are shared —
  remove call sites, not shared modules.

## Architecture Change Assessment

- Status: **required**
- Reason: introduces a **new persistence layer** (per-model registry + metadata), a **new
  higher-level public REST surface** (`/api/models/*` with a `Model` aggregate and job→model
  contract), a **new pre-mesh reference-generation capability** (mesh-free staged gpt-image
  generation), a **new execution decision** (network-bound generation vs the single GPU queue),
  a **deployment/origin change** (serving the Next.js app against `/api/*`), and **removes
  texture modes** from the dispatch/endpoints/UI. These alter how major parts interact, so the
  Design phase must present options with pros/cons and get explicit validation before
  implementation.
- Areas affected: `webapp/server.py` (new routes + registry + dispatch trim), possibly a new
  `webapp/models.py`/`webapp/reference_views.py`, `webapp/elevations.py`/`image_edit.py` (step-2
  prompts), persistence files under `OUTPUT_DIR`, removal of `mvadapter_*`/mvgpt/projection/
  gptproject code, `webapp/static/*` retirement, deployment config (compose / static serve),
  `WEBAPP_README.md` + `CHANGELOG.md` + tests.

## Initial Verification Ideas

- Contract tests for the per-model API: model CRUD + rename + persistence-across-restart;
  staged-generation gating honors `VIEW_INPUTS`; upload auto-approves; approve rejects
  empty/generating; completed jobs embed the full `Model`.
- Reference-generation smoke test (mesh-free) producing all 10 views from one seed with
  `OPENAI_API_KEY`, asserting per-view prompts and dependency feeding.
- End-to-end: create model → generate+approve 10 refs → `/texture/base` (shape + front paint,
  both `meshUrl` and `texturedUrl` populated) → reface 9 faces → `complete` → per-face edit →
  download glb/fbx/blend.
- Regression: hyface and reface still produce albedo-matte GLBs after mode removal; gap-fill
  still runs; removed endpoints return 404; no import breakage in `/api/health`.
- Run the new frontend with `NEXT_PUBLIC_USE_MOCK=false` against the live API.
