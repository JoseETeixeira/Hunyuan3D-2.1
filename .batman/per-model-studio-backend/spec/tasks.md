# Implementation Plan: Per-Model 3D Studio Backend

Traceability uses Requirement numbers from `spec/requirements.md` (R1–R13 + NFR). Files:
[webapp/studio.py](../../../webapp/studio.py) (new), [webapp/reference_views.py](../../../webapp/reference_views.py)
(new), [webapp/server.py](../../../webapp/server.py) (edit), and the Next.js app under
`Downloads/3-d-model-generation-workflow`. Tasks are ordered by dependency.

- [x] 1. Scaffold `studio.py` foundations: vocabularies, paths, persistence
  - Add `webapp/studio.py` with the closed vocabularies: `VIEWS` (10 ViewIds), `VIEW_TO_ANGLE`
    (`front-left→fl` … ), `TEXTURE_MODES={paint,reface}`, `FORMATS={glb,fbx,blend}`.
  - Implement path safety: reuse/define `assert_within(path, OUTPUT_DIR)` and `UUID_RE`; model dir =
    `OUTPUT_DIR/models/{id}/`; reject any caller string not matching the allowlists.
  - Implement `ModelStore`: `create/list/get/rename/delete`, `_load(id)`/`_save(id)` for
    `model.json` (atomic tmp+`os.replace`), and `assemble_model(id) -> dict` producing the exact
    frontend `Model` JSON (references/faces maps of 10, `seedImageUrl`, `meshUrl`, `texturedUrl`,
    `textureStage`, timestamps) with URL mapping per design (`/api/files/{id}_*.glb`,
    `/api/models/{id}/references/{view}/image`, `/api/models/{id}/seed`).
  - `summary(model) -> ModelSummary` (id, name, previewUrl, textured, updatedAt).
  - Unit tests: `model.json` round-trip, view↔angle mapping, `assert_within` rejects traversal,
    vocabulary validation.
  - _Requirements: 1, 4 (vocabulary/path safety), NFR-Security, NFR-Reliability_

- [x] 2. Model CRUD + asset-serving routes + router mount
  - In `studio.py` add `router = APIRouter()` with: `GET /api/models`, `POST /api/models`
    (multipart `name`, optional `seed_image` → store `models/{id}/seed.png`), `GET /api/models/{id}`,
    `PATCH /api/models/{id}` (`{name}`), `DELETE /api/models/{id}` (rm `models/{id}/` + `{id}_*.glb`).
  - Add `GET /api/models/{id}/seed` and `GET /api/models/{id}/references/{view}/image` (PNG, path-
    guarded, 404 if absent).
  - In `server.py`: `from webapp import studio` and `app.include_router(studio.router)` BEFORE the
    `app.mount("/", StaticFiles(...))` line so `/api/*` wins.
  - Integration tests (FastAPI `TestClient`): CRUD + rename + delete; **persistence across a reloaded
    `ModelStore`** (simulated restart); 404 on unknown id.
  - _Requirements: 1, 3 (seed storage), 13 (assets same-origin)_

- [x] 3. Unified `StudioJob` store + two execution lanes
  - In `studio.py` add `STUDIO_JOBS` dict + lock and helpers `new_job(label, model_id)`,
    `set_job(id, **kw)`, `complete_job(id, model_id)` (rebuild `Model`, embed in `job.model`),
    `fail_job(id, error)` (preserve artifacts). `GET /api/jobs/{id}` returns the public job shape
    (`id,status,progress,label,error,model,modelId`).
  - GPU lane: extend `server.py:_worker_loop` to handle work kinds
    `("studio_base"|"studio_reface"|"studio_face_edit", job_id)` → call `studio.run_gpu_job(kind, id)`;
    add `studio.submit_gpu(kind, id)` that `WORK.put((kind, id))`.
  - Network lane: in `studio.py` create a module-level `ThreadPoolExecutor` and
    `submit_net(fn, *args)` for reference-generation jobs; both lanes update `STUDIO_JOBS`.
  - Tests: async endpoints return `id`+numeric `progress` immediately; completed job embeds full
    `Model`; failed job sets `error` and preserves prior artifacts; progress is monotonic.
  - _Requirements: 2, 6 (job contract), 7 (separate lanes)_

- [x] 4. Reference generation module + reference endpoints (staged, gated)
  - Add `webapp/reference_views.py`: `VIEW_INPUTS` (mirror `lib/views.ts`), `build_prompt(view,
    edit_prompt)` (orthographic framing + rotation description e.g. "right = front rotated 90° CW,
    orthographic, no perspective" + 3D-cartoonish + "show only the faces visible in this view" +
    consistency), reusing `image_edit.CARTOON_STYLE/CONSISTENCY_RULE` and `gen_transfer.VIEW_SPEC/ADJ`;
    `generate_view(model_dir, view, dep_paths, seed_path, edit_prompt) -> png` via
    `image_edit.edit_image(images=[seed + approved deps in order], prompt, prefer='openai')`.
  - In `studio.py` add: `POST /api/models/{id}/references/{view}/generate` (`{edit_prompt?}`) — verify
    `VIEW_INPUTS[view]` deps are `approved` (else 409) and not already `generating`; set status
    `generating`; `submit_net(...)`; on success save `ref_{view}.png`, status `pending`, source
    `generated`, write `model.json`, complete job. `POST .../upload` (multipart `image` → store,
    `approved`, source `uploaded`, return Model). `POST .../approve` (400 if empty/generating else
    `approved`, return Model).
  - Unit tests: prompt contains required clauses per view; dependency resolution picks correct inputs
    (front←seed; cardinals←front+seed; corners←front/back+left+right+top). Integration: gating 409s;
    approve rejects empty/generating; upload auto-approves.
  - _Requirements: 3, 4, 5, 6_

- [x] 5. Texture base (GPU): shape + per-face paint over all approved refs
  - `studio.run_gpu_job("studio_base", id)`: validate all 10 refs `approved` (the endpoint returns
    400 otherwise); `TextureWorker.generate_shape(uid=id, image=approved front reference (fallback:
    seed image), **meshConfig)` →
    `{id}_shape.glb`; build a legacy-compatible job dict in `JOBS[id]` with `view_paths` = all approved
    refs mapped to angle tags + texture params, then call existing `_run_hyface(id)`; read resulting
    `{id}_textured.glb`.
  - Set `meshUrl`+`texturedUrl`, every painted face `done(mode:paint)`, `textureStage:"complete"`;
    persist `model.json` + `meshConfig`; `complete_job`.
  - `POST /api/models/{id}/texture/base` (JSON snake_case MeshConfig) — 400 unless all approved;
    `submit_gpu("studio_base", id)`; status `base-running` during.
  - Integration test (GPU/paint mockable): rejects unless all approved; on success both URLs set,
    faces `done(paint)`, stage `complete`; albedo-matte + gap-fill path preserved.
  - _Requirements: 7, 10, 13_

- [x] 6. Reface + per-face edit (GPU)
  - `studio.run_gpu_job("studio_reface"/"studio_face_edit", id)`: resolve the view's approved
    reference (or `image` override for edit); build a reface job dict (`reface_src_glb={id}_textured.glb`,
    `reface_face=angle`, `reference_paths=[approved ref]`) and call existing `_run_reface(id)`. For
    `mode=paint` (face edit), run the single-view Hunyuan paint at that face's camera and composite
    onto the base via the reface composite path (full visible-face mask, no depth band) — does NOT
    re-bake other faces.
  - `POST /api/models/{id}/texture/reface/{view}` (`{edit_prompt?}`) — 400 if no textured mesh; sets
    face `mode:reface`, stage stays `complete`. `POST /api/models/{id}/faces/{view}/edit` (multipart
    `mode`, `edit_prompt?`, `image?`) — dispatch reface|paint.
  - Integration test: reface rejects without textured mesh; sets face mode; stage stays complete;
    paint edit only changes the target face.
  - _Requirements: 8, 9, 10, 13_

- [x] 7. Download endpoint (glb/fbx/blend)
  - `GET /api/models/{id}/download/{fmt}` — validate `fmt` ∈ FORMATS (400 else); serve
    `{id}_textured.glb` (or `{id}_shape.glb`), converting via `blender_convert.py` for fbx/blend;
    404 if no model. Reuse the existing convert helper in `server.py`.
  - Integration test: glb served; fbx/blend route reachable (Blender mockable); bad fmt → 400.
  - _Requirements: 11_

- [x] 8. Remove non-target texture modes + repoint serving + health cleanup
  - In `server.py` delete handlers/dispatch for `projection` (`_run_projection`), `gptproject`
    (`_run_gpt_projection`), `mvadapter` (`_run_mvadapter`), `mvgpt` (`_run_mvgpt`/`_mv_texture` + its
    helper cluster), and the default `hunyuan` fall-through; default `texture_mode → "hyface"`. Delete
    `POST /api/jobs/{id}/resume`. Remove the `mvadapter_texture` import + `mvadapter` health flag.
  - Keep `image_edit.py`, `gen_transfer.py`, `blender_project.py`, `blender_convert.py`, the
    `gpt_angles` field, and all `TextureWorker` methods (used by hyface/reface).
  - Repoint the static mount from `webapp/static` to the Next export dir (`out/`); keep it mounted
    AFTER the API routers. Retire/leave `webapp/static` unused.
  - Tests: removed modes/endpoints → 404/400; `import webapp.server` + `GET /api/health` succeed
    (no broken imports); hyface/reface still produce albedo-matte GLBs.
  - _Requirements: 12, 13_

- [x] 9. Frontend wiring for same-origin static serve
  - In `Downloads/3-d-model-generation-workflow/next.config.mjs` add `output: 'export'` (keeps
    existing `images.unoptimized`). Set `NEXT_PUBLIC_USE_MOCK=false` (env) for the served build.
  - Build (`pnpm build`) → `out/`; point FastAPI's static mount at it (Task 8). Confirm relative
    `/api/*` fetches + `<model-viewer>` GLB loads resolve same-origin. Optionally gate
    `@vercel/analytics` off for self-hosting (harmless either way).
  - Manual check: load the app, create a model, the library/SWR list populates from the real API.
  - _Requirements: 13, NFR-Usability_

- [x] 10. Test suite consolidation + E2E
  - Add `webapp/test_studio_api.py` (FastAPI `TestClient`, image gen + GPU mocked) covering Tasks
    2–8 integration cases, and `webapp/test_reference_views.py` (prompt + dependency unit tests).
  - Add a gated E2E (real keys + GPU) script: create → 10 refs → base → reface → edit → download
    glb/fbx/blend with `NEXT_PUBLIC_USE_MOCK=false`; assert albedo-matte + gap-fill.
  - Run the existing `webapp/test_gapfill_logic.py` to confirm no regression.
  - _Requirements: all (verification), NFR-Reliability_

- [x] 11. Documentation + changelog
  - Update `WEBAPP_README.md` (new per-model API table + flow; drop removed modes), add a
    `webapp/CHANGELOG.md` (or root `CHANGELOG.md`) entry for this branch, and refresh the
    `Hunyuan3D-2.1/CLAUDE.md` batman:spec block to Implementation/complete.
  - _Requirements: documentation (Phase 8)_

## Verification

- **Unit/contract**: `python -m webapp.test_reference_views` and `python -m webapp.test_studio_api`
  (prompt + dependency + ModelStore + endpoint contracts with gen/GPU mocked). Both PASS.
- **Import/health**: `python -c "import webapp.server"` and `GET /api/health` return cleanly after
  mode removal.
- **Regression**: `python webapp/test_gapfill_logic.py`; confirm hyface/reface emit albedo-matte GLBs.
- **Persistence**: create a model, reload `ModelStore`, assert it lists/loads (covered in
  `test_studio_api.py`).
- **E2E (gated, GPU + keys)**: full create→refs→base→reface→edit→download run with
  `NEXT_PUBLIC_USE_MOCK=false`.
- **Serve**: `next build` (export) + `python -m webapp.server --port 8080 --preload`; open the app and
  drive one model end to end.
