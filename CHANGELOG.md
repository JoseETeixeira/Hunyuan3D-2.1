# Changelog

## Unreleased

### Removed
- **Dropped the unused MV-Adapter / SDXL model fetch and its dead code.** The texture pipeline only
  runs `hyface` + `reface` (see `_run_texture` in `webapp/server.py`); the MV-Adapter SDXL mode was
  removed long ago but its model fetch and runner were still in the tree. Deleted
  `webapp/setup_mvadapter.sh` (fetched SDXL base ~6.9G, the ig2mv adapter ~3.6G, `RealESRGAN_x2plus`,
  `big-lama`, and BiRefNet) plus `webapp/mvadapter_runner.py`, `webapp/mvadapter_texture.py`, and
  `webapp/sdxl_geomatch.py`. Removed the now-orphaned `_unload_worker` helper (only freed VRAM for the
  MV-Adapter subprocess) and the vestigial `mv_viewset` form param / job field from `/generate`,
  `/api/jobs/{id}/texture`, `/api/retexture` (`webapp/server.py`) and the studio base job
  (`webapp/studio.py`). Pruned the `mvadapter` volume and `MVADAPTER_*` env from `docker-compose.yml`.
  Kept the still-used `RealESRGAN_x4plus` (Hunyuan paint super-res) and UniRig (rigging). No live path
  changed.

### Fixed
- **Hand-paint "AI fix" can no longer misplace elements — it only recolours now.** The fix sends the
  rendered face to a generative model (Gemini / gpt-image); even with a prompt that forbids it, the
  model regenerates pixels and drifts (shifts, redraws or reframes elements). Prompt text can't hard-lock
  geometry. New `image_edit.recolor_preserve_structure` rebuilds the result in CIELAB: lightness **L**
  (every element's position, edges, shapes and detail) is taken from the original render, and only the
  colour channels **a/b** come from the AI output — edge-aware aligned to the render with a guided filter
  so colours snap to the original's edges instead of bleeding. `_gpu_handpaint_ai` (`webapp/studio.py`)
  applies it to the AI result before baking, so AI fix can recolour but never reposition. Trade-off: it
  no longer repairs *structural* artefacts (e.g. luminance seams), by design. Toggle off with
  `HANDPAINT_AI_RECOLOR_ONLY=0` for the raw generative output. Files: `webapp/image_edit.py`,
  `webapp/studio.py`.
- **Studio bakes now embed the diffuse texture losslessly (PNG), not JPEG — the real cause of the
  per-bake quality loss and blocky "artifacts".** `MeshRender.save_mesh` writes the diffuse map through
  `mesh_utils._save_texture_map`, which hardcodes `.jpg` (`cv2.imwrite`, ~q95). Every studio bake reloads
  its own `_textured.glb` and re-bakes, so the whole atlas was re-encoded as JPEG on each edit —
  compounding ringing/blockiness and softening even on faces that weren't touched. New `_export_matte_glb`
  helper exports the GLB from the saved OBJ geometry but swaps in the pixel-exact in-memory texture as a
  lossless PNG (`_force_matte` gained an `override_texture` arg; `_lossless_png` pins the PIL format to
  PNG so trimesh embeds it losslessly). Routed through it: `paint_overlay` + `reface` (composite, native
  res) and the base bakes `hyface` / projection (`webapp/pipeline.py`), keeping each path's existing
  output resolution (`downsample=True` halves the override to match). Trade-off: GLBs are larger (PNG vs
  JPEG); quality is the point. Note: existing models still carry whatever JPEG damage they were baked
  with — re-generate the base to get a clean lossless start.
- **Hand-paint / reface bakes no longer soften the rest of the texture.** Every composite bake reloaded
  the stored texture and `set_texture` resized it up to the renderer's `texture_size` (e.g. 2048→4096),
  then `save_mesh(downsample=True)` halved the whole atlas again on export (4096→2048). That round-trip
  ran over EVERY texel — including the ones not being edited — so each hand-paint or reface lost a little
  sharpness/detail across the entire model. `paint_overlay` and `reface` (`webapp/pipeline.py`) now
  composite the freshly-baked texels over the existing texture at **its native resolution** via the new
  `_composite_paint_over_base` helper: only the painted texels change, every other texel stays
  pixel-exact (no global resize, no half-resolution downsample). Output resolution is unchanged; only the
  cumulative degradation is gone. `_force_matte` keeps only the diffuse map, so MR/normal are unaffected.
- **Hand-paint / reface bakes no longer leave dark UV-seam "vertex lines".** Keeping the atlas sharp (above)
  exposed a latent artifact: the thin gutter texels between UV islands had too small a margin, so the
  renderer's bilinear/mipmap sampling bled them in as dark lines tracing the triangulation. The old
  whole-atlas resample had been widening that margin incidentally. `_composite_paint_over_base` now re-pads
  the gutters with the renderer's own `uv_inpaint` (the same edge-padding the full base bake uses) via
  `_dilate_uv_gutters`: it fills ONLY the gutter texels (UV-coverage from `render.texture_indices`) and
  re-asserts island texels afterward so painted/untouched pixels stay bit-exact. Best-effort — if the
  inpaint fails the composite is exported unchanged.
- **Custom hand-paint ("Paint this angle"): the backdrop now matches the live 3D view exactly.** The
  capture sent only the orbit angle (`elev`/`azim`), but the backend rendered (and baked) with a fixed
  **orthographic** camera (`ortho_scale=1.2`, fixed distance, origin pivot) while `model-viewer` is
  **perspective** — so zoom, field of view and pan were dropped and the backdrop framed the whole
  object ("very close but not exact"). The capture now also reads `getFieldOfView()` (vertical fov),
  `getCameraOrbit().radius` (zoom) and `getCameraTarget()` (pan), and the backend renders + bakes
  through a matching **perspective** camera: fov → vertical perspective fov, radius → camera distance,
  target → pivot, and the live viewport **aspect ratio** → the backdrop's aspect (so a wide 3D viewer
  renders a wide backdrop, not a square centre-crop — the original square render only reproduced the
  narrow centre of a wide view, which read as a flatter/zoomed angle). radius/target are mapped into the
  renderer's normalized frame (`set_mesh` axis remap `R(P)=(-x,z,-y)` then `(P−C)·s`); the backdrop +
  paint canvas + bake all follow the captured aspect. The render and bake share the camera so strokes
  land where painted. Scoped strictly to the `custom` path: the 10 canonical face renders/bakes pass
  none of the new params and stay square/orthographic/byte-identical. `get_mv_matrix` now orbits its
  `center` arg (a no-op for the `center=None` every existing caller passes).
  `webapp/studio-ui/components/studio/model-3d-viewer.tsx`, `hand-paint-canvas.tsx`, `lib/api.ts`,
  `types/model-viewer.d.ts`, `webapp/studio.py`, `webapp/pipeline.py`,
  `hy3dpaint/DifferentiableRenderer/camera_utils.py`.
- **Hand paint / AI fix: grazing (near edge-on) surfaces now bake.** Strokes on steeply-angled but
  clearly-visible faces — e.g. a car's rear quarter at a 3/4 custom camera — silently failed to bake:
  `back_project`'s cosine gate (`bake_angle_thres=75°`) zeros any texel whose normal sits >75° off the
  camera. Since the user paints directly on the render, every rasterized pixel should be paintable, so
  `pipeline.py:paint_overlay` now widens the gate for the overlay bake (default 85°, env
  `PAINT_OVERLAY_ANGLE_THRES`, restored after — `render` is the shared pipeline renderer). `webapp/pipeline.py`.

### Added
- **AI fix on a custom view now anchors to the closest canonical face's reference.** Previously AI fix
  only passed a style reference for the 10 canonical faces; a free-camera ("Paint this angle") view had
  none, so Gemini had nothing telling it what the face *should* look like. It now picks the canonical
  view whose camera direction is nearest the custom (elev, azim) (`_closest_canonical_view`) and passes
  that view's approved reference as a look-only anchor (letterboxed to the output aspect via `_contain`
  so a square reference isn't stretched into a wide frame). The existing prompt already uses references
  only to resolve garbled areas and keep colours/identity — it never changes the captured image's view,
  framing or aspect. `webapp/studio.py`.
- **Hand paint — "AI fix" a captured view with Gemini.** The hand-paint surface gets an **AI fix**
  button: it flattens the captured face render plus any strokes and sends it to Gemini with a prompt
  that keeps the existing style + base colours and repairs only inconsistencies (seams, projection
  smears, stretched/blurry patches, colour bleed, artefacts) — the cleaned image bakes onto the face
  through the existing overlay path. The prompt reuses `image_edit.CARTOON_STYLE`/`CONSISTENCY_RULE`
  plus a new `HANDPAINT_FIX_PROMPT`, and runs with `prefer="gemini"` (layout-preserving so the bake
  stays aligned; falls back to gpt-image if no Gemini key). Works for canonical faces and the custom
  free-camera view; an optional `edit_prompt` steers a specific touch-up. New endpoint
  `POST /api/models/:id/faces/:view/handpaint-ai` (job kind `studio_handpaint_ai` → `_gpu_handpaint_ai`).
  Files: `webapp/image_edit.py`, `webapp/studio.py`, `webapp/studio-ui/lib/api.ts`,
  `components/studio/{hand-paint-canvas,texture-panel,model-3d-viewer}.tsx`, built → `webapp/webui/`;
  `webapp/test_studio_api.py` (`test_handpaint_ai_endpoint`).
- **GPU-aware image build + guaranteed first-run model download.** New `build.ps1` (Windows) and
  `build.sh` (Linux/WSL) wrappers read the host GPU's CUDA compute capability from
  `nvidia-smi --query-gpu=compute_cap` (name→arch fallback for old drivers; multi-GPU unioned) and
  build with the native CUDA extensions (`custom_rasterizer`, mesh painter) compiled for exactly that
  arch — e.g. `8.6` on a 3080 Ti, `12.0` on a 5080 — instead of the hardcoded `12.0`. The compose
  `build.args` wires `TORCH_CUDA_ARCH_LIST` (default `12.0`). On first container start a new entrypoint
  (`docker/entrypoint.sh` → `webapp/prefetch_models.py`) prefetches the `tencent/Hunyuan3D-2.1` weights
  (shape + paint) into the `hf-cache` volume; idempotent, so later starts only do a cheap etag check.
  The baked `Dockerfile` CMD now `--preload`s too. Files: `build.ps1`, `build.sh`,
  `docker/entrypoint.sh`, `webapp/prefetch_models.py`, `Dockerfile`, `docker-compose.yml`,
  `WEBAPP_README.md`.
- **An already-textured `.blend` imports as a textured model — paint it without re-texturing.** On
  `.blend` import the backend now detects an embedded base-color texture (`_glb_has_texture`); if
  present, the converted GLB is registered as the textured model (copied to `{id}_textured.glb`, stage
  `complete`, faces `done`) so the Textured tab, "Paint this angle", reface, and textured export work
  immediately — no "Generate textures"/regenerate step. Untextured `.blend`s still import as a fresh
  base. `webapp/studio.py` (`_gpu_mesh_upload`).
- **Uploaded `.blend` goes straight to texturing.** After a `.blend` import, the workflow now jumps to
  the Mesh & textures step and the uploaded mesh is texturable directly — "Generate textures" reuses
  the uploaded shape (no need to "Generate mesh" first; `_gpu_base` already reuses an existing shape
  GLB). `studio-provider.tsx` (`notifyMeshUploaded`), `model-3d-viewer.tsx`, `workflow-panel.tsx`.
- **Mesh versions — restore a model before a remesh or .blend import.** Both ops were destructive
  (overwrote the shape GLB, dropped the texture, cleared the texture timeline) with no way back. Now,
  before each remesh / `.blend` import, the backend snapshots the full mesh state (shape + textured +
  rigged GLBs + the texture timeline + face/stage/rig/config) under `mesh_history/{seq}/`, keeping the
  last 5. A new "Mesh versions" list in the Texture panel restores any of them instantly via
  `POST /api/models/:id/mesh/restore/:seq`. `webapp/studio.py` (`_push_mesh_snapshot`,
  `_restore_mesh_snapshot`, `meshHistory`/`meshSeq`), `webapp/studio-ui/lib/{types,api,mock-backend}.ts`,
  `components/studio/texture-panel.tsx`.
- **Step 3 — AI rigging (UniRig) with positionable joint markers.** A third workflow step rigs the
  model mesh with [UniRig](https://github.com/VAST-AI-Research/UniRig) (skeleton → skin → merge), run
  as a subprocess in its own env on the GPU worker lane (like Blender). It surfaces 12 named joints
  (groin, chin, L/R shoulder, elbow, hand, knee, ankle) as markers in the 3D viewer; select a joint
  and click the model to place it at the limb's **center** (a trimesh ray through the mesh →
  entry/exit midpoint). "Apply rig changes" edits the skeleton to the new joints and re-runs UniRig
  skin + merge. The rigged GLB carries armature + skin, and Export now serves it as GLB / FBX /
  .blend (FBX/.blend via the existing Blender convert).
  - Backend: `webapp/rig_pipeline.py` (orchestration + joint→marker mapping + ray-center),
    `webapp/server.py` (`_unirig_run`, `_blender_python`, `unirig` health flag, `UNIRIG_DIR` /
    `UNIRIG_PYTHON` / `UNIRIG_BASH` env), `webapp/blender_dump_skeleton.py`,
    `webapp/blender_edit_skeleton.py`, `webapp/studio.py` (`rig` persistence, `_gpu_rig` /
    `_gpu_reskin`, routes `POST /rig`, `/rig/apply`, `/rig/marker/{joint}`, rigged-GLB download
    preference).
  - Frontend: `components/studio/rig-panel.tsx`, marker hotspots + click-to-place in
    `model-3d-viewer.tsx`, 3rd step in `workflow-panel.tsx`, shared `rigJoint`/`rigActive` in
    `studio-provider.tsx`, `RigState` in `lib/types.ts`, `lib/api.ts`, `lib/mock-backend.ts`.
  - Container wiring (mirrors MV-Adapter): `webapp/setup_unirig.sh` clones UniRig into an isolated
    conda env on a persisted `unirig` volume; `docker-compose.yml` mounts it and sets `UNIRIG_DIR` +
    `UNIRIG_PYTHON`. One-time: `docker compose exec hunyuan3d-studio bash webapp/setup_unirig.sh`.
    Weights auto-download from HF `VAST-AI/UniRig` on first rig job.
  - Joint→marker mapping is **geometry/topology-based**, not name-based: UniRig emits generic
    `bone_{i}` names (its docs don't define anatomical names), so markers are inferred from the
    skeleton tree + positions (chin = highest joint; ankles/hands = lowest/most-lateral leaves;
    groin = LCA of the ankles; knees/elbows = mid-chain; shoulders = arm roots). Semantic bone names,
    if present, are used first. Skeleton dump/edit convert Blender Z-up ↔ glTF Y-up so marker coords
    match the viewer + the trimesh recenter ray.
  - Runtime-validation items (untestable without a GPU host): the geometry mapping heuristics +
    left/right-by-X convention against real UniRig skeletons, the setup-script wheel pins
    (spconv/flash-attn/PyG for cu128/sm_120), and the rigged-GLB ↔ skeleton-FBX axis assumption used
    when re-skinning edited joints.

- **Hand-paint brush preview + undo.** The hand-paint surface now draws a brush-sized ring that
  tracks the cursor (tinted to the active color, white dashed for the eraser) on a dedicated overlay
  canvas, so it scales correctly under zoom/pan. `Ctrl+Z` / `Cmd+Z` undoes the last stroke (snapshot
  taken before each stroke, capped at 30; clearing or loading a new face resets the history).
  `webapp/studio-ui/components/studio/hand-paint-canvas.tsx`.
- **Upload a `.blend` to replace a model's mesh.** New `POST /api/models/:id/mesh/upload` (multipart
  `mesh`) converts an uploaded `.blend` to the shape GLB via headless Blender
  (`webapp/blender_blend_to_glb.py`) and adopts it as a new untextured base: references + seed are
  kept, the current texture is reset (faces → pending, `textureStage` → none, history cleared) so it
  can be re-textured on the new geometry. Surfaced as an "Upload .blend" button on the 3D viewer
  toolbar (always available once a model exists, every stage). `webapp/studio.py`, `webapp/server.py`
  (`_blender_blend_to_glb`), `webapp/studio-ui/lib/api.ts` (`uploadMesh`),
  `webapp/studio-ui/components/studio/model-3d-viewer.tsx`.
- **Blender hole-fill on mesh generation.** Every generated shape now gets a headless Blender
  fill-holes pass (`webapp/blender_fillholes.py`: fill boundary loops → recalc normals →
  triangulate) so meshes come out watertight. Always-on, best-effort (a failed pass keeps the
  original mesh), and disable-able via `HY3D_FILL_HOLES=0`. Runs in `_gpu_mesh` and the inline-shape
  branch of `_gpu_base`. `webapp/studio.py`, `webapp/server.py` (`_fill_holes_glb`).

### Changed
- **Studio UI source moved into the repo at `webapp/studio-ui/`** (was an external
  `Downloads/3-d-model-generation-workflow` checkout). Build flow unchanged: `cd webapp/studio-ui &&
  pnpm install && pnpm build`, then copy `out/*` → `webapp/webui/`. `node_modules`/`.next`/`out` are
  git-ignored by the project's own `.gitignore`.

### Fixed
- **Viewer vertex count looked ~3x too high (e.g. 61k vs Blender's 16k).** The badge reported the raw
  glTF vertex-buffer size, which stores one vertex per face-corner wherever normals (flat shading) or
  UVs seam — so a flat-shaded mesh inflates ~3x (`verts ≈ 3 × faces`). `_mesh_stats` now reports
  unique vertex *positions*, matching DCC tools like Blender. Geometry is unchanged (the split is
  correct/required for glTF). `webapp/studio.py`.
- **Remesh / .blend import left a stale rig.** Regenerating or replacing the mesh reset the texture
  but not the rig, so the old `{id}_rigged.glb` (old geometry) lingered and was even preferred by
  Export. Both now invalidate the rig (delete rig artifacts + reset the rig row); the prior rig is
  still recoverable via Mesh versions. `webapp/studio.py` (`_invalidate_rig`).
- **Uploading a modern `.blend` failed with "not a blend file".** Blend files are forward-only: the
  container's tuned Blender 4.2 can't read files saved by Blender 5.x (new header). The Dockerfile now
  also installs Blender 5.0.1 as `blender5`, and `_blender_blend_to_glb` uses it (`BLENDER_NEW_BIN`,
  falling back to `blender` when absent) so modern uploads import while the texture/projection
  pipeline stays on 4.2. The error now also hints at a version mismatch. Requires an image rebuild
  (`docker compose up --build`). `Dockerfile`, `webapp/server.py`.
- **3D viewer spun on its own.** The `<model-viewer>` had the `auto-rotate` attribute, so the model
  rotated without any input. Removed it — the viewer now only moves via mouse (`camera-controls`).
  `webapp/studio-ui/components/studio/model-3d-viewer.tsx`.
- **Studio showed only the in-memory demo, never the real models in `outputs/models`.** The Next.js UI
  defaults to its mock backend (`USE_MOCK = NEXT_PUBLIC_USE_MOCK !== "false"`) and the production build
  shipped with the env unset, so it ran the mock ("Demo robot") instead of calling the real `/api/*`.
  Added `.env.production` with `NEXT_PUBLIC_USE_MOCK=false`; the build now constant-folds `USE_MOCK` to
  `false` and talks to the FastAPI studio backend (rebuild → `webapp/webui/`). Restart the server to pick
  up the rebuilt UI.
- **First hand-paint open showed a broken/blank backdrop.** `useJobRunner.run` resolved as soon as the
  job was *created* (it kicked off polling but didn't await it), so the hand-paint setup set the backdrop
  URL before the render finished → 404, no loader. `run` now resolves on the job's terminal state, so the
  "Rendering the current face…" loader shows until the image is ready, and the render-then-paint and
  sequential reface/reference loops are correctly ordered.
- **Back-left / back-right corners rendered + baked swapped.** The 3/4 back-corner azimuths were
  `bl=225, br=135` in both `studio.CORNER_AZ` (handpaint / face render / clear, via `_cam_for`) and
  `server.HYFACE_CORNER_CAMS` (reface, via `_run_reface`, and base corner fills). In practice those
  framed the opposite corner, so selecting `bl` rendered/baked the back-RIGHT and `br` the back-LEFT
  (front corners `fl=315`/`fr=45` and the left/right cardinals were already correct). Swapped to
  `bl=135, br=225` in both tables (kept consistent). MV-Adapter's separate corner convention
  (`mvadapter_runner.py`, `mvadapter_texture.py`) is unchanged. Re-run any back-corner reface/paint to
  apply.
- **New studio GPU kinds were dropped by the worker loop.** `server._worker_loop` routed only a
  hardcoded tuple (`studio_base/mesh/reface/face_edit`) to `studio.run_gpu_job`, so `studio_face_clear`,
  `studio_face_render` and `studio_handpaint` were dequeued and silently dropped (job stuck at 5%). Now
  routes any `studio_*` kind — no per-kind list to keep in sync.
- **Texture baked onto the mirror face (left↔right, and the 3/4 corners swapped).** Hunyuan's azimuth
  is left-handed (`get_mv_matrix` puts the camera at `[-sin(azim), cos(azim), 0]`), so azim 90 frames
  the object's RIGHT and azim 270 the LEFT — the opposite of the reference convention. `PROJECTION_CAMS`
  fed `left`→90/`right`→270, so a `left` reface/paint landed on the right face; the corner tables had the
  same swap (`fl`/`fr` and `bl`/`br` reversed). Corrected to `left=270, right=90` (`pipeline.py`) and
  `fl=315, fr=45, bl=225, br=135` (`server.HYFACE_CORNER_CAMS`, `studio.CORNER_AZ`). Verified by render:
  the per-face canvas now matches each view's reference instead of its mirror. Re-run base / per-face
  texturing to apply (existing textures were baked at the old azimuths).
- **Worker never drained the studio queue (`__main__` vs `webapp.server` split).** `python -m webapp.server`
  loads the file as `__main__`, a different module object from the `webapp.server` that `studio.py`
  imports — so the GPU worker read `__main__.WORK` while studio jobs were enqueued onto `webapp.server.WORK`.
  Mesh/texture jobs sat unprocessed (stuck at 5%). The launcher now runs `webapp.server.main()` so the
  worker, queue, health, and submit path share one module instance.

### Added
- **Poly-count indicator on the 3D viewer.** A small badge shows the current mesh's `N faces · M verts`,
  computed on the backend from the GLB with trimesh (textured GLB preferred, else the shape), cached per
  file version so `assemble_model` never reloads the mesh on a poll. Exposed as `model.meshStats`
  (`{ faces, vertices }`, null when there's no/unreadable mesh). `webapp/studio.py` (`_mesh_stats`),
  `webapp/studio-ui/components/studio/model-3d-viewer.tsx`, `lib/types.ts`.
- **Custom (free-camera) hand-paint — "Paint this angle".** A button on the 3D viewer captures the
  live `model-viewer` orbit (`elev = 90 − phi`, `azim = theta mod 360` — matches the built-in cameras
  PROJECTION_CAMS front=0/right=90/back=180/left=270; an earlier `360 − theta` mirrored the render) and opens the hand-paint
  canvas on a render at that exact camera. Paint or upload, then bake — the render and bake share the
  camera, so strokes always land where painted. It's a free touch-up (not tied to one of the 10 faces):
  no face slot changes, and it pushes its own `Hand paint custom (e°/a°)` history snapshot. Backend:
  `faces/custom/render?elev&azim`, `faces/custom/render-image`, `faces/custom/handpaint` (multipart
  overlay + elev/azim) reusing `render_textured_view` / `paint_overlay`; `_vview_any` + `_vangles`
  validate the pseudo-view and clamp elev∈[-90,90], azim mod 360. Frontend: `model-3d-viewer.tsx`,
  `lib/api.ts`. The viewer now cache-busts the textured GLB on `updatedAt` so a bake is visible.
- **Hand-paint touch-ups on a face.** A "Hand paint" method on each face renders the face AS IT
  CURRENTLY LOOKS on the mesh (`POST faces/{view}/render` → `studio_face_render` →
  `facerender_{view}.png`, reused while fresh) and lets you brush strokes on it with a palette sampled
  client-side from that view's reference. "Apply" sends an RGBA overlay (transparent except the strokes)
  to `POST faces/{view}/handpaint`, baked straight onto the face by a new
  `TextureWorker.paint_overlay` (direct `back_project`, no rembg/silhouette-fit — the strokes are
  pixel-locked to the camera). Pushes a "Hand paint {view}" history snapshot.
  `components/studio/hand-paint-canvas.tsx`.
- **Hand-paint download / upload + zoom.** The hand-paint surface gains: **Download** (saves the
  current face render as `handpaint-{view}.png` for editing elsewhere), **Upload** (picks an image,
  contain-fits it into the square overlay buffer and bakes it straight onto the face via the existing
  `POST faces/{view}/handpaint` — a downloaded backdrop round-trips 1:1), and **zoom/pan** (mouse-wheel
  zoom centered on the cursor, drag-pan via a Pan toggle or the middle mouse button, Reset-to-fit). The
  drawing buffer stays `dim×dim` and the exported overlay is always the full buffer, so zoom never
  reduces bake resolution; brush mapping stays pixel-accurate under transform. Frontend-only — no
  backend, API, or `static/` change. `components/studio/hand-paint-canvas.tsx` (+ `texture-panel.tsx`
  passes `downloadName`); rebuilt into `webapp/webui/`.
- **Remesh button (textured state).** A collapsible "Remesh" control in the texture panel regenerates
  the 3D shape from a chosen reference view (reuses `POST /mesh`); it warns that this resets the texture
  and history.
- **Texture history (undo/redo per step) + reset + per-face clear.** Every completed base/reface/paint
  auto-snapshots the whole-mesh texture GLB into `outputs/models/{id}/texture_history/{seq}.glb` plus a
  metadata entry (`textureHistory`, `textureSeq` in `model.json`, exposed on the `Model`). New endpoints:
  `POST texture/restore/{seq}` (roll the texture back to any prior step — instant file copy, then
  re-texture forward), `POST texture/reset` (delete the texture back to the untextured mesh; history
  kept), `POST faces/{view}/clear` (GPU re-bake reverting ONE face to the base by projecting the base
  snapshot's render onto that face). Regenerating the mesh clears the history (snapshots belong to the
  old UV). Frontend: a "Texture history" timeline with per-step Restore, a "Reset" button, and a per-face
  Clear (↩) on each face tile. Tests: `test_texture_history_snapshot_restore_reset_clear`,
  `test_mesh_regen_clears_texture_history`. A per-face edit now also self-heals `textureStage` to
  `complete` (it only runs on an already-textured model).


- **Per-model Studio API + new Next.js frontend.** A durable, named *model* owns its 10 reference
  views, mesh, and texture (reusable across runs; survives restart). New modules `webapp/studio.py`
  (registry + `/api/models/*` router + unified job store, persisted as
  `outputs/models/{id}/model.json`) and `webapp/reference_views.py` (staged, mesh-free gpt-image-2
  reference generation along the imperative graph front→cardinals→corners, with per-view
  orthographic/cartoonish prompts + tweak prompts). Two execution lanes: the existing single GPU
  worker (shape + per-face paint + reface) and a `ThreadPoolExecutor` network lane for reference
  generation. Endpoints: model CRUD + rename; references generate/upload/approve/image; seed;
  `texture/base` (mesh + per-face AI paint over all approved refs → `complete`);
  `texture/reface/{view}` (depth-aware, uses the view's approved reference + optional tweak);
  `faces/{view}/edit` (`paint` = localized single-face composite | `reface`); jobs poll (completed
  jobs embed the full `Model`); download glb/fbx/blend. The built Next.js static export is served
  same-origin via `HY3D_WEBUI_DIR` (point it at the `out/` dir).
  - `webapp/pipeline.py`: `TextureWorker.paint_single_view` (single-face paint for localized edits).
  - `webapp/gen_transfer.py`: `restyle_to_references(..., extra_prompt=)` threads the reface tweak.
  - `webapp/server.py`: studio router mounted before the static UI; worker loop dispatches
    `studio_base`/`studio_reface`/`studio_face_edit`; `GET /api/jobs/{id}` delegates studio job ids.
  - Tests: `python -m webapp.test_studio_api` (contract/gating/persistence; gen + GPU mocked),
    `python -m webapp.test_reference_views` (prompt + dependency graph).
  - Docker: `docker-compose.yml` sets `HY3D_WEBUI_DIR=…/webapp/webui` (under the existing `./webapp`
    bind mount) so the built Next.js export is served same-origin with no image rebuild
    (`pnpm build` → copy `out/.` → `webapp/webui/` → `docker compose up -d`).

- **Draw-to-edit (masked inpaint) on reference views.** In a reference view's Edit dialog you can now
  brush over the exact region to change (adjustable brush size + Clear) and "Edit masked region"
  repaints ONLY that area via gpt-image's mask, keeping everything else pixel-identical — far more
  control than a full regenerate (still available as "Regenerate whole view").
  - `webapp/reference_views.py`: `edit_view_masked` (OpenAI inpaint; converts the brush mask to the
    gpt-image alpha mask) + `build_edit_prompt`. Requires `OPENAI_API_KEY` (Gemini has no mask API).
  - `webapp/studio.py`: `POST /api/models/{id}/references/{view}/edit` (multipart `mask` +
    `edit_prompt`) → Job on the network lane; the edited view returns to `pending` for re-approval.
  - Frontend: `components/studio/mask-canvas.tsx` (brush overlay on the image), `lib/api.ts`
    `editReferenceMasked`, wired into the reference Edit dialog (`image-dialog.tsx` `imageSlot`).
  - Test: `webapp/test_studio_api.py::test_masked_edit_reference`.

- **Choose the mesh source view + iterate on the mesh.** Mesh generation is now a separate step from
  texturing: pick which reference view drives the 3D shape (a 3/4 corner often gives Hunyuan more
  depth than a flat front), **Generate mesh**, preview it in the viewer's Shape tab, and **Regenerate
  mesh** with a different view if you don't like it — then **Generate textures**.
  - `webapp/studio.py`: `POST /api/models/{id}/mesh` (`source_view` + shape params) → GPU
    `studio_mesh` job → `generate_shape` from that view; regenerating resets any existing texture.
    `texture/base` now reuses an existing mesh instead of always regenerating the shape; `Model`
    exposes `meshSourceView`.
  - Frontend: source-view selector + split **Generate mesh / Regenerate mesh / Generate textures** in
    `texture-panel.tsx`; `api.generateMesh`.
  - Test: `webapp/test_studio_api.py::test_mesh_endpoint`.

- Web app: new `hyface` texture mode — "Per-face AI paint (Hunyuan)". Paints each
  face/side of the model individually with single-view Hunyuan paint, conditioned on
  that face's own uploaded reference (front = main image; back/left/right/top/bottom
  from the projection panel). Faces with no upload are filled by gpt-image-2 from the
  face geometry + the other references (needs `OPENAI_API_KEY`). All painted faces bake
  into one shared UV texture (cosine-blend + inpaint); albedo-only matte output.
  - `webapp/pipeline.py`: `TextureWorker.paint_faces`.
  - `webapp/server.py`: `_run_hyface` + dispatch in `_run_texture` (`texture_mode == "hyface"`).
  - `webapp/static/index.html` + `app.js`: mode option, hint, per-side upload wiring.
  - Depth coverage: besides the 6 cardinal faces, the bake adds down-weighted FILL views —
    tilted cardinals (elev ±`HYFACE_TILT_ELEV`) and 3/4 corners (`HYFACE_CORNER_ELEV`
    down-tilt) — so oblique / recessed texels the head-on cardinal views only graze get
    painted instead of inpainted. Corner references can be uploaded directly (Front-L/-R,
    Back-L/-R slots → `fl/fr/bl/br` uploads); any corner left empty is gpt-synth'd from its
    geometry + adjacent faces (needs `OPENAI_API_KEY`).
    Tunable via `HYFACE_CORNERS`, `HYFACE_TILT`, `HYFACE_TILT_ELEV`, `HYFACE_CORNER_ELEV`,
    `HYFACE_FILL_WEIGHT`, `HYFACE_BAKE_EXP`. New `TextureWorker.render_geometry_at`;
    `paint_faces` now takes per-view `(ref, elev, azim, weight)` specs + `bake_exp`.
  - Explicit Front reference: a hyface-only Front slot in the panel (`front` upload on
    `/api/generate`, `/api/retexture`, `/api/jobs/{id}/texture`) overrides the front face
    reference; when omitted it falls back to the main image (`processed_image_path` /
    `source_paths[0]`). Projection mode is unchanged (the Front slot is hidden there, and it
    still treats front as the main image).
  - Additive only — existing modes (`hunyuan`, `projection`, `gptproject`) are unchanged.

- Web app: new `reface` texture mode — "Reface, depth-aware single face (existing texture)".
  Re-textures ONE view of an already-textured mesh and is depth-aware: only the nearest depth
  band (foreground) is repainted; farther surfaces keep their existing texture (a car in front
  of a wall → only the car changes). View can be any of the 6 cardinal faces or the 4 3/4
  corners (fl/fr/bl/br); corners resolve to the corner camera (azimuth + `HYFACE_CORNER_ELEV`).
  - `webapp/pipeline.py`: `TextureWorker.reface` — loads the mesh preserving its existing UVs +
    texture (the base), computes screen-space depth (`render_position` decoded back to world,
    camera pos = `-Rᵀt` from the view matrix, euclidean), builds a foreground mask (nearest depth
    band or a user mask), GEOMETRY-MATCHES the generated view to the mesh silhouette
    (`_align_photo` + `_silhouette_bbox`, same as projection/gptproject — so scale + position
    follow the geometry, not gpt's drift), bakes it as RGBA so `back_project` carries the mask in
    its alpha, and composites foreground texels over the base. Albedo-only matte output.
  - `webapp/server.py`: `_run_reface` generates the face via the MV+GPT per-face workflow —
    gpt generates from a grey geometry render + references, then Gemini transfers it onto the
    geom to lock proportions/position (`_view_texture`/`gen_transfer`), so the result is
    geometry-locked, not gpt-drifted. The grey geom is rendered from the Hunyuan camera
    (`TextureWorker.render_geom_shaded`) so the locked output aligns to reface's bake. Plus the
    `/api/reface` endpoint + dispatch in `_run_texture`. Foreground band tunable via
    `REFACE_DEPTH_BAND` (default 0.35); optional upload mask overrides it.
  - `webapp/static/index.html` + `app.js`: mode option, reface panel (face selector + optional
    mask), references via the existing reference panel, `texBtn` → `/api/reface` on an existing
    textured model.
  - Note: output is matte (existing metallic/roughness is not reloaded from the GLB).

### Fixed
- Editing the seed image after creation overwrote the **front reference** instead of the seed
  (the SeedRow posted to `references/front/upload`). Added a dedicated `POST /api/models/{id}/seed`
  endpoint + `api.updateSeed`; the SeedRow now updates the seed only (front reference untouched). The
  create form also shows a thumbnail **preview** of the chosen seed so you can confirm it uploaded.
  (`webapp/studio.py`, frontend `lib/api.ts` / `mock-backend.ts` / `references-panel.tsx` /
  `model-library.tsx`)
- Reference-view generation prompts (`build_prompt`) reworked for correctness and made
  model-agnostic. They originally stripped the seed's surroundings ("a single centered object … no
  ground plane, no scenery"); a first fix ("keep every element") overcorrected and dragged
  front-only elements into views that can't see them. They now use per-view VISIBILITY rules that
  keep any surrounding elements in their REAL positions and show them ONLY where the camera can see
  them, with no scene-type hardcoding (generic "object / surrounding elements" — no cars/lot/facade/
  building wording). Specifics:
  - The four side views (front/back/left/right) use a **ground-aligned, perfectly horizontal,
    untilted** camera, so the floor is seen edge-on (a thin strip) — the full floor only appears in
    the **top** view.
  - Top = strict overhead plan with no vertical faces; bottom = bare underside; back / back-corners
    exclude front-only elements.
  - Left/right state the **exact rotation direction** (counter-clockwise / clockwise, as seen from
    directly above) to reach the correct side.
  - **Each view is fed only the references it needs.** left/right ← front + top (the side doesn't
    show the back); corners ← their two adjacent cardinals + top (`fl`←front+left+top,
    `fr`←front+right+top, `bl`←back+left+top, `br`←back+right+top). The side prompts use the TOP view
    as the overhead layout map and the FRONT view for appearance, preserving every visible element in
    its **relative position with correct occlusion**. (frontend `lib/views.ts` + backend
    `VIEW_INPUTS` kept in sync.)
  - **3/4 corners no longer flip left/right.** Each corner now depends only on the side it shows
    (`fl`←front+left+top, `fr`←front+right+top, `bl`←back+left+top, `br`←back+right+top) so the
    opposite-side reference can't leak in, and the corner prompts use the TOP view as the handedness
    authority with an explicit anti-mirror rule ("the visible side MUST match the approved LEFT/RIGHT
    view; do not mirror"). Each corner prompt also states its rotation explicitly: fl/fr = the FRONT
    rotated 45° counter-clockwise / clockwise; bl/br = the BACK rotated 45° clockwise /
    counter-clockwise (seen from above).
  - Each cardinal view is **single-face** — explicitly no other canonical view and no 3/4 angle (the
    3/4 corners still show two sides + top).
  - Cardinals are generated from the original **seed** (Image 1 = the authority for the model's
    content and layout) plus the approved front (Image 2 = colour/finish reference).
  Regenerate references to pick this up. (`webapp/reference_views.py`)
- Studio web UI not refreshing a reference image / 3D texture after a replacement upload or a
  reface/paint edit: the backend rewrites the same file path, so the asset URL was unchanged and the
  browser served the cached image. `assemble_model` now appends a cache-busting `?v={updatedAt}` to
  every reference/seed/mesh/textured URL (so the URL changes on each write) and the studio image
  responses send `Cache-Control: no-store`. (`webapp/studio.py`)

### Removed
- Web app: texture modes `hunyuan`, `projection`, `gptproject`, `mvadapter`, `mvgpt` and their
  handlers/helpers (`_run_projection`, `_run_gpt_projection`, `_mv_texture`, the mvgpt
  geometry/colorize/corner/view-texture/PBR-base cluster), the `POST /api/jobs/{id}/resume`
  endpoint, and the `mvadapter` health flag. Only **per-face AI paint (`hyface`)** and **`reface`**
  remain; the default texture mode is now `hyface`. Shared helpers (`image_edit.py`,
  `gen_transfer.py`, `blender_project.py`, the `gpt_angles` field) are retained.

### Changed
- **Reface restyle directives.** Where a colour already matches the reference, reface now sharpens and
  lifts the detail/resolution to the reference's crispness instead of just leaving it; and it keeps each
  colour inside its object's silhouette — no colour bleed onto neighbouring objects (`gen_transfer.py`,
  both gpt + gemini stages).

- MV+GPT (`mvgpt`) blender path is now resumable and parallel:
  - Resume: `MVGPT_REUSE` (default on) reuses any per-side artifact already on disk —
    elevations (`generate_elevations`), grey geometry renders (`_blender_geometry`), per-side
    genview/elevmatched (`_view_texture`), and the Hunyuan PBR base (`_hunyuan_pbr_base`) — so a
    re-queued job finishes only the missing sides + the final bake. Set `MVGPT_REUSE=0` to force
    a clean regen.
  - `POST /api/jobs/{id}/resume` rebuilds an interrupted mvgpt job from its on-disk artifacts
    (same id) and re-queues it. Side tags aren't persisted, but each side is re-derived from the
    existing elevations, so resume doesn't need them.
  - Parallel genviews: the per-side geomatch generation now runs `front` first (the style anchor)
    then fans the remaining sides out concurrently (`MVGPT_GENVIEW_WORKERS`, default 4) instead of
    one-at-a-time. Each genview is a network-bound gpt/gemini call, so wall time drops ~Nx.
- MV-Adapter + GPT refine (`mvgpt`) and `mvadapter`: references can now be tagged with a
  3/4-corner side (`fl`/`fr`/`bl`/`br`) in the reference panel, not just cardinal faces.
  - Blender elevation path (default mvgpt): a corner-tagged reference is used as the colour
    authority for that 3/4 view (adjacent-face elevations still ride as consistency refs).
  - Raw MV refine path: `_view_sides` emits the corner tag first and `_refs_for` feeds a
    corner-specific reference outright instead of blending both adjacent cardinals.
  - `webapp/static/app.js`: corner options + labels in the reference side dropdown.
  - No-op when no corner-tagged reference is provided (existing behavior preserved).
