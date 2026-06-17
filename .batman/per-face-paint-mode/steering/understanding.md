# Understanding — Per-Face AI Paint Texture Mode

Task slug: `per-face-paint-mode`

## User goal

Add a NEW texture mode that paints each face/side of a model individually with
Hunyuan paint, using a per-face uploaded reference, instead of one global
reference for the whole model. Must be purely additive — existing texture modes
(`hunyuan`, `projection`, `gptproject`, `mvadapter`, `mvgpt`) unchanged.

## Locked decisions (from grilling)

- Architecture: **view-project**. Keep the mesh whole. Render a view per face,
  Hunyuan single-view paints each face's upload, back-project all into ONE shared
  UV texture (cosine-blend + inpaint). No geometry split, no seams.
- Flow: **batch one-pass** (no iterative accumulation / persistent texture state).
- Empty faces (no upload): **gpt-image-2 synthesizes a reference** from that
  face's geometry render + the other uploaded refs, then Hunyuan paints it.
  Reuses the existing gptproject path.
- Output: **albedo-only matte** (drop metallic/roughness), like projection mode.

## Feasibility — confirmed YES

Hunyuan paint can run a single view. The multiview net derives the view count at
runtime:

- `multiview_utils.py:90` — `num_view = len(control_images) // 2`. Pass ONE
  camera's normal+position map (`control = [normal, position]`) → `num_view = 1`
  → it paints exactly that one face and returns `{"albedo": [img], "mr": [img]}`.
- No hard 6-view assertion in the inference path. `num_in_batch` is set
  dynamically and the multiview attention degenerates to self+reference attention
  at 1 view.
- Caveat: the model was trained on a joint 6-view layout
  (`cfgs/hunyuan-paint-pbr.yaml` `num_view: 6`), so an isolated single view loses
  cross-view consistency. For per-face painting that is acceptable — each view is
  independently conditioned on its own geometry + its own reference, which is the
  whole point of the mode.

Why a new mode is needed: today's `hunyuan` mode conditions ALL views on the
FIRST image only (`server.py:651-654` comment: "cannot take a different face
reference per view without retraining"). Running the net once per face with that
face's own upload bypasses that limit cleanly.

## Current behavior — how the pieces work today

- `Hunyuan3DPaintPipeline.__call__` (`hy3dpaint/textureGenPipeline.py:92-188`):
  view-selection → render normal+position per view → `multiview_model(style, normals+positions)`
  paints ALL views jointly → super-res → `bake_from_multiview(albedo, elevs, azims, weights)`
  → inpaint → set texture → save. This is the canonical single-call flow. Note it
  bakes the painted views DIRECTLY (no silhouette re-align) because the views are
  rendered from the exact geometry.
- `multiviewDiffusionNet.forward_one` (`hy3dpaint/utils/multiview_utils.py:73-128`):
  the single entrypoint that derives `num_view` and returns the PBR dict.
- Web worker `TextureWorker` (`webapp/pipeline.py`):
  - `generate_texture` (238-280): runs the full joint paint (one ref → all views).
  - `project_texture` (292-344): bakes per-angle photo dict via `_align_photo`
    silhouette fit + `bake_from_multiview`, albedo-only matte. Used by
    projection/gptproject. Per-angle camera table = `PROJECTION_CAMS` (283-290).
  - `render_view_geometry` (399-425): `{angle: normal map}` from `PROJECTION_CAMS`.
  - `_move_multiview` (171-182): GPU↔CPU swap for the paint UNet (sequential VRAM).
- Server (`webapp/server.py`):
  - `_run_texture` dispatcher (621-661): branches on `job["texture_mode"]`.
  - `_run_gpt_projection` (~390-432): renders geometry → `_openai_paint_view(geom[angle], refs, angle)`
    per angle → `project_texture`. This is the gpt-image fill pattern to reuse.
  - `_openai_paint_view` (158+) + `webapp/image_edit.py` `edit_image`: gpt-image-2
    paints a view from a normal render + style refs (Gemini fallback).
  - Per-side uploads already plumbed everywhere: `/api/generate` (935-1026),
    `/api/retexture` (1149-1225), `/api/jobs/{id}/texture` (1029+) accept
    `back/left/right/top/bottom` + front and store them in `job["view_paths"]`
    (`{angle: path}`). `texture_mode` Form param already threaded through.

## Likely change surface (additive only)

- `webapp/pipeline.py`: NEW `TextureWorker.paint_faces(uid, shape_glb_path, face_refs: {angle: PIL}, albedo_only=True)`.
  Per face: render its normal+position (1 view) → `multiview_model([ref], [normal, position])`
  → albedo. Collect `{angle: albedo}` → bake DIRECTLY via `bake_from_multiview`
  (NOT `_align_photo`, views already geometry-aligned) → inpaint → matte export.
  Handle `_move_multiview` GPU swap like `generate_texture`.
- `webapp/server.py`: NEW `_run_hyface(job_id)` + one dispatch line in
  `_run_texture`. Gather `job["view_paths"]` as per-face refs; for the chosen face
  set, fill empty faces with `_openai_paint_view(geom[angle], other_refs, angle)`;
  call `worker.paint_faces`. New `texture_mode == "hyface"` value (no signature
  changes — the Form param already exists).
- `webapp/static/index.html`: NEW `<option value="hyface">` in `#texmode` + a hint.
- `webapp/static/app.js`: `applyTextureMode` shows the per-side upload panel
  (`els.projPanel`) for `hyface`; `appendTextureFields` / generate handler append
  per-side `viewFiles` for `hyface` (mirror the `projection` branch).

## Open questions / risks

- Single-view quality vs joint 6-view (flagged above) — acceptable per goal;
  verify on a real model during implementation.
- Face set to paint: default to the 6 `PROJECTION_CAMS` faces; only paint
  provided + gpt-filled faces, inpaint the rest.
- gpt-image fill requires `OPENAI_API_KEY`; degrade gracefully (skip fill, paint
  only uploaded faces) when absent, like gptproject's fallback.

## Verification ideas

- Unit-ish: `paint_faces` with 1 face ref → produces a GLB, only that face's
  region textured, rest inpainted.
- Each existing mode still routes/behaves unchanged (dispatch table untouched).
- End-to-end: retexture an existing shape in `hyface` mode with 2-3 face uploads.
