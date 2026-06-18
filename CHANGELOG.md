# Changelog

## Unreleased

### Added
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
