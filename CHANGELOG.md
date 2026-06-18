# Changelog

## Unreleased

### Added
- Web app: auto **coverage-gap fill** for `reface` and `hyface` (default ON). After the base
  bake, texels no standard view covered — normals grazing every camera >75° (the
  `bake_angle_thres` gate) OR occluded behind another part — are blank and get smeared by UV
  inpaint (the oblique ramp/facade walls). This stage re-probes the 10 standard cameras' real
  coverage (`back_project` a dummy → accumulate `cos_map` → gap = valid & ~covered, catching both
  grazing AND occlusion), greedily aims a capped set of extra fill cameras at the uncovered
  normals, gets a reference per camera, and composites the paint ONLY onto the gap region (+ a
  small dilation) so covered texels are untouched. Projection bake only (no diffusion UNet).
  - `webapp/pipeline.py`: new `TextureWorker.fill_coverage_gaps` (runs as an auto-targeted reface
    on the reloaded textured GLB — one path shared by both modes).
  - `webapp/gapfill_logic.py`: pure (numpy) camera-ranking core (`best_candidate`), unit-tested.
  - `webapp/server.py`: gap-fill wired into `_run_reface` + `_run_hyface` (best-effort, non-fatal —
    the base bake still ships on any failure); `_gap_reference` ladder (gpt/Gemini synth from the
    gap geometry + nearest colour refs → reuse nearest ref → skip), `_nearest_face_imgs`,
    `_gapfill_camera_sets`.
  - Reference per gap camera: reface uses the user's references; hyface uses the nearest
    already-painted faces; both degrade gracefully without `OPENAI_API_KEY` (reuse nearest ref).
  - Tunable via `GAPFILL_REFACE`/`GAPFILL_HYFACE` (default 1), `GAPFILL_MAX_CAMS` (6),
    `GAPFILL_DILATION` (4), `GAPFILL_COS_DEG` (75), `GAPFILL_MIN_TEXELS` (64),
    `GAPFILL_GRID_ELEVS` (`-60,-30,0,30,60`), `GAPFILL_GRID_AZ_STEP` (30). Set the per-mode toggle
    to 0 for exact legacy behavior.
  - Verify: `python webapp/diag_gapfill.py <uid>` (dumps the gap mask + coverage counts);
    `python webapp/test_gapfill_logic.py` (pure unit tests for the selection core).
  - Additive only — other modes (`hunyuan`, `projection`, `gptproject`, `mvadapter`, `mvgpt`) are
    unchanged.
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
