# Changelog

## Unreleased

### Added
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
    texture (the base), computes screen-space depth (`render_position` + `render_alpha`, camera
    pos = `-Rᵀt` from the view matrix, euclidean), builds a foreground mask (nearest depth band
    or a user mask), bakes the generated face as RGBA so `back_project` carries the mask in its
    alpha, and composites foreground texels over the base. Albedo-only matte output.
  - `webapp/server.py`: `_run_reface` (generates the face via the gpt geomatch — the mvgpt
    "gpt refine" generation — from references) + `/api/reface` endpoint + dispatch in
    `_run_texture`. Foreground band tunable via `REFACE_DEPTH_BAND` (default 0.35); optional
    upload mask overrides it.
  - `webapp/static/index.html` + `app.js`: mode option, reface panel (face selector + optional
    mask), references via the existing reference panel, `texBtn` → `/api/reface` on an existing
    textured model.
  - Note: output is matte (existing metallic/roughness is not reloaded from the GLB).

### Changed
- MV-Adapter + GPT refine (`mvgpt`) and `mvadapter`: references can now be tagged with a
  3/4-corner side (`fl`/`fr`/`bl`/`br`) in the reference panel, not just cardinal faces.
  - Blender elevation path (default mvgpt): a corner-tagged reference is used as the colour
    authority for that 3/4 view (adjacent-face elevations still ride as consistency refs).
  - Raw MV refine path: `_view_sides` emits the corner tag first and `_refs_for` feeds a
    corner-specific reference outright instead of blending both adjacent cardinals.
  - `webapp/static/app.js`: corner options + labels in the reference side dropdown.
  - No-op when no corner-tagged reference is provided (existing behavior preserved).
