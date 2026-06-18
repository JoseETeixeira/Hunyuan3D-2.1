# Understanding — Texture the weird-angle surfaces no view covers (reface + per-face AI paint)

Task slug: `oblique-angle-coverage`

## User goal

Some surfaces sit at "weird angles" that NO existing camera view sees head-on —
the red-circled slanted ramp / facade walls in the screenshot (inward-facing
oblique walls between parking levels). They come out untextured / smeared. Make
both **reface** and **per-face AI paint (`hyface`)** able to texture those
oblique surfaces. The fix must work in BOTH modes.

## Root cause (confirmed in code)

The Hunyuan bake is cosine-gated. `DifferentiableRenderer/MeshRender.py:366`
sets `bake_angle_thres = 75`; `back_project` (line 1192-1193, 1245-1247) zeroes
any texel whose surface normal is more than 75° off the camera axis
(`cos_image[cos_image < cos_thres] = 0`). A texel only gets real colour if SOME
baked view sees it within 75°. If every available camera grazes it (>75°), it is
never written, then `texture_inpaint` (`MeshRender.py:1411`, `cv2.INPAINT_NS`,
island-blind) floods it with the nearest neighbour colour → the "unpainted /
wrong colour" weird-angle patches.

So the gap is purely **camera coverage**: the current view sets leave a cone of
surface normals unhit.

## Available cameras today

Canonical table — `pipeline.py:374` `TextureWorker.PROJECTION_CAMS`
(elev, azim): front (0,0) back (0,180) left (0,90) right (0,270) top (90,0)
bottom (-90,0).

3/4 corners — `server.py:735` `HYFACE_CORNER_CAMS` azim fl45 bl135 br225 fr315,
elevation `_HYFACE_CORNER_ELEV` = 45.

- **reface** (`_run_reface` server.py:944; `TextureWorker.reface` pipeline.py:578):
  re-textures ONE face per job over an already-textured mesh. The UI face picker
  (`index.html:89` `#refaceFace`) offers exactly {front,back,left,right,top,bottom,
  fl,fr,bl,br}. Server resolves that name → (elev,azim) from the two tables above
  (server.py:959-966). KEY: `worker.reface(elev, azim, ...)` already accepts an
  ARBITRARY (elev,azim) — only the name→camera resolver and the dropdown limit it.
  A user mask is already supported (`reface_mask_path`, `index.html:104`).
- **hyface** (`_run_hyface` server.py:767; `TextureWorker.paint_faces`
  pipeline.py:491): paints each face with its own ref via 1-view Hunyuan, bakes
  all into one UV texture. Fill views already exist to chase oblique texels:
  tilted cardinals elev ±20 (`_HYFACE_TILT`, server.py:846), 4 corners elev 45
  (`_HYFACE_CORNERS`, 859), optional below-horizon elev −45 (`_HYFACE_LOW`, 907,
  default OFF). KEY: `paint_faces(view_specs=[(ref,elev,azim,weight),...])`
  already accepts ARBITRARY (elev,azim) specs — the server just builds the list
  from fixed tables.

## The lever that makes a fix cheap

Both back-ends are already angle-generic:
- `worker.reface(elev, azim, view_image, mask=...)` — any camera.
- `worker.paint_faces(view_specs)` — any list of (ref, elev, azim, weight).
- `worker.render_geometry_at(shape_glb, cams)` (pipeline.py:779) — normal render
  at ARBITRARY labelled cameras (already used to seed gpt-synth corners).
- `worker.render_geom_shaded` / `render_textured_view` — grey/colour render at any
  camera, the canvas reface restyles.
- `_openai_paint_view(geom, refs, angle)` (server.py:218) — gpt-image-2 synth of a
  view from a geometry render + refs (Gemini fallback). Already used to fill empty
  hyface faces and corners.

So a "custom angle" view needs: render geometry at (elev,azim) → get/synth a
reference → feed to `reface` or append to `paint_faces` specs. No renderer or
baker change required.

## Likely change surface (additive)

- `webapp/server.py`
  - reface: extend the face→camera resolver (`_run_reface` ~959) to accept a
    custom (elev,azim); add params to `/api/reface` (`server.py:1648`).
  - hyface: append extra custom-angle specs in `_run_hyface` (~903) from a new
    job field; each ref gpt-synth'd via `render_geometry_at` + `_openai_paint_view`.
  - (If auto path chosen) a coverage-gap detector: bake → read `cos_map` /
    uncovered UV mask → cluster uncovered texels by world normal → emit fill
    cameras aimed down those normals; run them through the same spec path.
- `webapp/pipeline.py`: likely no core change (back-ends already angle-generic);
  maybe a helper to compute the uncovered-texel → normal clustering if auto.
- `webapp/static/index.html` + `app.js`: UI to specify the weird angle(s) —
  custom elev/azim entry, or "capture current viewer angle", or an "auto-cover
  gaps" toggle (depends on the design decision below).

## Resolved decisions (grill-me)

1. **Approach = Auto-detect gaps** (Option A). After the bake, find texels no
   view covered, cluster them by surface normal, auto-aim fill cameras down each
   cluster's mean normal, get a reference, re-bake. No manual aiming. Same
   mechanism in both modes.
2. **Activation = Default ON in BOTH modes.** reface and hyface both always run
   the gap-fill stage. ⚠ This CHANGES current behavior of both modes (architecture
   change — validated by user). A sane **env-tunable camera cap** is kept as a
   guardrail (uncapped was explicitly rejected).
3. **Gap definition = zero-coverage + small dilation.** Target only texels NO
   view baked (cos_map==0 → currently inpaint-smeared), plus a few-texel border
   around each gap so new paint blends over the inpaint seam instead of stopping
   at the gap edge. Grazing/low-cos texels are NOT in scope.
4. **Reference per mode (follows existing patterns):**
   - reface: render the already-textured mesh at the gap camera
     (`render_textured_view`) → `restyle_to_references` toward the user's refs →
     `worker.reface` bake. Multiple gap clusters → multiple sequential reface
     passes in one job, each compositing over the prior base.
   - hyface: gpt-synth the fill reference from the geometry render at the gap
     camera (`render_geometry_at` + `_openai_paint_view`) using adjacent/nearest
     face refs — same path the corner fills already use. Degrade gracefully
     (skip a cluster) when `OPENAI_API_KEY` is absent and no usable ref exists.

## Major design decision (architecture — RESOLVED above; options kept for record)

How does the user target a surface "not covered by any view"? Options:

- **A. Auto coverage-gap fill.** After the normal bake, detect texels no view hit
  (uncovered UV mask / low cos_map), cluster by surface normal, auto-place fill
  cameras down each cluster's mean normal, synth refs from geometry, bake. Zero
  user aiming — directly solves "not covered by any view". Most code; needs a
  reliable uncovered-island→normal pass. Works the same for both modes.
- **B. Manual custom angle(s).** User types elev/azim (or N custom angles); each
  becomes a reface pass / hyface fill spec. Simple, deterministic; user must know
  the angle.
- **C. Pick angle from the 3D viewer.** Orbit `model-viewer` to face the surface,
  capture its camera (elev/azim) → custom view. Intuitive given the screenshot;
  needs viewer→(elev,azim) plumbing in `app.js`.
- **D. Add a fixed extra tier** (e.g. steeper tilts / mid-elevation corners like
  the `_hi` set in `gen_transfer.VIEW_SPEC`). Cheapest; may still miss arbitrary
  oblique walls.

These are mostly combinable (e.g. A as default + C for manual touch-up).

## Gap-detection algorithm — two candidates (decide in Design)

The crux of the auto approach is "find uncovered texels → place fill cameras."
After baking, the merged `cos_map` (UV space, `texture_size`) marks coverage; a
texel is a gap if it belongs to the mesh (valid `tex_grid` / UV rasterization,
MeshRender.py:935/1059) but `cos_map == 0` across all views.

- **Candidate 1 — UV-normal clustering.** Interpolate vertex normals into UV
  space (same UV rasterization the texture uses) → per-texel world normal. Take
  gap texels, cluster their normals on the sphere (binning / k-means), and for
  each cluster aim a camera down the mean normal → (elev, azim). Direct, but
  needs careful UV-normal extraction + cluster tuning.
- **Candidate 2 — greedy camera set-cover (likely simpler/robust).** Sample a
  dense candidate camera set (e.g. Fibonacci sphere of N directions). For each
  candidate, render the mesh and count how many CURRENTLY-uncovered texels it
  would newly bake (cos within 75° + visible via back_project). Greedily pick the
  top-K candidates (up to the cap) that maximise newly-covered gap texels, stop
  when a round adds < ε new texels. Reuses existing render/back_project; no UV
  normal math; naturally bounded by the cap.

Both feed the SAME downstream: per chosen camera, get a reference (per-mode rule
above) and bake the gap region (zero-coverage ∧ dilation mask) so only gaps are
written and existing good texels are preserved.

## Open questions / risks

- Reference for a custom view: gpt-synth from geometry (needs `OPENAI_API_KEY`)
  vs a user upload per custom angle.
- reface is one-pass single-face today; covering several weird angles may need
  multiple reface passes (sequential composite over the base) or a multi-angle
  reface.
- Auto path: clustering oblique normals robustly, and avoiding double-painting
  already-good texels (respect existing cos coverage / depth band).
- More views = more diffusion passes (hyface) or API calls (gpt-synth) → cost.
- Inpaint still fills truly-unseeable texels (deep cavities); coverage can shrink
  the gap, not always eliminate it.

## Verification ideas

- Reproduce on the screenshot model: bake, dump the uncovered-texel UV mask,
  confirm the red-circled ramp walls are in it.
- After fix: those texels carry real colour (cos_map > 0 there), not inpaint.
- Each existing mode/route unchanged when the new option is off (additive).

## Notes

- Cocoindex index points at the `coding-cli` workspace, not this `3D-Gen` repo, so
  `search_codebase` returned lifeverse hits only; understanding built via
  Grep/Read of `Hunyuan3D-2.1/webapp` + `hy3dpaint` (allowed fallback — target
  repo not in the semantic index).
- Prior related task: `.batman/per-face-paint-mode/` (added the `hyface` mode).
