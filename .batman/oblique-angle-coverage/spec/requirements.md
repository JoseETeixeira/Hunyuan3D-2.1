# Requirements — Oblique-Angle Coverage (auto gap-fill for reface + per-face AI paint)

**Document Information**
- Feature Name: Oblique-Angle Coverage (auto coverage-gap fill)
- Version: 1.0
- Date: 2026-06-17
- Author: Batman
- Stakeholders: 3D-Gen texture pipeline owner (single dev); end users of the webapp texture modes

## Introduction

Surfaces at "weird angles" — oblique walls whose surface normal grazes every
fixed camera by more than 75° — never receive real texture. The Hunyuan bake is
cosine-gated (`bake_angle_thres = 75`, `MeshRender.py:366`); a texel is coloured
only if some baked view sees it within 75°, otherwise it falls to UV inpaint
(`cv2.INPAINT_NS`, island-blind), which floods it with the nearest neighbour
colour. The result is the smeared / wrong-colour patches the user circled (the
inward-facing parking-ramp facade walls).

This feature adds an automatic coverage-gap fill stage to the two affected modes
— **reface** (`_run_reface` / `TextureWorker.reface`) and **per-face AI paint /
hyface** (`_run_hyface` / `TextureWorker.paint_faces`). After the normal bake it
detects texels that no view covered, places extra fill cameras aimed at those
uncovered surface normals (bounded by a cap), obtains a reference per camera, and
bakes ONLY the gap region (plus a small dilation border) so existing good texels
are preserved. It is additive: no other texture mode changes.

## Feature Summary

Detect zero-coverage texels after the bake, auto-place a capped set of fill
cameras down their surface normals, and re-bake only those texels in both reface
and hyface, default on.

## Business Value

Removes the most visible texture defect (blank/smeared oblique walls) without
manual camera aiming, raising output quality of every reface and hyface run.

## Scope

In scope:
- Coverage-gap detection from the existing bake trust map + per-texel UV normals.
- Automatic fill-camera placement aimed at uncovered normals, capped.
- Integration into reface and hyface, default on.
- Gap-only baking with small dilation; preserve already-covered texels.
- Env-var configuration + guardrails; structured logging.

Out of scope:
- Grazing / low-cos texels that ARE covered (only zero-coverage is targeted).
- Other texture modes (`hunyuan`, `projection`, `gptproject`, `mvadapter`,
  `mvgpt`) — unchanged.
- A new manual angle-entry or viewer-pick UI (auto only; default-on needs no UI).
- Eliminating texture on truly unseeable interior cavities (no camera can reach
  them; they remain inpaint-filled).

---

## Requirements

### Requirement 1: Coverage-gap detection
**User Story:** As the pipeline, I want to know exactly which mesh texels no
camera view baked, so that I can target only the real gaps.

**Acceptance Criteria (EARS)**
- WHEN a base bake completes THEN the gap-fill stage SHALL compute a UV-space gap
  mask = valid mesh texels (from the texture rasterization / `tex_grid`) whose
  accumulated coverage trust is ≤ the bake's coverage threshold (`trust_map`
  not > 1e-8).
- WHERE the mesh exposes per-texel world normals (`render.tex_normal`) THE gap-fill
  stage SHALL associate each gap texel with its surface normal.
- IF the gap mask is empty THEN the gap-fill stage SHALL skip all further gap work
  and return the base bake unchanged.
- WHEN computing the gap mask THE gap-fill stage SHALL NOT mark as a gap any texel
  that was already covered by a view (trust > threshold).

**Additional Details** — Priority: High · Complexity: Medium · Dependencies: — · Assumptions: `bake_from_multiview` exposes the trust/coverage map and `extract_textiles` has populated `tex_normal`/`tex_grid`.

### Requirement 2: Automatic fill-camera placement
**User Story:** As the pipeline, I want fill cameras aimed at the uncovered
surfaces, so that those texels become camera-visible within the 75° gate.

**Acceptance Criteria (EARS)**
- WHEN a non-empty gap mask exists THEN the gap-fill stage SHALL derive a set of
  fill cameras (elev, azim) whose view axes face the uncovered texels' normals.
- WHILE selecting fill cameras THE gap-fill stage SHALL prefer cameras that cover
  the most still-uncovered gap texels first (greedy / cluster-by-normal), so few
  cameras cover many gaps.
- WHERE a configured camera cap is set THE gap-fill stage SHALL place at most that
  many fill cameras and SHALL stop early when an additional camera would newly
  cover fewer than a small epsilon of gap texels.
- IF the number of gap clusters exceeds the cap THEN the gap-fill stage SHALL keep
  the highest-coverage cameras and SHALL log the number of gap texels left
  uncovered.

**Additional Details** — Priority: High · Complexity: High · Dependencies: R1 · Assumptions: arbitrary (elev,azim) cameras render correctly (confirmed: `render_geometry_at`, `render_normal/position/alpha`).

### Requirement 3: hyface auto gap-fill
**User Story:** As a user running per-face AI paint, I want oblique walls textured
automatically, so that the output has no blank oblique patches.

**Acceptance Criteria (EARS)**
- WHEN `_run_hyface` finishes its standard view set THEN the system SHALL run the
  gap-fill stage by default.
- WHEN a fill camera needs a reference THEN the system SHALL resolve it via this
  ladder: (1) gpt-image-2 / Gemini synth from the gap camera's geometry render plus
  the nearest/adjacent face references (`render_geometry_at` + `_openai_paint_view`);
  (2) if no image-model key is available, reuse the nearest already-painted face as
  the reference; (3) only if neither is possible, skip that camera.
- IF a fill camera is skipped THEN the system SHALL continue (no crash) and SHALL
  leave its texels to the base bake / inpaint.
- WHEN gap-fill produces fill views THEN the system SHALL bake them into the same
  shared UV texture as the face views, preserving albedo-only matte output.

**Additional Details** — Priority: High · Complexity: High · Dependencies: R1, R2, R5 · Assumptions: `paint_faces` accepts arbitrary (ref,elev,azim,weight) specs (confirmed).

### Requirement 4: reface auto gap-fill
**User Story:** As a user refacing an already-textured mesh, I want oblique walls
covered automatically, so that a reface pass leaves no smeared oblique patches.

**Acceptance Criteria (EARS)**
- WHEN `_run_reface` finishes the user-selected face pass THEN the system SHALL run
  the gap-fill stage by default against the resulting textured mesh.
- WHEN a reface gap camera needs a reference AND the user provided references THEN
  the system SHALL render the current textured mesh at that camera
  (`render_textured_view`) and restyle it toward those references
  (`restyle_to_references`), matching the existing reface flow.
- WHEN a reface gap camera needs a reference AND the user provided NO references
  THEN the system SHALL resolve it via the R3 ladder (gpt-synth from geometry →
  nearest painted face → skip), because restyling the inpaint-smeared base render
  would not introduce real colour into the gap.
- IF multiple gap clusters require multiple cameras THEN the system SHALL run them
  as sequential reface passes within the one job, each compositing over the prior
  result.
- WHEN gap-fill bakes a camera THE system SHALL composite only the gap texels over
  the existing base texture and SHALL leave all other texels unchanged.

**Additional Details** — Priority: High · Complexity: High · Dependencies: R1, R2, R5 · Assumptions: `worker.reface(elev,azim,...,mask=)` already bakes a masked region over the base (confirmed).

### Requirement 5: Gap-only baking with dilation
**User Story:** As the pipeline, I want to write only the gap (plus a thin border),
so that I never overwrite texels that already look correct.

**Acceptance Criteria (EARS)**
- WHEN baking a fill camera THE gap-fill stage SHALL restrict the written texels to
  the gap mask dilated by a configured small border.
- WHERE a fill camera's painted region overlaps already-covered texels OUTSIDE the
  dilated gap mask THE gap-fill stage SHALL NOT overwrite them.
- WHEN gap texels remain unreachable by any placed camera THEN the gap-fill stage
  SHALL leave the existing inpaint result for those texels (no regression).

**Additional Details** — Priority: High · Complexity: Medium · Dependencies: R1, R2 · Assumptions: a per-texel/region write mask can gate the bake (reface already supports a mask; hyface bake is maskable via the cos/coverage path).

### Requirement 6: Configuration & guardrails
**User Story:** As the operator, I want to tune and bound the gap-fill, so that
runtime and cost stay predictable.

**Acceptance Criteria (EARS)**
- WHERE environment variables follow the existing `os.environ.get(...)` convention
  THE system SHALL expose: a per-mode enable toggle (default on), a max fill-camera
  cap, a gap-dilation size, and the candidate-sampling density.
- IF the enable toggle for a mode is set off THEN that mode SHALL behave exactly as
  before this feature (no gap stage).
- WHEN the camera cap is reached THE system SHALL NOT place additional fill cameras
  regardless of remaining gaps.
- WHEN no env override is provided THE system SHALL apply documented sane defaults.

**Additional Details** — Priority: High · Complexity: Low · Dependencies: R2 · Assumptions: matches `_HYFACE_*` / `_REFACE_*` patterns.

### Requirement 7: Additivity & no regression
**User Story:** As the maintainer, I want the change to be additive, so that
existing modes and the client contract are untouched.

**Acceptance Criteria (EARS)**
- WHEN any non-reface, non-hyface mode runs THEN its behavior and output SHALL be
  identical to before this feature.
- WHEN gap-fill completes THE system SHALL return the result via the existing
  output contract (`{uid}_textured.glb`, `textured_url=/api/files/{name}`, job
  `status`/`progress`/`message` fields).
- IF the gap-fill stage raises an error THEN the system SHALL fall back to the base
  bake result and SHALL still complete the job (gap-fill failure is non-fatal).

**Additional Details** — Priority: High · Complexity: Low · Dependencies: R3, R4 · Assumptions: existing `_set(...)` status plumbing reused.

### Requirement 8: Observability
**User Story:** As the operator, I want to see what gap-fill did, so that I can
verify and tune it.

**Acceptance Criteria (EARS)**
- WHEN the gap-fill stage runs THE system SHALL log the gap-texel count, the number
  of fill cameras placed (with their elev/azim), and the gap-texel count remaining
  after fill.
- WHEN gap-fill updates job progress THE system SHALL emit a human-readable
  `message` consistent with existing `_set(job_id, message=...)` usage.
- WHERE logging occurs THE system SHALL NOT log sensitive data.

**Additional Details** — Priority: Medium · Complexity: Low · Dependencies: R1–R5 · Assumptions: stdout logging matches existing `print("[reface] ...")` / `[server]` style.

---

## Non-Functional Requirements

**Performance**
- WHEN gap-fill runs THEN the number of extra diffusion passes (hyface) or
  gpt-calls + bakes (reface) SHALL be bounded by the camera cap.
- IF the gap mask is empty THEN the stage SHALL add no measurable cost beyond the
  one coverage-mask read.

**Reliability**
- IF gap-fill fails for any reason THEN the job SHALL still complete with the base
  bake result (never worse than current behavior).
- WHEN `OPENAI_API_KEY` / `GEMINI_API_KEY` is unavailable THEN gap-fill SHALL
  degrade gracefully per R3/R4 instead of erroring.

**Usability**
- WHERE the feature is default-on THE system SHALL require no new user action or UI
  to benefit from gap coverage.

**Maintainability**
- WHEN adding the stage THE code SHALL reuse existing helpers (`render_geometry_at`,
  `render_textured_view`, `restyle_to_references`, `_openai_paint_view`,
  `bake_from_multiview` / `back_project`) rather than duplicating bake logic.

---

## Constraints and Assumptions

**Technical Constraints**
- Single GPU, sequential-VRAM mode supported; gap-fill must respect the existing
  `_move_multiview` GPU/CPU swap and `low_vram_mode` cache clears.
- Albedo-only matte output for both modes is preserved.
- Coverage signal comes from the bake trust map; per-texel normals from
  `render.tex_normal`.

**Business Constraints**
- Default-on changes both modes' current behavior and cost — explicitly accepted
  (architecture decision, user-validated).

**Assumptions**
- Arbitrary (elev, azim) cameras render and back-project correctly (precedent: the
  existing tilt/corner/low fills already use non-cardinal angles).
- A camera cap keeps runtime acceptable on messy meshes.

---

## Success Criteria

**Definition of Done**
- The red-circled oblique walls on the screenshot model carry real colour after a
  reface and a hyface run (gap texels show trust > 0, not inpaint).
- All other modes byte-for-byte unchanged in behavior.
- Env toggles off → pre-feature behavior; on → gap coverage with bounded cost.
- A diagnostic probe can dump the gap mask before/after and show it shrank.

**Acceptance Metrics**
- Gap-texel count after fill < gap-texel count before fill on a model with oblique
  walls (measurable via the coverage mask).
- Zero change to non-target modes' outputs in a spot check.

---

## Glossary

| Term | Definition |
|---|---|
| Gap / zero-coverage texel | A valid mesh texel no camera view baked (trust ≤ 1e-8); currently inpaint-filled. |
| Coverage / trust map | The bake's accumulated per-texel cos weight; >1e-8 means covered. |
| Fill camera | An extra (elev, azim) view auto-placed to cover gap texels. |
| reface | Depth-aware single-view re-texture over an already-textured mesh. |
| hyface | Per-face AI paint: 1-view Hunyuan paint per face, baked into one UV texture. |
| Dilation | Expanding the gap mask by a few texels so new paint blends over the inpaint seam. |
