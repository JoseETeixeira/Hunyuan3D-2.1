# Constitution: Oblique-Angle Coverage (auto gap-fill)

- **Version**: 1.0.0
- **Ratified**: 2026-06-17
- **Last Amended**: 2026-06-17
- **Scope**: task-scoped

## Purpose

Governs the design of the auto coverage-gap fill stage added to the `reface` and
`hyface` texture modes. These principles keep the change additive, cheap to run,
safe under failure, and consistent with the existing webapp/hy3dpaint conventions.
Any deviation must be recorded in the design's Complexity Tracking table.

## Core Principles

### 1. Additive, Non-Regressive

**Statement**: The change SHALL add a new stage/branch only. Non-target modes
(`hunyuan`, `projection`, `gptproject`, `mvadapter`, `mvgpt`) and the client
contract (`{uid}_textured.glb`, `/api/files/{name}`, job status fields) SHALL be
behaviorally identical to before.

**Rationale**: Six modes share one dispatcher; a regression in shared code breaks
unrelated flows. The repo's whole mode design is additive branches off
`_run_texture`.

**Evidence of compliance**: No edits to non-target `_run_*` paths; existing helper
signatures unchanged or extended with defaulted params; a spot check shows an
unrelated mode's output unchanged.

### 2. Reuse Renderer/Bake Primitives — No Duplicated Bake Math

**Statement**: The stage SHALL reuse `back_project`, `fast_bake_texture`,
`render_*`, `tex_normal/tex_grid/texture_indices`, `_align_photo`,
`render_geometry_at`, `render_textured_view`, `restyle_to_references`,
`_openai_paint_view`. It SHALL NOT reimplement projection, baking, or UV math.

**Rationale**: The renderer encodes hard-won camera/mirror/UV conventions; a parallel
implementation drifts and silently mis-bakes.

**Evidence of compliance**: Gap baking goes through `back_project` + masked composite
(the reface pattern); no new rasterization or cosine code.

### 3. Best-Effort, Non-Fatal Stage

**Statement**: The gap-fill stage SHALL be isolated from the functional bake. IF it
raises, the job SHALL still complete with the base bake result. The stage SHALL NOT
sit inside a try-block that would discard the already-baked texture on failure.

**Rationale**: Gap-fill is a quality enhancement, not a load-bearing step. Team rule:
best-effort side-effects must not break the critical path.

**Evidence of compliance**: Stage wrapped in try/except that logs and returns the
pre-gap textured result; failure path tested.

### 4. Bounded Cost

**Statement**: Fill cameras SHALL be capped (env-tunable) with early-stop when an
added camera covers < epsilon new gap texels. Image-model fallbacks (gpt → gemini →
nearest face → skip) SHALL be bounded and loop-free.

**Rationale**: Each fill view = a diffusion pass (hyface) or gpt call + bake (reface).
Default-on means cost must be predictable on messy meshes. Team rule: bound
provider/model retries + fallbacks.

**Evidence of compliance**: A hard cap constant; an early-stop check; no unbounded
while loop over clusters.

### 5. Externalized, Reversible Config

**Statement**: All tuning SHALL use `os.environ.get("UPPER_SNAKE", default)` matching
the `_HYFACE_*` / `_REFACE_*` style, with documented sane defaults. A per-mode enable
flag (default on) SHALL fully revert to pre-feature behavior when off.

**Rationale**: Matches the codebase's per-mode knob convention and lets the operator
disable the stage instantly without a code change.

**Evidence of compliance**: New `GAPFILL_*` constants read at import; toggling off
yields byte-identical legacy behavior.

### 6. Observability Without Sensitive Data

**Statement**: The stage SHALL log gap-texel count, fill cameras placed (elev/azim),
and remaining gap count, via the existing `print("[gapfill] ...")` / `_set(job_id,
message=...)` style. It SHALL NOT log secrets, API keys, or raw file system secrets.

**Rationale**: Default-on behavior needs to be auditable/tunable; logs are the only
window. Team rule: structured operational logs, no sensitive data.

**Evidence of compliance**: Diagnostic line per run with the three counts; no key/secret
in any log.

## Additional Constraints

- **Platform**: Python 3 + PyTorch + CUDA; single GPU; sequential-VRAM mode must keep
  working (`_move_multiview` swap, `low_vram_mode` empty_cache respected).
- **Output**: Albedo-only matte preserved for both modes (`_force_matte`).
- **Performance**: Empty gap mask → no measurable added cost beyond one coverage read.
  Non-empty → bounded by the camera cap.
- **Security**: No new external input surface beyond existing endpoints; no secret in logs.

## Development Workflow

- Requirements (EARS) → this constitution gates design → tasks trace to requirement
  IDs → a diagnostic probe + visual check verify gap shrinkage → code review against
  `codeReview.instructions.md`.
- Manual gate: user approval at each Batman phase.

## Governance

- The constitution supersedes ad-hoc decisions. A design violating a principle without
  a Complexity Tracking entry fails review.
- Amendments require explicit justification, an in-flight migration note, and a semver
  bump (Major: principle removed/replaced; Minor: principle added/expanded; Patch:
  clarification).

## Amendment Log

| Version | Date | Change | Author |
|---|---|---|---|
| 1.0.0 | 2026-06-17 | Ratification | Batman |
