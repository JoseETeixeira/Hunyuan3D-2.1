# Constitution: Per-Model 3D Studio Backend

- **Version**: 1.0.0
- **Ratified**: 2026-06-18
- **Last Amended**: 2026-06-18
- **Scope**: task-scoped

## Purpose

Governs the design of the per-model studio backend (the new REST layer, persistence, staged
reference generation, and texture-mode cleanup) over the existing Hunyuan3D-2.1 webapp. Deviations
require an explicit entry in the design's Complexity Tracking table. Principles are derived from the
prior `oblique-angle-coverage` constitution, `code-patterns.md.instructions.md`,
`codeReview.instructions.md`, and the approved understanding/requirements.

## Core Principles

### 1. Additive and Non-Regressive

**Statement**: New behavior SHALL be additive. The two kept texture modes (per-face AI paint /
`hyface`, and `reface`) and their outputs SHALL remain behaviorally equivalent to today (albedo-only
matte GLBs, best-effort gap-fill). Removed modes SHALL be deleted cleanly without altering kept-mode
behavior.

**Rationale**: The texturing pipeline is hard-won and validated; the rewire must not silently change
what already works.

**Evidence of compliance**: hyface/reface produce equivalent GLBs after the change; removed-mode
endpoints 404; shared helpers (`image_edit.py`, `gen_transfer.py`, `blender_project.py`,
`gpt_angles`) retained.

### 2. Reuse Primitives, Don't Duplicate

**Statement**: The new orchestration layer SHALL call existing `TextureWorker` methods
(`generate_shape`, `paint_faces`, `reface`, `fill_coverage_gaps`, `render_*`) and `image_edit`/
`elevations` for generation. It SHALL NOT reimplement bake, render, camera, or image-call logic.

**Rationale**: Duplicated bake/camera math diverges and rots; the camera tables and bake paths are
subtle.

**Evidence of compliance**: no new bake/back-project code; camera angles come from `PROJECTION_CAMS`
+ `HYFACE_CORNER_CAMS`; image calls go through `edit_image`.

### 3. Best-Effort Stages Never Destroy Work

**Statement**: Enhancement/secondary stages (coverage gap-fill, fbx/blend conversion, preview
synthesis) SHALL be isolated from the load-bearing result so their failure never discards a
completed mesh/texture or rolls back a persisted model. Best-effort calls SHALL NOT share a failure
boundary with the functional persist/dispatch.

**Rationale**: A reaction/convert failure must not erase a baked texture or a saved model record
(see code-patterns "best-effort side-effects must not live in functional transactions").

**Evidence of compliance**: gap-fill/convert wrapped in try/except, logged, non-fatal; model JSON is
written only after the functional artifact exists.

### 4. Durable, Path-Safe Source of Truth

**Statement**: Each model's identity, references, faces, mesh, texture, stage, and params SHALL be
persisted so they survive a process restart. Caller-controlled identifiers (model id, view, format)
SHALL be validated against closed vocabularies / strict slug rules, and every resolved file path
SHALL be asserted inside `OUTPUT_DIR`. HTTP responses SHALL expose URLs, never absolute disk paths,
and SHALL NOT log secrets.

**Rationale**: The whole point is reusable models; and caller-controlled paths are an injection
surface (code-patterns §Path & Config Boundaries; codeReview §13).

**Evidence of compliance**: restart test lists/loads models; `UUID_RE`/view-allowlist/format-allowlist
guards; `assert_within(OUTPUT_DIR)`; `_public()`-style stripping of internal paths.

### 5. Closed Vocabularies, Fail Fast

**Statement**: View ids (the 10 canonical views), texture modes (`paint`/`hyface`, `reface`), and
download formats (`glb`/`fbx`/`blend`) SHALL be closed sets. Unknown view/mode/format or an
out-of-order generation (unmet `VIEW_INPUTS` deps) SHALL raise a clear error (HTTP 400/409), never a
silent fallback or a late `FileNotFoundError`.

**Rationale**: Silent fallbacks hide bugs; the dependency graph must be enforced server-side.

**Evidence of compliance**: enums/allowlists for view/mode/format; dependency check before generate;
explicit `detail`/`error` JSON.

### 6. Job→Model Contract Integrity

**Statement**: Every async operation SHALL return a `Job` with `id` + numeric `progress`
immediately, report monotonic progress, and on completion embed the full updated `Model` in
`job.model`. On failure it SHALL set `status:"failed"` + `error` and leave prior artifacts intact.
Sync operations SHALL return the `Model` (or `ModelSummary[]`) directly.

**Rationale**: The frontend only refreshes on `job.model`; breaking this silently freezes the UI.

**Evidence of compliance**: contract tests assert shapes; completion embeds `Model`; failures
preserve artifacts.

### 7. Separate GPU and Network Work

**Statement**: GPU work (shape, per-face paint, reface) SHALL run on the single sequential GPU
worker. Network-bound reference generation (gpt-image-2/Gemini, no GPU) SHALL NOT block the GPU
worker.

**Rationale**: Reference generation is network-bound; routing it through the GPU queue wastes the GPU
and serializes unrelated work.

**Evidence of compliance**: reference-generation jobs run on a separate lane; GPU jobs remain
strictly sequential on the GPU worker.

## Additional Constraints

- **Security**: local single-user; open CORS acceptable; no secret logging; validate all uploads and
  paths; no absolute paths in responses.
- **Performance**: one 16 GB GPU, sequential GPU jobs; per-model load = up to ~10 sequential
  reference generations + 1 base + optional refaces; progress visible to a 700 ms poll.
- **Platform**: Python 3.12 / FastAPI / uvicorn backend; Next.js 16 frontend served same-origin.

## Development Workflow

- Requirements → design → tasks → tests → review enforce these principles.
- Automated checks: endpoint/contract tests (P6), restart-persistence test (P4), removed-mode 404
  test (P1), reference-graph gating test (P5). Manual: PR review against this constitution.

## Governance

- The constitution supersedes ad-hoc decisions; a violating design without a Complexity Tracking
  entry fails review. Amendments require justification + version bump (semver).

## Amendment Log

| Version | Date | Change | Author |
|---|---|---|---|
| 1.0.0 | 2026-06-18 | Ratification | Batman |
