# Constitution: Hand-paint view download/upload + zoom

- **Version**: 1.0.0
- **Ratified**: 2026-06-19
- **Last Amended**: 2026-06-19
- **Scope**: task-scoped

## Purpose

Governs the design of the hand-paint download/upload + zoom feature in the Next.js per-model studio
(`Downloads/3-d-model-generation-workflow`, built into `webapp/webui/`). The feature is a UI-only
addition over a hard-won, validated texture bake. These principles keep it additive, reuse-first, and
non-regressive. Deviations require an explicit entry in the design's Complexity Tracking table.
Derived from the sibling `per-model-studio-backend` constitution, `code-patterns.md.instructions.md`,
`codeReview.instructions.md`, and the approved understanding/requirements.

## Core Principles

### 1. Frontend-Only, No Backend or Contract Change

**Statement**: The feature SHALL be implemented entirely in the Next.js studio source. It SHALL reuse
the existing endpoints (`POST …/faces/:view/render`, `GET …/render-image`, `POST …/faces/:view/handpaint`)
and SHALL NOT add, modify, or remove any backend route, the bake, or the legacy `static/` SPA.

**Rationale**: The bake and job contract are validated; UI work must not risk them or fork behavior.

**Evidence of compliance**: diff touches only `Downloads/3-d-model-generation-workflow/**` (+ rebuilt
`webui/`); `studio.py`, `pipeline.py`, `server.py`, and `static/**` unchanged; studio backend tests
pass untouched.

### 2. Reuse the Existing Bake Path

**Statement**: Upload SHALL produce an RGBA PNG overlay and route it through the SAME callback as
manual Apply (`onApply` → `api.handpaintFace`). The client SHALL NOT add bake, back-projection,
camera, or texture logic; it only composites pixels into a 2D canvas.

**Rationale**: Duplicated bake/camera math diverges and rots; the overlay contract already exists.

**Evidence of compliance**: no new API method required; upload and Apply share one bake call; the
overlay is a plain `dim×dim` RGBA PNG.

### 3. Smallest Sufficient Diff

**Statement**: Changes SHALL be confined to `hand-paint-canvas.tsx` plus minimal prop threading in
`texture-panel.tsx`. No new runtime dependency, no speculative configurability, no abstraction for a
single use. New UI SHALL match existing component conventions (Button variants, lucide icons,
Tailwind classes).

**Rationale**: Karpathy guardrails — surgical edits, match surrounding style, no over-engineering.

**Evidence of compliance**: ≤2 source files changed (plus build artifact); `package.json` deps
unchanged; styling reuses existing tokens.

### 4. Preserve Bake Fidelity Under Zoom

**Statement**: Zoom and pan SHALL be view-only (CSS transform). The drawing buffer SHALL remain
`dim×dim` and the exported overlay SHALL always be the full buffer, regardless of zoom or pan. Brush
input SHALL map to the correct canvas pixel at every zoom level and pan offset.

**Rationale**: Zoom is a convenience; it must never crop, downscale, or misregister what gets baked.

**Evidence of compliance**: `canvas.width/height` never change with zoom; `toBlob` exports the whole
canvas; a stroke at 2×+ zoom bakes under the cursor.

### 5. Fail Visibly, Degrade Safely

**Statement**: Non-image uploads SHALL be ignored (no bake started). Bake failures SHALL surface
through the existing `jobError` banner and SHALL leave the face unchanged. Controls SHALL be disabled
when unavailable (no backdrop, or busy) rather than silently no-op.

**Rationale**: Silent failures hide bugs and confuse users (code-patterns: ambiguous output → safe
default; no silent no-ops).

**Evidence of compliance**: file-type guard before bake; errors flow through `runJob`/`jobError`;
disabled states on Upload/Download/Apply/Clear/paint during busy.

## Additional Constraints

- **Security**: local single-user studio; client-side file handling only; validate file is an image
  before processing; backend already validates the uploaded PNG (`_save_upload_png`) and the view
  allowlist. No secrets, no absolute paths exposed.
- **Performance**: zoom/pan via CSS transform (no re-fetch/re-render of the backdrop); uploaded-image
  resize done once on an offscreen `dim×dim` canvas.
- **Platform**: Next.js 16 / React 19, static export (`output: 'export'`); desktop mouse-wheel target;
  built with pnpm 10 / node 22 and copied to `webapp/webui/`.

## Development Workflow

- Requirements → design → tasks → implementation → tests → review enforce these principles.
- Checks: `pnpm build` must succeed; manual verification of download round-trip, upload bake, and
  zoom stroke-accuracy; backend studio tests (`test_studio_api.py`) remain green (proving no backend
  regression). No automated UI test harness exists in this project.

## Governance

- The constitution supersedes ad-hoc decisions; a violating design without a Complexity Tracking entry
  fails review. Amendments require justification + semver bump.

## Amendment Log

| Version | Date | Change | Author |
|---|---|---|---|
| 1.0.0 | 2026-06-19 | Ratification | Batman |
