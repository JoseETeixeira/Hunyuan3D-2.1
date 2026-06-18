# Requirements: Per-Model 3D Studio Backend

**Document Information**
- **Feature Name:** Per-Model 3D Studio Backend (wire the Next.js studio to the webapp)
- **Version:** 1.0
- **Date:** 2026-06-17
- **Author:** Batman (with Josee)
- **Stakeholders:** Project owner / sole operator; future maintainers of `webapp`.

## Introduction

The user rebuilt the frontend into a per-model studio (Next.js, `Downloads/3-d-model-generation-workflow`)
that currently runs on an in-memory mock. The backend (`Hunyuan3D-2.1/webapp/server.py` +
`pipeline.py`) already implements the heavy lifting — Hunyuan shape generation, per-face AI paint
(`hyface`), depth-aware single-face reface, gpt-image-2 / Gemini synthesis, and coverage gap-fill —
but exposes a flat, per-run, job-keyed API with no durable model identity and no mesh-free reference
generation. This feature adds the per-model persistence and REST layer the new frontend expects, a
staged mesh-free reference generator, the texture orchestration the studio flow needs, and removes
every texture mode that is not per-face AI paint or reface.

The work is grounded in the approved understanding
(`.batman/per-model-studio-backend/steering/understanding.md`) and the exact frontend contract in
`lib/api.ts`, `lib/types.ts`, `lib/views.ts`, and `lib/mock-backend.ts`.

## Feature Summary

A per-model backend that persists each model's references, mesh, and texture, generates the ten
orthographic reference views from a seed image via gpt-image-2 along an imperative dependency graph,
builds and refines textures with per-face AI paint + reface, and serves it all through the REST
contract the new frontend already calls — with the legacy texture modes removed.

## Business Value

The user gets a working, reusable studio: name a model once, generate/approve references, texture
it, refine faces, and come back later without re-uploading. Removing unused modes shrinks the
maintenance surface to the two modes the product actually uses.

## Scope

**Included**
- Per-model registry + persistence (durable across process restart).
- The full `/api/models/*`, reference, texture, face-edit, job, and download contract from
  `lib/api.ts`.
- Mesh-free staged reference generation (gpt-image-2) with per-view prompts and a tweak prompt.
- Texture base = mesh + per-face paint over all approved references; reface per view using each
  view's approved reference.
- Removal of `hunyuan`, `projection`, `gptproject`, `mvadapter`, `mvgpt` from API/UI/dispatch.
- Serving the Next.js app same-origin as `/api/*` (mechanism chosen in Design).

**Excluded**
- Auth / multi-user / cloud scaling.
- New shape model or TRELLIS.2 wiring.
- Any texture mode beyond per-face AI paint and reface.
- Changes to the approved gap-fill behavior beyond what mode removal requires.

---

## Requirements

### Requirement 1: Per-Model Registry and Persistence
**User Story:** As a creator, I want each model (its references, mesh, and texture) saved under a
name, so that I can reuse it across sessions without re-uploading anything.

**Acceptance Criteria (EARS)**
- WHEN the user creates a model via `POST /api/models` with a `name` and optional `seed_image`,
  THEN the Backend SHALL persist a new model record and return the full `Model` aggregate.
- WHEN the user requests `GET /api/models`, THEN the Backend SHALL return a `ModelSummary[]`
  including `id`, `name`, `previewUrl`, `textured`, and `updatedAt`, newest first.
- WHEN the user requests `GET /api/models/:id`, THEN the Backend SHALL return the full `Model`
  (references map of 10 views, faces map of 10 views, `seedImageUrl`, `meshUrl`, `texturedUrl`,
  `textureStage`, `createdAt`, `updatedAt`).
- WHEN the user renames via `PATCH /api/models/:id` with `{ name }`, THEN the Backend SHALL update
  the name and return the updated `Model`.
- WHEN the user deletes via `DELETE /api/models/:id`, THEN the Backend SHALL remove the model record
  and its artifacts and return HTTP 204.
- IF the Backend process restarts, THEN the Backend SHALL still list and load every previously
  persisted model with its references, mesh, texture, and stage intact.
- IF a requested model id does not exist, THEN the Backend SHALL return HTTP 404 with a JSON body
  carrying a `detail` or `error` message.

**Additional Details**
- **Priority:** High · **Complexity:** High
- **Dependencies:** none (foundational).
- **Assumptions:** persistence is local filesystem under `HY3D_OUTPUT_DIR`; concrete mechanism
  (per-model JSON sidecar + index vs SQLite) is a Design decision.

### Requirement 2: Model Aggregate and Job→Model Contract
**User Story:** As the frontend, I want every async operation to return a job I can poll and, on
completion, the full updated model, so that the UI refreshes without extra fetches.

**Acceptance Criteria (EARS)**
- WHEN any async endpoint (`references/:view/generate`, `texture/base`, `texture/reface/:view`,
  `faces/:view/edit`) is called, THEN the Backend SHALL respond with a `Job` containing a string
  `id` and a numeric `progress` immediately.
- WHILE a job is running, the Backend SHALL report `status` in {`queued`, `processing`} and a
  monotonically non-decreasing `progress` in 0..100 via `GET /api/jobs/:id`.
- WHEN a job completes, THEN the Backend SHALL set `status: "completed"`, `progress: 100`, and embed
  the full updated `Model` in `job.model`.
- IF a job fails, THEN the Backend SHALL set `status: "failed"` and a human-readable `error` string,
  and SHALL leave the model's prior artifacts intact.
- WHEN a synchronous endpoint (`createModel`, `getModel`, `renameModel`, `uploadReference`,
  `approveReference`) is called, THEN the Backend SHALL return the `Model` (or `ModelSummary[]`)
  directly, not a job.

**Additional Details**
- **Priority:** High · **Complexity:** Medium
- **Dependencies:** R1.
- **Assumptions:** the frontend only refreshes on `job.model` at completion (`lib/use-job.ts:27-29`).

### Requirement 3: Seed Image and Front Reference Seeding
**User Story:** As a creator, I want to upload one image (any view) to seed a model, so that the
front reference can be generated or set from it.

**Acceptance Criteria (EARS)**
- WHEN `POST /api/models` includes a `seed_image` file, THEN the Backend SHALL store it and set
  `seedImageUrl` on the model.
- WHEN the user uploads a front image via `POST /api/models/:id/references/front/upload`, THEN the
  Backend SHALL store it as the `front` reference, set its `source` to `uploaded`, set its status to
  `approved`, and return the updated `Model`.
- IF front generation is requested and a seed image exists, THEN the Backend SHALL use the seed
  image as the sole input for generating the `front` view.
- IF neither a seed image nor an uploaded front exists when front generation is requested, THEN the
  Backend SHALL fail the job with a clear `error`.

**Additional Details**
- **Priority:** High · **Complexity:** Low
- **Dependencies:** R1, R4.

### Requirement 4: Staged Reference Generation with Imperative Dependency Graph
**User Story:** As a creator, I want the ten reference views generated in dependency order from my
seed image, so that each view is consistent with the ones it derives from.

**Acceptance Criteria (EARS)**
- WHEN `POST /api/models/:id/references/:view/generate` is called for `front`, THEN the Backend SHALL
  generate it from the seed image only.
- WHEN generation is requested for a cardinal view (`left`, `right`, `top`, `bottom`, `back`), THEN
  the Backend SHALL condition it on the approved `front` reference plus the seed image.
- WHEN generation is requested for a front corner (`front-left`, `front-right`), THEN the Backend
  SHALL condition it on the approved `front`, `left`, `right`, and `top` references.
- WHEN generation is requested for a back corner (`back-left`, `back-right`), THEN the Backend SHALL
  condition it on the approved `back`, `left`, `right`, and `top` references.
- IF any input view required by `VIEW_INPUTS[view]` is not yet `approved`, THEN the Backend SHALL
  reject the generation request (HTTP 409 / failed job with a clear `error`) rather than generate
  from incomplete inputs.
- WHEN a view is generated, THEN the Backend SHALL set that reference's `status` to `pending`,
  `source` to `generated`, and populate its `url`.
- WHILE a view's `status` is `generating`, the Backend SHALL reject a second concurrent generate for
  the same view.

**Additional Details**
- **Priority:** High · **Complexity:** High
- **Dependencies:** R3, R5.
- **Assumptions:** dependency graph is exactly `lib/views.ts:VIEW_INPUTS`; backend enforces it
  (defense in depth — the UI also gates).

### Requirement 5: Per-View Prompts and Tweak Editing
**User Story:** As a creator, I want each generated view to be a correct orthographic, cartoonish
depiction showing only that view's faces, and I want to tweak it with an instruction, so that I get
usable, consistent references.

**Acceptance Criteria (EARS)**
- WHERE a reference view is generated, the Backend SHALL use a per-view prompt that specifies the
  orthographic head-on framing for that view (e.g. "right = the front rotated 90° clockwise,
  orthographic, no perspective"), the 3D cartoonish style, and the rule that only the faces visible
  in that view are shown.
- WHEN `references/:view/generate` includes an `edit_prompt`, THEN the Backend SHALL incorporate that
  instruction into the (re)generation and record it on the reference (`editPrompt`).
- WHEN a view is regenerated with an `edit_prompt`, THEN the Backend SHALL replace that view's image
  and set its `status` back to `pending` for re-approval.
- WHERE corner views are generated, the Backend SHALL show only the faces visible from that 3/4
  corner per the established face-visibility map (`ADJ`).

**Additional Details**
- **Priority:** High · **Complexity:** High
- **Dependencies:** R4.
- **Assumptions:** generation uses `edit_image` (gpt-image-2 default, Gemini fallback); orthographic
  fidelity is prompt-enforced and may need mirror/consistency safeguards.

### Requirement 6: Reference Lifecycle (Approve / Upload / Replace)
**User Story:** As a creator, I want to approve, replace, or upload a custom image for any view, so
that I control exactly which references drive texturing.

**Acceptance Criteria (EARS)**
- WHEN `POST /api/models/:id/references/:view/approve` is called and that view has a `url`, THEN the
  Backend SHALL set its `status` to `approved` and return the updated `Model`.
- IF approve is called for a view that is `empty` or `generating`, THEN the Backend SHALL reject it
  with a clear error and not change status.
- WHEN `POST /api/models/:id/references/:view/upload` is called with an image, THEN the Backend SHALL
  store it, set `source` to `uploaded`, set `status` to `approved`, and return the updated `Model`.
- WHEN a reference is uploaded or regenerated, THEN the Backend SHALL update the model's `updatedAt`.

**Additional Details**
- **Priority:** High · **Complexity:** Low
- **Dependencies:** R4.

### Requirement 7: Texture Base — Mesh + Per-Face Paint over All Approved References
**User Story:** As a creator, I want one action to build the mesh and paint it with per-face AI paint
using all my approved references, so that I get a complete base texture to refine.

**Acceptance Criteria (EARS)**
- IF not all ten references are `approved`, THEN the Backend SHALL reject `POST /api/models/:id/texture/base`
  with a clear error.
- WHEN `texture/base` is called with a `MeshConfig` (`inference_steps`, `guidance_scale`,
  `octree_resolution`, `texture_views`, `seed`, `mesh_faces`), THEN the Backend SHALL generate the
  mesh with those parameters and then run per-face AI paint (`hyface`) using **all approved
  references** as per-face view specs.
- WHEN the base job completes, THEN the Backend SHALL populate both `meshUrl` (untextured shape GLB)
  and `texturedUrl` (textured GLB) so every approved face carries base paint, and SHALL set each
  painted face's `status` to `done` with `mode` `paint`.
- WHEN the base job completes, THEN the Backend SHALL set `textureStage` to `complete` (base per-face
  paint already yields the final usable texture; reface is OPTIONAL refinement) and embed the updated
  `Model` in `job.model`.
- WHERE per-face paint runs, the Backend SHALL preserve the existing albedo-only matte output and the
  best-effort, non-fatal coverage gap-fill behavior.

**Additional Details**
- **Priority:** High · **Complexity:** High
- **Dependencies:** R2, R6, R10.
- **Assumptions:** base uses all approved refs (user-confirmed 2026-06-17), overriding the mock's
  front-only behavior.

### Requirement 8: Reface Per View Using the Approved Reference
**User Story:** As a creator, I want to refine each face by refacing it with its own approved
reference, so that each face is sharp and correct without re-uploading.

**Acceptance Criteria (EARS)**
- WHEN `POST /api/models/:id/texture/reface/:view` is called, THEN the Backend SHALL reface that face
  of the model's textured mesh using that view's **approved reference** as the reference image.
- WHEN `reface/:view` includes an `edit_prompt`, THEN the Backend SHALL incorporate it into the
  face's restyle step.
- WHERE reface runs, the Backend SHALL repaint only the nearest (foreground) depth band and preserve
  the rest of the existing texture, and SHALL output albedo-only matte.
- WHEN a reface job completes, THEN the Backend SHALL set that face's `status` to `done` with `mode`
  `reface`, update `texturedUrl`, and embed the updated `Model`.
- IF the model has no textured mesh yet (base not run), THEN the Backend SHALL reject reface with a
  clear error.

**Additional Details**
- **Priority:** High · **Complexity:** High
- **Dependencies:** R7.
- **Assumptions:** reface is OPTIONAL refinement (user-confirmed 2026-06-18). The frontend's
  Reface-all / per-face loop calls this endpoint per view; no batch endpoint required.

### Requirement 9: Per-Face Edit (Step 4)
**User Story:** As a creator, once texturing is complete I want to edit any individual face by reface
or per-face AI paint, so that I can fix any face at will.

**Acceptance Criteria (EARS)**
- WHEN `POST /api/models/:id/faces/:view/edit` is called with `mode: "reface"`, THEN the Backend
  SHALL reface that single face (R8 behavior).
- WHEN `faces/:view/edit` is called with `mode: "paint"`, THEN the Backend SHALL re-run per-face AI
  paint for that single face using its approved reference.
- IF `faces/:view/edit` includes an optional `image` file, THEN the Backend SHALL use that image as
  the face's reference instead of the stored approved reference.
- WHEN `faces/:view/edit` includes an `edit_prompt`, THEN the Backend SHALL incorporate it.
- WHEN the edit completes, THEN the Backend SHALL update the face's `status`/`mode` and embed the
  updated `Model`.

**Additional Details**
- **Priority:** Medium · **Complexity:** Medium
- **Dependencies:** R7, R8.
- **Assumptions:** the current UI never sends `image`, but the endpoint accepts it (forward-compat).

### Requirement 10: Texture-Stage State Machine
**User Story:** As the frontend, I want a consistent `textureStage` progression, so that the UI shows
the right step, badges, and affordances.

**Acceptance Criteria (EARS)**
- WHEN no texture exists, THEN the Backend SHALL report `textureStage: "none"`.
- WHILE the base job runs, the Backend SHALL report `textureStage: "base-running"`.
- WHEN the base job completes, THEN the Backend SHALL report `textureStage: "complete"` and mark
  every painted face `done` with `mode` `paint` (base per-face paint is the final usable texture;
  reface is OPTIONAL).
- WHILE an optional reface/per-face-paint edit job runs, the Backend MAY report
  `textureStage: "refacing"` and SHALL return to `"complete"` when it finishes.
- WHEN a face is refaced, THEN the Backend SHALL set that face's `mode` to `reface` and keep
  `textureStage: "complete"`.
- WHERE per-face editing requires `textureStage === "complete"` in the UI, the Backend SHALL reach
  `complete` immediately after base so optional reface/paint edits are available right away.

**Additional Details**
- **Priority:** High · **Complexity:** Low
- **Dependencies:** R7, R8; resolves Open Question 1 in understanding.md (reface is OPTIONAL,
  user-confirmed 2026-06-18). Note: the frontend mock's `base-done → refacing → complete` path
  becomes `base-running → complete`; intermediate stages remain valid but are not required.

### Requirement 11: Multi-Format Download
**User Story:** As a creator, I want to download my model as GLB, FBX, or .blend, so that I can use it
in my target tools.

**Acceptance Criteria (EARS)**
- WHEN `GET /api/models/:id/download/:fmt` is called with `fmt` in {`glb`, `fbx`, `blend`}, THEN the
  Backend SHALL return the textured model in that format (converting via Blender for fbx/blend).
- IF the requested model has no textured mesh, THEN the Backend SHALL serve the untextured shape GLB
  (or 404 if neither exists), consistent with current behavior.
- IF `fmt` is not one of the three, THEN the Backend SHALL return HTTP 400.

**Additional Details**
- **Priority:** Medium · **Complexity:** Low
- **Dependencies:** R7. **Assumptions:** Blender is installed (`BLENDER_BIN`); to be confirmed.

### Requirement 12: Remove Non-Target Texture Modes
**User Story:** As a maintainer, I want only per-face AI paint and reface to remain, so that the
codebase and UI carry no unused texture modes.

**Acceptance Criteria (EARS)**
- WHEN the Backend dispatches texturing, THEN it SHALL support only `hyface` (per-face AI paint) and
  `reface`, with `hyface` as the default.
- WHERE `hunyuan`, `projection`, `gptproject`, `mvadapter`, or `mvgpt` were previously selectable,
  the Backend SHALL no longer expose or dispatch them, and their dedicated endpoints (e.g.
  `/api/jobs/:id/resume`) and the `mvadapter` health flag SHALL be removed.
- WHERE shared helpers are used by kept modes (`image_edit.py`, `gen_transfer.py`,
  `blender_project.py`, the `gpt_angles` field, `TextureWorker` methods), the Backend SHALL retain
  them.
- WHEN a removed mode or endpoint is requested, THEN the Backend SHALL return HTTP 404/400, not run
  the removed path.
- IF the legacy static UI is retained, THEN it SHALL not offer the removed modes; otherwise it is
  retired in favor of the Next.js app.

**Additional Details**
- **Priority:** High · **Complexity:** Medium
- **Dependencies:** none, but must not break R7/R8.

### Requirement 13: Preserve Existing Functionality
**User Story:** As a creator, I want everything the current frontend/backend did (minus removed
modes) to keep working, so that nothing is lost in the rewire.

**Acceptance Criteria (EARS)**
- WHEN per-face AI paint or reface runs after the rewire, THEN the Backend SHALL produce albedo-only
  matte GLBs equivalent to the pre-rewire behavior.
- WHILE texturing, the Backend SHALL keep the best-effort, non-fatal coverage gap-fill stage.
- WHEN the 3D preview loads, THEN the Backend SHALL serve `meshUrl` and `texturedUrl` as
  model-viewer-loadable GLBs from the same origin.
- WHERE corner references (`fl/fr/bl/br`) and per-side references were supported, the Backend SHALL
  continue to support them through the per-model reference flow.

**Additional Details**
- **Priority:** High · **Complexity:** Medium
- **Dependencies:** R7, R8, R11, R12.

---

## Non-Functional Requirements

**Performance**
- WHERE GPU work is required (shape, per-face paint, reface), the Backend SHALL process jobs on the
  single sequential GPU worker without concurrent GPU execution.
- WHERE reference generation (gpt-image-2) is network-bound and needs no GPU, the Backend SHOULD NOT
  block GPU shape/texture jobs behind reference-generation jobs (execution path chosen in Design).
- WHEN a job runs, THEN the Backend SHALL report progress updates frequently enough for the UI's
  700ms poll to show movement.

**Reliability**
- IF the Backend restarts mid-session, THEN persisted models SHALL remain listable and loadable
  (R1); only in-flight jobs are lost.
- IF a best-effort stage (gap-fill, fbx/blend convert) fails, THEN the Backend SHALL still deliver
  the base textured GLB and SHALL surface the degradation without discarding completed work.

**Security**
- WHERE the studio runs locally, the Backend MAY keep open CORS and no auth, but SHALL NOT log secret
  values (API keys) and SHALL NOT expose absolute filesystem paths in HTTP responses.
- IF neither `OPENAI_API_KEY` nor `GEMINI_API_KEY` is configured when generation is requested, THEN
  the Backend SHALL fail the job with a clear, non-secret error message.

**Usability**
- WHEN an operation fails, THEN the Backend SHALL return a JSON error body with `detail` or `error`
  so the frontend `JobBanner`/error path can display it.
- WHERE the Next.js app issues relative `/api/*` calls, the Backend (or its deployment) SHALL serve
  the app same-origin so fetches and GLB loads succeed.

---

## Constraints and Assumptions

**Technical Constraints**
- Single 16 GB GPU; one sequential worker thread; long jobs block subsequent GPU jobs.
- gpt-image-2/Gemini drift, possible side-view mirroring, and prompt-only orthographic fidelity.
- Albedo-only matte output is the product standard.
- View id mapping: frontend hyphenated (`front-left`) ↔ backend tag (`fl`).

**Business Constraints**
- Keep only per-face AI paint and reface.
- No regression of existing behavior except the removed modes.

**Assumptions**
- Persistence is local-filesystem based.
- Blender is available for fbx/blend export (to confirm).
- The frontend contract in `lib/api.ts`/`lib/types.ts` is authoritative and stable.

---

## Success Criteria

**Definition of Done**
- All Requirement acceptance criteria met.
- End-to-end: create model → generate + approve 10 references → base (mesh + per-face paint over all
  refs, both `meshUrl` and `texturedUrl` populated) → reface pass → `complete` → per-face edit →
  download glb/fbx/blend — all driven by the real backend with `NEXT_PUBLIC_USE_MOCK=false`.
- Models persist across a backend restart.
- Removed modes/endpoints return 404/400; kept modes still produce albedo-matte GLBs; gap-fill still
  runs.

**Acceptance Metrics**
- 10/10 reference views generatable in dependency order from a single seed.
- 100% of frontend `api.*` methods served by real endpoints with matching shapes.
- 0 references silently dropped (correct upload channel per mode).

---

## Glossary

| Term | Definition |
|---|---|
| Model | A named per-model aggregate: 10 references, 10 face states, seed image, mesh, texture, stage. |
| Reference view | One of the 10 orthographic views (front…back-right) used to condition texturing. |
| hyface / per-face AI paint | Paint each face from its own reference, baked into one shared UV. |
| reface | Depth-aware single-face re-texture of an already-textured mesh, foreground only. |
| Seed image | The single uploaded image that seeds front generation. |
| Tweak / edit prompt | A user instruction to refine a generation or face edit. |
| Imperative dependency graph | `VIEW_INPUTS`: which approved views feed each view's generation. |
| Albedo-only matte | Flat-cartoon output with metallic/roughness dropped. |
| Gap-fill | Best-effort auto-painting of oblique/recessed texels no fixed camera covers. |

---

## Requirements Review Checklist

- [x] Each requirement has a User Story and EARS acceptance criteria.
- [x] Functional + non-functional requirements covered.
- [x] Criteria are testable and avoid implementation detail where possible.
- [x] Consistent terminology (glossary).
- [x] Dependencies and assumptions recorded.
- [x] User-confirmed decisions (base paints all refs; reface uses approved ref + is OPTIONAL; model
      `complete` right after base; remove all 5 modes) reflected in R7, R8, R10, R12.
- [ ] Persistence / execution-path / deployment mechanisms deferred to Design.
