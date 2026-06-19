# Requirements: Hand-paint view download/upload + zoom

**Document Information**
- **Feature Name:** Hand-paint view download/upload + zoom
- **Version:** 1.0
- **Date:** 2026-06-19
- **Author:** Batman (with Jose)
- **Stakeholders:** Studio users hand-painting per-face textures; maintainer of the Next.js studio UI
- **Steering:** `.batman/handpaint-upload-zoom/steering/understanding.md` (approved). Foundation
  files (`product.md`/`tech.md`/`structure.md`) intentionally not generated — this is a single-component
  frontend change and the understanding doc carries the needed tech/structure context.

## Introduction

The per-model studio lets users hand-paint touch-ups on a face: a square render of the current face
(the "backdrop") is shown, the user brushes strokes on a transparent canvas, and **Apply** exports an
RGBA overlay that the backend (`paint_overlay`) bakes onto the visible texels of that face.

Two gaps slow real use. First, there is no way to take a face out of the app, edit it in a real image
editor, and bring it back — users can only paint with the in-app brush. Second, the canvas is fixed at
a small size, so fine detail is hard to place. This feature adds (1) **Download** of the current face
render and **Upload** of an externally-edited image that bakes straight onto the face, and (2)
**zoom + pan** of the paint surface for precise work.

**Feature Summary**
On the Hand Paint surface, add Download (save the current face render), Upload (bake an uploaded image
onto the face immediately), and mouse-wheel zoom with drag-pan.

**Business Value**
Faster, higher-quality face texturing: users leverage external tools (Photoshop, etc.) for detailed
edits and place brush strokes precisely, without any new backend or GPU cost.

**Scope**
- **Included:** Download current backdrop; Upload → immediate bake via the existing handpaint endpoint;
  wheel-zoom + drag-pan on the paint surface; correct brush mapping and full-resolution bake under zoom.
- **Excluded:** Any backend/API/GPU change; new endpoints; editing the uploaded image in-app before
  baking (upload bakes immediately); downloading the reference or a strokes-composite; changes to the
  old `static/` SPA; multi-layer/undo history beyond the existing Clear and texture history.

---

## Requirements

### Requirement 1: Download the current face render

**User Story:** As a studio user, I want to download the current face image while hand-painting, so that
I can edit it in an external image editor and bring it back.

**Acceptance Criteria (EARS)**
- WHEN the Hand Paint surface has a loaded backdrop AND the user activates Download, THEN the Hand
  Paint UI SHALL save the current face render (the backdrop PNG) to the user's device.
- WHERE the saved file is named, the Hand Paint UI SHALL use a descriptive, view-identifying name
  (e.g. `handpaint-<view>.png`).
- IF the backdrop has not finished rendering (no backdrop yet), THEN the Hand Paint UI SHALL keep the
  Download control disabled.
- WHEN the saved file is the backdrop, THEN it SHALL be the same square image used as the paint
  backdrop, so re-uploading it aligns 1:1 with the bake.

**Additional Details**
- **Priority:** High · **Complexity:** Low
- **Dependencies:** Existing `api.faceRenderUrl` / `…/render-image` backdrop.
- **Assumptions:** The backdrop URL is reachable from the browser (it is served same-origin).

### Requirement 2: Upload an image and bake it as hand-paint

**User Story:** As a studio user, I want to upload an image of a view and have it applied as if I had
hand-painted it, so that I can paint a whole face from an externally-edited image in one step.

**Acceptance Criteria (EARS)**
- WHEN the user selects an image file via the Upload control, THEN the Hand Paint UI SHALL bake that
  image onto the current face immediately, using the existing handpaint bake (POST
  `…/faces/:view/handpaint`), with no extra confirmation step.
- WHERE the uploaded image is prepared for baking, the Hand Paint UI SHALL fit it into a square buffer
  matching the backdrop using **contain** scaling (letterboxed, no distortion), so the painted area
  aligns to the face and a downloaded backdrop round-trips 1:1.
- IF the selected file is not an image, THEN the Hand Paint UI SHALL ignore it and SHALL NOT start a
  bake.
- IF there is no textured mesh for the model (backend returns 400), THEN the Hand Paint UI SHALL surface
  the error to the user and SHALL leave the face unchanged.
- WHILE an upload bake is in progress, the Hand Paint UI SHALL show the busy state and SHALL disable the
  paint, Apply, Upload, Download, and Clear controls.
- WHEN the upload bake job completes, THEN the studio SHALL reflect the repainted face (same completion
  path as the existing Apply).

**Additional Details**
- **Priority:** High · **Complexity:** Low-Medium
- **Dependencies:** Existing `api.handpaintFace`, `runJob` polling, and `paint_overlay` bake (which
  composites only camera-facing visible texels, so a full-opaque image repaints only the visible face).
- **Assumptions:** Letterbox (transparent) areas bake nothing because their alpha is 0.

### Requirement 3: Zoom and pan the paint surface

**User Story:** As a studio user, I want to zoom into the face and pan around while hand-painting, so
that I can place strokes precisely on small details.

**Acceptance Criteria (EARS)**
- WHEN the user scrolls the mouse wheel over the paint surface, THEN the Hand Paint UI SHALL zoom the
  backdrop and canvas together, centered on the cursor, within a bounded zoom range (min = fit, a
  defined max).
- WHILE zoomed in, WHEN the user performs the pan gesture (a gesture distinct from painting), THEN the
  Hand Paint UI SHALL pan the view and SHALL NOT paint a stroke during that gesture.
- WHILE the user paints with the normal brush gesture at any zoom level, THEN each stroke SHALL be
  recorded at the canvas pixel under the cursor (brush mapping stays correct under zoom and pan).
- WHEN the user bakes (Apply or Upload) at any zoom level, THEN the exported overlay SHALL be the full
  `dim×dim` canvas buffer, so zoom does not reduce bake resolution or crop the face.
- WHERE a reset affordance is provided, WHEN the user activates it, THEN the Hand Paint UI SHALL return
  zoom to fit and pan to centered.
- IF the view is zoomed out to fit (zoom = min), THEN the pan offset SHALL be clamped so the face stays
  within the frame.

**Additional Details**
- **Priority:** High · **Complexity:** Medium
- **Dependencies:** None beyond the canvas component.
- **Assumptions:** A CSS transform (or equivalent) on the backdrop+canvas wrapper keeps the canvas'
  on-screen rect consistent so the existing rect-based `ptr()` mapping remains correct; the pan gesture
  is decided in Design (default: Space-hold or middle/right-mouse drag, plus a Pan toggle).

---

## Non-Functional Requirements

**Performance**
- WHEN zooming or panning, THEN the Hand Paint UI SHALL update smoothly via CSS transform without
  re-rendering or re-fetching the backdrop image.
- WHEN preparing an uploaded image for bake, THEN the client-side resize SHALL complete without a
  perceptible freeze for typical image sizes.

**Usability**
- WHEN the Hand Paint surface is shown, THEN the Download and Upload controls SHALL be visible and
  labeled, and the zoom/pan interactions SHALL be discoverable (e.g. a short hint and/or a Pan/Reset
  control).
- IF a control is unavailable (no backdrop, or busy), THEN the Hand Paint UI SHALL disable it rather
  than fail silently.

**Reliability / Error Handling**
- IF an upload bake request fails, THEN the Hand Paint UI SHALL show the error and SHALL leave the face
  in its prior state (no partial UI change).
- WHEN the bake completes and the backend drops the stale backdrop, THEN the next open of the face
  SHALL re-render a fresh backdrop (existing behavior, preserved).

**Compatibility**
- WHERE changes are made, the Hand Paint UI SHALL change only the Next.js studio source; the legacy
  `static/` SPA and all backend endpoints SHALL remain unchanged.

---

## Constraints and Assumptions

**Technical Constraints**
- Frontend-only: React/Next.js source at `Downloads/3-d-model-generation-workflow`, static-exported
  (`output: 'export'`) and copied into `webapp/webui/`. Changes require `pnpm build` + copy.
- Must reuse the existing endpoints `…/faces/:view/render`, `…/render-image`, `…/faces/:view/handpaint`.
- The bake resizes the overlay to the render size and bakes `alpha > 0.5`; the client must produce an
  RGBA PNG whose opaque area is the intended paint.

**Business Constraints**
- No GPU/cost increase; no backend deploy.

**Assumptions**
- Users typically download the square backdrop, edit it, and re-upload (the 1:1 path); arbitrary
  aspect-ratio uploads are handled by contain-fit without guaranteeing perfect registration.
- The studio is used on desktop with a mouse wheel; touch zoom is not a target for this iteration.

---

## Success Criteria

**Definition of Done**
- All acceptance criteria for Requirements 1–3 met.
- Non-functional requirements satisfied (smooth zoom/pan, disabled-when-unavailable controls, error
  surfaced on failure).
- `pnpm build` succeeds and `out/` is copied to `webapp/webui/`; the feature works in the served studio.
- No backend or `static/` change; existing studio backend tests still pass unchanged.

**Acceptance Metrics**
- Download produces the square face PNG; re-uploading it bakes the face with correct registration.
- Upload of an edited backdrop visibly repaints the face after the job completes.
- At 2×+ zoom, a stroke lands under the cursor (within ~1 brush radius), and the baked result matches
  the painted location.

---

## Glossary

| Term | Definition |
|---|---|
| Backdrop | The square PNG render of the face's current appearance, shown under the paint canvas (`…/render-image`). |
| Overlay | The exported RGBA PNG sent to the bake; opaque pixels (alpha > 0.5) are painted onto the face. |
| Bake / paint_overlay | Backend step that composites the overlay onto the visible, camera-facing texels of the face. |
| Reface / AI paint | Other per-face methods (model-driven); out of scope here. |
| Contain fit | Scaling an image to fit inside a box without distortion, leaving transparent letterbox margins. |
| dim | The canvas pixel buffer size (square, backdrop natural width clamped 384–1024). |

---

## Requirements Review Checklist

- [x] Each requirement has a user story and EARS acceptance criteria
- [x] Positive, negative, and busy/error scenarios covered
- [x] Non-functional aspects (perf, usability, reliability, compatibility) addressed
- [x] Scope explicitly bounds out backend/API/static-SPA changes
- [x] Decisions from the grill-me pass encoded (download = backdrop; upload = immediate bake, contain
      fit; zoom = wheel + drag-pan; I build + copy)
- [x] Traceable, numbered requirements with dependencies and assumptions
