# Understanding: Hand-paint view download/upload + zoom

## User Goal

On the per-model studio's **Hand paint** surface, the user wants to:

1. **Download** a view's image and **upload** an image that is then **applied as if hand-painted** onto that face.
2. **Zoom in** on the view while hand-painting, to paint fine detail.

Both are touch-ups on the existing hand-paint canvas. The intended outcome is a faster round-trip
(edit a face's render in an external editor → re-upload → bake) plus finer control while painting.

## Task Slug

`handpaint-upload-zoom`

## Current Behavior

### Workflow Summary

- The **active studio UI is a separate Next.js app** at
  `C:\Users\josee\Downloads\3-d-model-generation-workflow` (source). Its static export (`out/`) is
  copied verbatim into [webapp/webui/](Hunyuan3D-2.1/webapp/webui/) and served by FastAPI. The
  bundled `static/` SPA ([static/app.js](Hunyuan3D-2.1/webapp/static/app.js)) is the older simple
  UI and has **no** hand-paint surface.
- Hand paint lives in [components/studio/hand-paint-canvas.tsx](Hunyuan3D-2.1/webapp/webui/) (source:
  `Downloads/3-d-model-generation-workflow/components/studio/hand-paint-canvas.tsx`), driven by
  `FaceTile` in `components/studio/texture-panel.tsx`.
- Flow today:
  1. User opens a face tile → picks **Hand paint** method.
  2. `FaceTile.prepareCanvas()` calls `api.renderFace()` (POST `…/faces/:view/render`), polls the
     job, then sets `backdrop = api.faceRenderUrl(...)` (GET `…/faces/:view/render-image`). The
     backdrop is a **square PNG render of the face as it currently looks on the mesh**.
  3. `HandPaintCanvas` shows the backdrop `<img>` (object-contain) with a transparent `<canvas>`
     (internal buffer `dim×dim`, `dim` = backdrop natural width clamped 384–1024) on top. A palette
     is sampled from the reference image. User paints strokes.
  4. **Apply** → `canvas.toBlob()` → RGBA overlay (transparent except strokes) →
     `api.handpaintFace()` (POST `…/faces/:view/handpaint`, multipart `overlay`).
- Backend bake (`studio.py` `_gpu_handpaint` → `pipeline.py` `paint_overlay`): the overlay is
  resized to the render size, `alpha > 0.5` marks strokes, `back_project` composites only the
  painted, **camera-facing, visible** texels over the base texture; everything else is untouched.
  Albedo-only matte, like reface.

### Why This Evidence Answers The Question

- `hand-paint-canvas.tsx`: the exact surface the user is describing — owns the backdrop `<img>`, the
  brush `<canvas>`, pointer→canvas mapping (`ptr()`), and the Apply/Clear export path. Both features
  are implemented here.
- `texture-panel.tsx` (`FaceTile`): owns `backdrop`/`refUrl` state, `prepareCanvas()`,
  `applyHandpaint()`, and the `ImageDialog`/`imageSlot` mount. It supplies the data the canvas
  needs and decides what happens after Apply.
- `lib/api.ts`: the single backend contract. `faceRenderUrl()` is the downloadable face image;
  `handpaintFace()` is the bake call. Tells us no new endpoint is needed for either feature.
- `pipeline.py paint_overlay`: proves a **fully-opaque** uploaded image bakes the whole *visible*
  face (background/off-mesh pixels project to nothing), so "upload an image, apply as hand-paint"
  works through the existing overlay path with **no backend change**.
- `webapp/webui` vs `Downloads/.../out` (identical `index.html`, same build id `8q2OrD2fQkaZlxgKmscuh`):
  proves the served UI is a copy of the Next.js export, so edits require a rebuild + copy.

### Process Distinctions And Terminology

- **Hand paint** (this task) vs **Reface** vs **AI paint**: hand paint bakes a pixel-locked RGBA
  overlay the user drew on the face render (`paint_overlay`, no model call). Reface re-projects a
  reference through the reface model; AI paint regenerates the face with Hunyuan per-face paint.
  Only the hand-paint path is touched here.
- **Backdrop** (render of the *current* face, `render-image`) vs **Reference** (`references[view].url`,
  the target look, used only for the color palette). "Download a view's image" most naturally means
  the **backdrop** (round-trips 1:1 with the bake), not the reference.
- **`mask-canvas.tsx`**: a *separate* brush surface for masked inpaint of reference views (exports an
  opaque-where-painted mask). Not the hand-paint canvas, but a useful precedent for canvas sizing.

### Components Likely To Change And Why They Exist

- `hand-paint-canvas.tsx` — the canvas component. Add: an **Upload** control (draw the uploaded image
  into the canvas buffer, aligned to the backdrop, mark dirty), a **Download** control (save the
  backdrop and/or the backdrop+strokes composite), and **zoom/pan** of the backdrop+canvas viewport
  with stroke-mapping kept correct.
- `texture-panel.tsx` (`FaceTile`) — may pass the model id/view or a download handler so the canvas
  can build the backdrop download URL; otherwise unchanged.
- `lib/api.ts` — likely **no change** (reuses `faceRenderUrl` + `handpaintFace`). Listed only if we
  add a convenience helper.
- Backend (`studio.py`, `pipeline.py`) — **no change expected**; the existing overlay bake already
  supports a full-image overlay.

### Execution Locations

- Painting, palette extraction, upload-compositing, zoom/pan, download: all **client-side** in the
  browser (the Next.js static export). No server round-trip except the existing render + bake jobs.
- The bake runs on the **single GPU worker** via `submit_gpu("studio_handpaint", …)` in `studio.py`.
- Build/deploy executes on the dev machine: `pnpm build` in the source repo → `out/` → copied to
  `webapp/webui/` (or `HY3D_WEBUI_DIR` pointed at `out/`).

## Likely Change Surface

- `Downloads/3-d-model-generation-workflow/components/studio/hand-paint-canvas.tsx` (primary)
- `Downloads/3-d-model-generation-workflow/components/studio/texture-panel.tsx` (minor, if a
  download URL / id+view must be threaded in)
- Possibly `lib/api.ts` (only if a helper is added)
- Build artifact: regenerate `out/` and copy to `Hunyuan3D-2.1/webapp/webui/`
- No backend, no tests currently cover the Next.js UI (studio backend tests are
  `test_studio_api.py` / `test_reference_views.py`; unaffected).

## Source References

- `search_codebase`: `studio.py` `handpaint_face` / `_gpu_handpaint` / `render_face` /
  `face_render_image`; `pipeline.py` `paint_overlay`.
- Read: `hand-paint-canvas.tsx`, `texture-panel.tsx`, `lib/api.ts`, `mask-canvas.tsx`,
  `image-dialog.tsx`, `static/index.html`, `static/app.js`, `server.py` static mount, project
  `CLAUDE.md`, `WEBAPP_README.md`.

## Resolved Decisions (grill-me pass)

1. **Download** saves the **current face render (backdrop)** — the square PNG that round-trips 1:1
   with the bake. (Not the composite or the reference.)
2. **Upload** is **upload + bake immediately**: picking a file converts it to the overlay and POSTs
   `handpaintFace` straight away (no manual editing step). Reuses the existing bake path — **no
   backend change**.
3. **Upload fit**: draw the uploaded image into a `dim×dim` offscreen buffer with **contain** fit
   (letterbox transparent, no distortion) before export, so a downloaded backdrop aligns 1:1 and an
   arbitrary aspect ratio is not stretched. Transparent letterbox bakes nothing (alpha 0).
4. **Zoom**: **mouse-wheel zoom (at cursor) + drag-to-pan**, no slider. The brush `ptr()` mapping
   stays correct under CSS transform. Buffer stays `dim×dim`, so bake resolution is unchanged.
5. **Build/deploy**: I run `pnpm build` in the source repo and copy `out/` → `webapp/webui/`
   (node v22 + pnpm 10.15 present).

### Remaining design detail (resolve in Design)

- **Pan vs paint gesture conflict**: left-drag already paints, so panning needs a distinct gesture.
  Default plan: pan on **Space-hold drag** and/or **middle/right-mouse drag**, with a small **Pan**
  toggle as a discoverable fallback; plain left-drag still paints. Wheel always zooms at the cursor.

## Architecture-Change Risk

**None.** Pure client-side additions to one React component (plus the build/copy step). No new
infrastructure, endpoints, contracts, or source-of-truth changes. The only cross-repo wrinkle is that
the editable source lives outside `Hunyuan3D-2.1/webapp` (in `Downloads/`), and the served `webui/` is
a build artifact that must be regenerated.

## Initial Verification Ideas

- `pnpm build` succeeds; copy `out/` → `webapp/webui/`; load the studio, open a face → Hand paint.
- Download → file is the square face PNG; re-upload → it fills the canvas aligned to the backdrop;
  Apply → bake replaces the visible face with the uploaded image.
- Zoom in → strokes still land under the cursor (mapping correct); pan reaches all edges; Apply bakes
  at full resolution regardless of zoom.
