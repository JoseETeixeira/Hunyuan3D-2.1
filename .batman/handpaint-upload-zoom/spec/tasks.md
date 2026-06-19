# Implementation Plan: Hand-paint view download/upload + zoom

Spec: `requirements.md` · `design.md` · `steering/constitution.md`
Primary file: `Downloads/3-d-model-generation-workflow/components/studio/hand-paint-canvas.tsx`
Secondary: `Downloads/3-d-model-generation-workflow/components/studio/texture-panel.tsx`
Deploy: `pnpm build` in the source repo → copy `out/*` → `Hunyuan3D-2.1/webapp/webui/`

All icons confirmed present in `lucide-react`: `Upload`, `Download`, `Hand`, `RotateCcw`
(+ existing `Check`, `Eraser`, `Loader2`, `Trash2`).

---

- [x] 1. Scaffold props, imports, and the toolbar shell
  - In `hand-paint-canvas.tsx`, add the `downloadName?: string` prop to `HandPaintCanvas` (default
    `"handpaint"`); keep `backdropUrl`, `refUrl`, `onApply`, `busy` unchanged.
  - Import new icons from `lucide-react`: `Upload`, `Download`, `Hand`, `RotateCcw`.
  - Add a hidden `<input type="file" accept="image/*">` (`uploadInput` ref) inside the component.
  - Add a controls row (reuse `Button` `size="xs"` + existing Tailwind tokens) with placeholder
    Upload / Download / Pan / Reset buttons next to the existing Erase control; wire the zoom readout
    span (shows `${Math.round(zoom*100)}%`).
  - In `texture-panel.tsx` `FaceTile`, pass `downloadName={`handpaint-${view}`}` to `<HandPaintCanvas>`.
  - _Requirements: 1.2, 2.1_

- [x] 2. Implement Download of the current backdrop
  - Add `async function downloadBackdrop()`: `fetch(backdropUrl)` → `blob()` → create object URL →
    anchor with `download = `${downloadName}.png`` → click → `URL.revokeObjectURL`.
  - Wire the Download button; disable it when `!backdropUrl` (and during `busy`).
  - Verify the saved file is the square backdrop (so it round-trips 1:1 with the bake).
  - _Requirements: 1.1, 1.3, 1.4_

- [x] 3. Implement Upload → immediate bake (reuse `onApply`)
  - Add `containFit(dim, w, h)` helper returning `{dx, dy, dw, dh}` (centered, no distortion).
  - Add `function bakeUploadedFile(file: File)`: guard `file.type.startsWith("image/")` (else ignore +
    reset input); load into `new Image()`; on load draw it contain-fit onto an offscreen
    `dim×dim` canvas; `toBlob((b) => b && onApply(b), "image/png")`.
  - Wire the Upload button → `uploadInput.current.click()`; `onChange` → `bakeUploadedFile`; reset
    `input.value` after.
  - Disable Upload when `busy || !backdropUrl`. Confirm the bake reuses `FaceTile.applyHandpaint`
    (`api.handpaintFace`) so success closes the dialog and failure surfaces via `jobError`.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 4. Implement zoom + pan (view-only) with correct brush mapping
  - Add state: `zoom` (init 1, `MAX_ZOOM=6`), `pan` `{x,y}` (init `{0,0}`), `panMode` (bool); add
    `viewportRef` (fixed `overflow-hidden` frame) and an inner transform wrapper holding the existing
    `<img>` + `<canvas>` with `style={{ transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`,
    transformOrigin: "0 0" }}`.
  - `onWheel` (preventDefault): `z2 = clamp(zoom*(1 - e.deltaY*k), 1, MAX_ZOOM)`; keep the point under
    the cursor fixed — `world = (cursor - pan)/zoom; pan2 = cursor - world*z2` (cursor relative to
    `viewportRef`); clamp `pan` so content stays in frame; force `pan={0,0}` when `zoom===1`.
  - Pan gesture: in `down`/`move`/`up`, if `e.button===1` (middle mouse) OR `panMode`, drag updates
    `pan` instead of painting; otherwise paint via the existing path. Set cursor `grab`/`grabbing` vs
    `crosshair`.
  - Wire the Pan (`Hand`) toggle → `setPanMode` and the Reset (`RotateCcw`) button → `zoom=1, pan={0,0}`.
  - Confirm `ptr()` is unchanged and strokes still land under the cursor at ≥2× (rect-based mapping is
    transform-agnostic); confirm `canvas.width/height` stay `dim` so `toBlob`/Apply/Upload export the
    full buffer (no resolution loss / crop).
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 5. Finalize disabled/busy states, hints, and error behavior
  - Ensure paint, Apply, Clear, Upload, Download, Pan, Reset are disabled while `busy` (and Download/
    Upload while `!backdropUrl`).
  - Update the helper hint text to mention download/upload and wheel-zoom + Pan toggle.
  - Verify a failed bake (e.g. no texture → 400) shows the existing `jobError` banner and leaves the
    face unchanged (no partial UI change).
  - _Requirements: 2.3, 2.4, 2.5, NFR-Usability, NFR-Reliability_

- [x] 6. Build, deploy, and verify
  - Run `pnpm build` in `Downloads/3-d-model-generation-workflow` (must produce `out/`).
  - Copy `out/*` → `Hunyuan3D-2.1/webapp/webui/` (overwrite).
  - Backend regression: `python -m webapp.test_studio_api` stays green (proves no backend touch).
  - Manual E2E (serve studio, open a textured face → Hand paint):
    - Download → square PNG; re-upload it → bakes with correct registration.
    - Upload an edited backdrop → face repaints after the job; non-image ignored.
    - Zoom ≥2× → stroke under cursor; pan (middle-mouse + toggle) reaches edges; Reset → fit; Apply/
      Upload bake full face regardless of zoom.
  - _Requirements: 1.1, 2.1, 2.2, 3.1, 3.3, 3.4, 3.5, Success Criteria_

---

## Verification

- **Build**: `cd Downloads/3-d-model-generation-workflow && pnpm build` → `out/` regenerated; then copy
  to `webapp/webui/`.
- **Backend regression**: `cd Hunyuan3D-2.1 && python -m webapp.test_studio_api` (and
  `python -m webapp.test_reference_views`) pass unchanged.
- **Manual checks**: the four E2E scenarios in task 6.
- **Constitution**: diff limited to the two `.tsx` files (+ rebuilt `webui/`); no backend/`static/`/
  `package.json` change; overlay still a `dim×dim` RGBA PNG.

## Traceability summary

| Task | Requirements |
|---|---|
| 1 Scaffold | 1.2, 2.1 |
| 2 Download | 1.1, 1.3, 1.4 |
| 3 Upload→bake | 2.1–2.6 |
| 4 Zoom/pan | 3.1–3.6 |
| 5 States/errors | 2.3–2.5, NFRs |
| 6 Build/verify | 1.x, 2.x, 3.x, Success Criteria |

## Decisions
- Upload reuses `onApply`→`api.handpaintFace` (no new API, free error handling).
- Zoom/pan are CSS-transform only; buffer stays `dim×dim`; `ptr()` untouched.
- Pan = middle-mouse drag or Pan (Hand) toggle; left-drag paints; Reset returns to fit.
