<!-- batman:spec:start -->
## Active Batman Spec(s)

### per-model-studio-backend (active)
- **Task slug**: per-model-studio-backend
- **Current phase**: Documentation (implementation + tests + code review complete; GPU/real-key E2E pending on the GPU box)
- **Steering**: `.batman/per-model-studio-backend/steering/understanding.md` (status: approved)
- **Requirements**: `.batman/per-model-studio-backend/spec/requirements.md` (status: approved)
- **Design**: `.batman/per-model-studio-backend/spec/design.md` (status: approved)
- **Tasks**: `.batman/per-model-studio-backend/spec/tasks.md` (status: implemented)
- **Code**: `webapp/studio.py` + `webapp/reference_views.py` (new); `webapp/server.py` (router mount,
  worker dispatch, mode removal), `webapp/pipeline.py` (`paint_single_view`), `webapp/gen_transfer.py`
  (reface tweak). Tests: `python -m webapp.test_studio_api` / `python -m webapp.test_reference_views`.
- **Last updated**: 2026-06-17
- Wire the new Next.js per-model studio (now `webapp/studio-ui/`, formerly
  `Downloads/3-d-model-generation-workflow`) to the webapp backend: per-model persistence, mesh-free
  staged reference generation (gpt-image-2), base = shape + hyface(front), reface per view, remove
  non-hyface/reface texture modes. Architecture change: REQUIRED.

> **Studio UI source lives in-repo at `webapp/studio-ui/`** (Next.js 16, static export). Edit there,
> then `cd webapp/studio-ui && pnpm build` and copy `out/*` → `webapp/webui/` (the FastAPI-served UI).
> `.env.production` sets `NEXT_PUBLIC_USE_MOCK=false` so the build talks to the real `/api/*` backend.

### handpaint-upload-zoom (active)
- **Task slug**: handpaint-upload-zoom
- **Current phase**: Documentation (implementation + tests + code review complete)
- **Steering**: `.batman/handpaint-upload-zoom/steering/understanding.md` (status: approved)
- **Requirements**: `.batman/handpaint-upload-zoom/spec/requirements.md` (status: approved)
- **Design**: `.batman/handpaint-upload-zoom/spec/design.md` (status: approved)
- **Tasks**: `.batman/handpaint-upload-zoom/spec/tasks.md` (status: implemented)
- **Last updated**: 2026-06-19
- Hand-paint surface: download/upload a view image (apply as hand-paint via the existing overlay
  bake) + zoom/pan while painting. Source: `webapp/studio-ui/components/studio/hand-paint-canvas.tsx`
  (+ `texture-panel.tsx` `downloadName`); built → `webapp/webui/`. Pure frontend, no backend change.
  tsc clean, `pnpm build` OK, studio backend tests green. Pending: manual in-browser E2E on the
  running studio.

Read the steering `understanding.md` first on session start. Treat each artifact as the source of truth for its phase.
<!-- batman:spec:end -->
