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
- Wire the new Next.js per-model studio (Downloads/3-d-model-generation-workflow) to the webapp
  backend: per-model persistence, mesh-free staged reference generation (gpt-image-2), base =
  shape + hyface(front), reface per view, remove non-hyface/reface texture modes. Architecture
  change: REQUIRED.

Read the steering `understanding.md` first on session start. Treat each artifact as the source of truth for its phase.
<!-- batman:spec:end -->
