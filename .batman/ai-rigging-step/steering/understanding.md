# Understanding: Step 3 — AI Rigging (UniRig) with positionable joint markers

## User Goal

Add a **third workflow step** ("Rigging") after "Mesh & textures". It auto-rigs the model's
mesh with **AI (UniRig)** and exposes the key joints as **positionable markers** the user can
move by clicking the mesh surface in the 3D viewer. Marker set (12): groin, chin, L/R shoulder,
L/R elbow, L/R hand, L/R knee, L/R ankle. Output: a **rigged mesh exportable as GLB, FBX, and
.blend**.

Decisions captured from the user:
- **Full rig in one go** (skeleton + skin + rigged export), not markers-only.
- **AI = UniRig** (https://github.com/VAST-AI-Research/UniRig), not a custom vision heuristic.
- **Reposition markers by clicking the mesh surface in 3D.**
- **Short design doc first**, then build.

## Task Slug

`ai-rigging-step`

## Current State (grounded)

### Frontend
- Stepper: [workflow-panel.tsx](Hunyuan3D-2.1/webapp/studio-ui/components/studio/workflow-panel.tsx)
  — `type Step = "references" | "texture"`, two `StepTab`s with an auto-advance effect. A third
  `StepTab` + a new `RigPanel` slot here.
- 3D viewer: [model-3d-viewer.tsx](Hunyuan3D-2.1/webapp/studio-ui/components/studio/model-3d-viewer.tsx)
  uses `<model-viewer>`. It already reads the live orbit (`getCameraOrbit`) for "Paint this angle".
  `<model-viewer>` also exposes `positionAndNormalFromPoint(px, py)` (screen→surface raycast) and
  **hotspot slots** (`slot="hotspot-<id>"` + `data-position="x y z"`) — the primitives for showing
  + placing markers. The type decl
  [types/model-viewer.d.ts](Hunyuan3D-2.1/webapp/studio-ui/types/model-viewer.d.ts) must gain
  `positionAndNormalFromPoint`.
- Model type: [lib/types.ts](Hunyuan3D-2.1/webapp/studio-ui/lib/types.ts) `Model` — gains a `rig`
  field (markers + artifact urls + stage). `lib/api.ts` gains rig endpoints; `lib/mock-backend.ts`
  gains mocks.

### Backend
- Registry + router + jobs: [studio.py](Hunyuan3D-2.1/webapp/studio.py). `model.json` per model
  (`assemble_model` derives URLs from file existence). GPU jobs go on the single worker lane
  (`submit_gpu(kind, sjid)` → `run_gpu_job` dispatch → `complete_job`). New: a `rig` lane handler +
  routes, plus rig artifact paths + persistence.
- Subprocess precedent: Blender runs as `BLENDER_BIN --background --python <script> -- <args>` via
  `server._blender_convert` / `_blender_run` (env-gated, 503 if missing). **UniRig will mirror this
  exactly** — its own venv + repo path via env, invoked as a subprocess on the GPU lane.
- Export: `download_model` already serves GLB and converts to FBX/.blend via `_blender_convert`
  (gltf import keeps armature + skin; FBX export + `save_as_mainfile` keep them). So once a rigged
  GLB exists, FBX/.blend export mostly reuses the existing path.
- Mesh source for rigging: `{id}_shape.glb` (untextured) or `{id}_textured.glb`. UniRig rigs geometry;
  texture is irrelevant to skeleton/skin but should be preserved into the rigged output.

### Not present
- No rigging/armature/skeleton/bone/skin-weight code anywhere in the repo. Fully greenfield.
- UniRig is not vendored locally (`3D-Gen/` has only `Hunyuan3D-2.1`, `TRELLIS.2`). It is a **new
  external dependency** with its own environment — an architecture change requiring validation.

## Key Open Questions (resolved in design.md, need user sign-off)

1. **Skeleton source of truth**: use UniRig's fully-predicted skeleton and map its joints onto the
   12 named markers (B1), vs. use a fixed 12-joint humanoid template we control and use UniRig only
   for skinning (B2). Tradeoffs in design.
2. **Marker "center" semantics**: clicking the surface places a joint at the *center* of the limb
   (ray entry/exit midpoint via trimesh) vs. on the surface.
3. **Re-skin on edit**: moving a marker re-runs UniRig skinning on the edited skeleton (correct but
   slow) vs. just translates the bone (instant, weights approximate).
4. **Non-humanoid meshes**: behavior when the mesh isn't a character (UniRig still predicts a
   skeleton; the named-marker mapping may not apply).

## Risks
- UniRig env is heavy and CUDA/flash-attn/spconv-specific; install + first-run weight download are
  the main setup risks. Must be isolated from the Hunyuan env.
- Joint→marker mapping depends on UniRig's output naming/topology, which varies by mesh.
- Editing joints after skinning invalidates weights unless re-skinned.
- Cannot run UniRig/GPU here; verification is design review + a documented manual run on a GPU host.
