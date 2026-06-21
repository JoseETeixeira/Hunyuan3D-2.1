# Understanding: Upload .blend to edit + Blender hole-fill on mesh generation

## User Goal

Two additions to the per-model studio:

1. **Upload a `.blend` file to edit** — replace the model's existing geometry with an
   uploaded `.blend`, but **keep the references** (and seed). Decision: the uploaded
   `.blend` becomes a **new untextured base mesh**; the current texture is reset so the
   user re-runs texture base / reface on the new geometry. Mirrors the existing
   "generate mesh / regenerate mesh" behavior.
2. **Use Blender to fill holes when generating a mesh** — every generated shape gets a
   Blender fill-holes pass so meshes come out watertight. Decision: **always-on**, gated
   by an env flag (default on) so it can be disabled.

## Task Slug

`blend-upload-holefill`

## Current Behavior

### Mesh generation
- Studio GPU handlers in [studio.py](Hunyuan3D-2.1/webapp/studio.py): `_gpu_mesh`
  (mesh-only, [L473](Hunyuan3D-2.1/webapp/studio.py#L473)) and `_gpu_base`
  ([L417](Hunyuan3D-2.1/webapp/studio.py#L417), generates a shape inline if none exists)
  call `worker.generate_shape` ([pipeline.py L276](Hunyuan3D-2.1/webapp/pipeline.py#L276)):
  Hunyuan shape pipeline → trimesh → optional quadric decimation → export
  `OUTPUT_DIR/{id}_shape.glb`. **No hole filling anywhere.**
- `_gpu_mesh` already does the "new geometry" reset: unlink `{id}_textured.glb`, set every
  face `status=pending`, `textureStage=none`, record `meshSourceView`/`meshConfig`, and
  `_clear_history` (old snapshots belong to the previous UV). This is exactly the
  post-import behavior the `.blend` upload needs.

### Blender integration (already present)
- Invoked as a subprocess: `BLENDER_BIN --background --python <script> -- <args>`
  (`BLENDER_BIN` env, default `blender`) via helpers in
  [server.py](Hunyuan3D-2.1/webapp/server.py): `_blender_convert` (GLB→FBX/.blend,
  [L654](Hunyuan3D-2.1/webapp/server.py#L654)) and `_blender_run` (projection bake,
  [L665](Hunyuan3D-2.1/webapp/server.py#L665)). Both check `shutil.which(BLENDER_BIN)` and
  raise 503 / RuntimeError if Blender is missing.
- Scripts: [blender_convert.py](Hunyuan3D-2.1/webapp/blender_convert.py) (imports a GLB,
  exports FBX/.blend) and [blender_project.py](Hunyuan3D-2.1/webapp/blender_project.py)
  (projection bake). **No script imports a `.blend`** and **no script fills holes** today.
  TRELLIS.2's `data_toolkit/blender_script/dump_mesh.py` shows `.blend` import via
  `bpy.ops.wm.append` but is a separate subproject, not wired to the webapp.
- `download_model` ([L1044](Hunyuan3D-2.1/webapp/studio.py#L1044)) already converts the GLB
  to `.blend` on the fly via `server._blender_convert`. So export→.blend exists; the new
  work is the inbound .blend→GLB direction.

### Upload + job patterns to reuse
- Multipart upload precedent: `replace_seed` / `upload_reference` save bytes via
  `_save_upload_png`. The `.blend` is not an image, so it needs a raw-bytes save with an
  extension check (no `_save_upload_png`).
- GPU job lane: `submit_gpu(kind, sjid)` enqueues onto `server.WORK`; `run_gpu_job`
  ([L372](Hunyuan3D-2.1/webapp/studio.py#L372)) dispatches by `kind`
  (`studio_mesh`, `studio_base`, …) then `complete_job` returns the assembled `Model`.
  Frontend `runJob(start,label)` ([studio-provider.tsx](Hunyuan3D-2.1/webapp/studio-ui/components/studio/studio-provider.tsx))
  takes a fn returning a `Job` and polls to completion. So the upload endpoint should run
  the (blocking, ~minutes) Blender convert on the worker lane and return a `Job`, exactly
  like `generateMesh`.

### Frontend mesh surface
- [texture-panel.tsx](Hunyuan3D-2.1/webapp/studio-ui/components/studio/texture-panel.tsx)
  `generateMesh()` ([L45](Hunyuan3D-2.1/webapp/studio-ui/components/studio/texture-panel.tsx#L45))
  → `api.generateMesh`. The `stage === "none"` block ([L67+](Hunyuan3D-2.1/webapp/studio-ui/components/studio/texture-panel.tsx#L67))
  has the "Generate mesh / Regenerate mesh" buttons — the natural home for an
  "Upload .blend" button.
- [lib/api.ts](Hunyuan3D-2.1/webapp/studio-ui/lib/api.ts): `generateMesh` (JSON→Job),
  `uploadReference`/`updateSeed` (multipart→Model) are the mirrors for a new
  `uploadMesh` (multipart→Job).

## Likely Files to Change

- New: `Hunyuan3D-2.1/webapp/blender_blend_to_glb.py` — open `.blend`, export GLB.
- New: `Hunyuan3D-2.1/webapp/blender_fillholes.py` — import GLB, fill holes, export GLB.
- `Hunyuan3D-2.1/webapp/server.py` — `_blender_blend_to_glb()` + `_fill_holes_glb()`
  helpers + `HY3D_FILL_HOLES` flag.
- `Hunyuan3D-2.1/webapp/studio.py` — fill-holes call after `generate_shape` in `_gpu_mesh`
  and `_gpu_base`; new `_gpu_mesh_upload` handler + `studio_mesh_upload` dispatch + new
  `POST /api/models/{id}/mesh/upload` route.
- `Hunyuan3D-2.1/webapp/studio-ui/lib/api.ts` — `uploadMesh`.
- `Hunyuan3D-2.1/webapp/studio-ui/components/studio/texture-panel.tsx` — Upload .blend
  control.
- Docs: project `CHANGELOG.md`.

## Open Questions / Risks

- Resolved by user: focused approach; .blend → new untextured base (keep refs, reset
  texture); hole-fill always-on (env-gated).
- Blender `bpy.ops.mesh.fill_holes(sides=0)` fills boundary loops; large concave openings
  may produce ugly n-gons — acceptable for a watertight base, downstream trimesh
  triangulates on load. Recalc normals + triangulate after fill keeps it clean.
- `.blend` import via `wm.open_mainfile` loads the file's whole scene (cameras/lights too);
  GLB export only emits meshes, so extras are harmless. Multiple mesh objects export fine
  (the texture bake path joins on load).
- Cannot fully run Blender/GPU paths in this environment; helpers already 503 when Blender
  is absent. Verification is tsc for the frontend + code-path review + a documented manual
  run.
