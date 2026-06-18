# Tech — 3D-Gen / Hunyuan3D-2.1 webapp

## Stack
- Python, PyTorch, CUDA. FastAPI server (`webapp/server.py`) + threaded job queue
  (`JOBS`, `WORK` Queue, `_worker_loop`).
- Shape: `hy3dshape` (`Hunyuan3DDiTFlowMatchingPipeline`). Paint: `hy3dpaint`
  (`Hunyuan3DPaintPipeline`, multiview diffusion UNet + DINO + RealESRGAN super-res).
- Differentiable renderer `hy3dpaint/DifferentiableRenderer/MeshRender.py` —
  rasterize, `render_normal/position/alpha`, `back_project`, `fast_bake_texture`,
  `uv_inpaint`, `set_texture/get_texture`; per-texel `tex_normal`/`tex_grid`.
- trimesh for mesh IO/export; GLB out (albedo-only matte via `_force_matte`).
- gpt-image-2 + Gemini fallback for view synth (`webapp/image_edit.py`,
  `webapp/gen_transfer.py`). Optional headless Blender for projection bakes.
- Frontend: vanilla JS (`webapp/static/app.js`) + `<model-viewer>`; `index.html`.

## Key invariants
- Bake is cosine-gated: `MeshRender.bake_angle_thres = 75`. Coverage = trust map
  from `fast_bake_texture` (`> 1e-8`). Uncovered texels → `cv2.INPAINT_NS`.
- Camera convention: `PROJECTION_CAMS` (elev,azim); `get_mv_matrix` flips elev
  internally (above = negative elev for the renderer). Corners in
  `HYFACE_CORNER_CAMS` (azim 45/135/225/315, elev 45).
- Both reface (`worker.reface(elev,azim,mask=...)`) and hyface
  (`worker.paint_faces(view_specs)`) already accept arbitrary cameras.
- Config knobs via `os.environ.get("UPPER_SNAKE", "default")`; bool =
  `.lower() not in ("0","false","no")`, numeric wrapped in `float(...)`/`int(...)`.
- Sequential VRAM: `_move_multiview("cuda"/"cpu")`; `low_vram_mode` → empty_cache.

## Output / API contract
- Result `{uid}_textured.glb`; served `/api/files/{name}`. Jobs carry
  `status`/`progress`/`message`/`textured_path`/`textured_url` via `_set(...)`.
- `/api/health` exposes `{"openai": bool(OPENAI_API_KEY), ...}`; `app.js`
  `openaiAvailable` gates AI features.

## Testing
- Diagnostic probes: `webapp/diag_bake_probe.py` (re-bake + forward render),
  `diag_geom_check.py`, `diag_topbottom.py`, `diag_glbframe.py`, `diag_render_glb.py`.
- `test_api_server.py` (repo root) exercises generate → poll → fetch GLB.
- No formal pytest suite for the webapp; verification is probe + visual.
