# Structure — 3D-Gen / Hunyuan3D-2.1

## Layout (relevant to this task)
```
Hunyuan3D-2.1/
  webapp/
    server.py          # FastAPI app, job queue, mode dispatch (_run_texture),
                       # _run_reface, _run_hyface, env knobs, /api endpoints
    pipeline.py        # TextureWorker: generate_shape, generate_texture,
                       # project_texture(_angles), paint_faces, reface,
                       # render_* helpers, PROJECTION_CAMS
    gen_transfer.py    # gpt/gemini view gen + restyle_to_references (reface)
    image_edit.py      # edit_image (gpt-image-2 / gemini), CARTOON_STYLE
    elevations.py      # mvgpt elevation synth
    static/
      index.html       # mode <select> #texmode, #projPanel slots, #refacePanel
      app.js           # applyTextureMode, generate/reface handlers, health poll
    diag_*.py          # diagnostic probes (no pytest suite)
  hy3dpaint/
    textureGenPipeline.py        # Hunyuan3DPaintPipeline, config (bake_exp,
                                 # render_size, texture_size)
    utils/pipeline_utils.py      # bake_from_multiview, texture_inpaint
    DifferentiableRenderer/
      MeshRender.py              # back_project, fast_bake_texture, uv_inpaint,
                                 # tex_normal/tex_grid, bake_angle_thres=75
  hy3dshape/                     # shape pipeline
  .batman/<task_slug>/           # steering + spec per task
```

## Conventions
- Modes are additive: a `texture_mode` value → one `_run_*` dispatch line in
  `_run_texture`. New behavior should be a new stage/branch, not a rewrite.
- Per-mode tuning via `_UPPER_SNAKE` module constants read from env at import.
- Reuse existing render/bake/synth helpers; do not duplicate bake math.
- Albedo-only matte output (`_force_matte`) for reface/hyface/projection.
- File naming: `{uid}_shape.glb`, `{uid}_textured.glb`, intermediate
  `{uid}_<tag>_<side>.png` saved in `OUTPUT_DIR`.

## Where this task lands
- New gap-fill stage helpers in `pipeline.py` (coverage mask + camera placement).
- New default-on stage calls in `server.py` `_run_reface` / `_run_hyface` + env
  knobs (shared `GAPFILL_*` prefix matching the `_HYFACE_*`/`_REFACE_*` style).
- Optional: a `diag_gapfill.py` probe to dump the gap mask before/after.
