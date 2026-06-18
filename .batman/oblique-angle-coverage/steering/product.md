# Product — 3D-Gen webapp texture pipeline

## Purpose
Web app on top of Hunyuan3D-2.1 that turns a reference image into a textured,
Roblox-ready 3D asset. Splits the upstream monolithic generate() into steps:
image → shape (untextured mesh) → preview → texture (multiple modes) → export.

## Key features (texture modes)
- `hunyuan` — Hunyuan PBR paint (one ref conditions all views).
- `projection` — bake user photos onto canonical cameras.
- `gptproject` — gpt-image-2 paints each canonical view from geometry + refs.
- `hyface` — per-face AI paint: 1-view Hunyuan per face, baked into one UV texture.
- `mvadapter` / `mvgpt` — multi-view adapter (+ gpt elevation refine).
- `reface` — depth-aware single-view re-texture over an already-textured mesh.

## Objectives
- High-fidelity, stylized (flat-cartoon, albedo-matte) game assets.
- Each mode additive + independently selectable; existing modes never regress.
- Run on limited VRAM (16GB) via sequential shape/paint model swap.

## This task's slice
Oblique surfaces (normals grazing every fixed camera >75°) come out blank/smeared.
Add an automatic coverage-gap fill stage to `reface` and `hyface` so those
surfaces get real texture, default on. See `spec/requirements.md`.
