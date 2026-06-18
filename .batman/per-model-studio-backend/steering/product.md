# Product Steering — Per-Model 3D Studio

## Purpose

Turn a single seed image into a fully textured, stylized (flat-cartoon, albedo-matte) 3D game
asset through a guided, **per-model** workflow, then let the user refine each face. Target output:
Roblox-ready meshes.

## Users

- Solo creator / small team running the studio locally against a single GPU.

## Key Features

- **Per-model workspace**: a named model owns its 10 reference views, mesh, and texture, reusable
  across sessions without re-uploading.
- **Staged reference generation**: from one seed image, generate the 10 orthographic reference
  views with gpt-image-2 following an imperative dependency graph (front → cardinals → corners),
  each individually approvable, editable (tweak prompt), or replaceable by upload.
- **Per-face AI paint base**: build mesh + base texture by per-face AI paint over the approved
  references.
- **Reface refinement**: depth-aware single-face re-texture per view using that view's approved
  reference; available per-face after base.
- **Export**: GLB, FBX, .blend.

## Objectives

- Nothing from the existing frontend is lost except the removed texture modes.
- Each reference view is orthographic, cartoonish, and shows only the faces visible in that view.
- The model structure is durable: name → references + mesh + texture persists across runs.

## Non-Goals

- Multi-user accounts, auth, cloud scaling.
- Texture modes other than per-face AI paint and reface.
- Replacing the Hunyuan shape model or wiring TRELLIS.2.
