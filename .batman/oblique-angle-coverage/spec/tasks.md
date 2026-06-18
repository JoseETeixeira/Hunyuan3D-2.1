# Implementation Plan — Oblique-Angle Coverage (auto gap-fill)

Traceability: requirement IDs reference `spec/requirements.md` (R#.AC#). Design:
`spec/design.md`. Anchors are current line numbers in
[webapp/server.py](../../../webapp/server.py) and
[webapp/pipeline.py](../../../webapp/pipeline.py).

- [ ] 1. Add `GAPFILL_*` configuration constants (reversible, default-on)
  - In [server.py](../../../webapp/server.py) after the `_HYFACE_*` / `_HYFACE_BAKE_SIDES` block (~line 203), add: `_GAPFILL_REFACE`, `_GAPFILL_HYFACE` (bool, default "1"), `_GAPFILL_MAX_CAMS` (int 6), `_GAPFILL_DILATION` (int 4), `_GAPFILL_COS_DEG` (float 75), `_GAPFILL_MIN_TEXELS` (int 64), `_GAPFILL_GRID_ELEVS` ("-60,-30,0,30,60"), `_GAPFILL_GRID_AZ_STEP` (int 30).
  - Match the exact `os.environ.get("UPPER_SNAKE", default)` style (bool = `.lower() not in ("0","false","no")`, numeric wrapped).
  - Add a short comment block explaining the stage + that toggles off → legacy behavior.
  - _Requirements: 6.1, 6.2, 6.4_

- [ ] 2. Implement `TextureWorker.fill_coverage_gaps` in pipeline.py
  - Add the method after `reface` (~line 705) in [pipeline.py](../../../webapp/pipeline.py); signature per design (uid, textured_glb_path, get_reference, standard_cams, candidate_cams, max_cams, dilation_px, cos_thres_deg, min_texels).
  - Reload mesh preserving UVs + seed base texture (mirror `reface` step 1: `trimesh.load(force="mesh")` → `render.load_mesh(mesh)` → `_extract_base_texture` → `render.set_texture` → `base = get_texture()` tensor). `load_mesh` populates `tex_normal`/`tex_grid`/`texture_indices` (confirmed [MeshRender.py:724](../../../hy3dpaint/DifferentiableRenderer/MeshRender.py#L724)).
  - Coverage re-probe: build `STANDARD_CAMS` = 6 cardinals (`PROJECTION_CAMS`) + 4 corners (`HYFACE_CORNER_CAMS` azim, elev `_HYFACE_CORNER_ELEV`). For each, `_, cos_map, _ = render.back_project(ones_rgba, elev, azim)`; accumulate `trust += cos_map[...,0]`. `valid = texture_indices >= 0`; `covered = trust > 1e-8`; `gap_uv = valid & ~covered`.
  - Early return: if `gap_uv.sum() < min_texels` → log + return input path unchanged.
  - Dilate: `gap_dilated = cv2.dilate(gap_uv.uint8, kernel(dilation_px))`; keep undilated `remaining_gap` for tracking.
  - Build `candidate_cams` from `_GAPFILL_GRID_ELEVS` × azim steps (+ standard cams). Rank by analytic score: `tex_normal` of `remaining_gap` texels vs each candidate's `lookat` (from `DifferentiableRenderer.camera_utils.get_mv_matrix`, `cam = -w2c[:3,:3].T @ w2c[:3,3]`).
  - Greedy loop (≤ `max_cams`, ≤2 low-yield strikes): pick top candidate → `ref = get_reference(elev, azim, nearest_faces)` (skip None) → align (`_align_photo` + `_best_silhouette_fit` vs the camera's normal silhouette) → `new_tex, cos_map, _ = back_project(rgba, elev, azim)` → `m = gap_dilated & (cos_map[...,0]>1e-4) & (new_tex[...,3]>0.5)`; if `(remaining_gap & m).sum() < min_texels` → strike, next; else `out[m] = new_tex[...,:3][m]`; `remaining_gap &= ~m`; re-rank.
  - Save: `render.set_texture(out, force_set=True)` → `render.save_mesh(obj)` → `trimesh.load` → `_force_matte` → export `{uid}_textured.glb` (mirror `reface` step 5/save). Respect `low_vram_mode` empty_cache.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 5.1, 5.2, 5.3_

- [ ] 3. Add the shared reference ladder + nearest-face helper in server.py
  - Add `_gap_reference(worker, shape_glb, elev, azim, nearest_face_imgs)` near `_openai_paint_view` (~line 218): (1) gpt/gemini synth via `render_geometry_at([("gap",elev,azim)])` + `_openai_paint_view` when `OPENAI_API_KEY` + refs; (2) else reuse `nearest_face_imgs[0]`; (3) else return None.
  - Add a `_nearest_face(elev, azim, face_imgs_by_label)` helper: pick the standard face whose `lookat` is closest to the camera's `lookat` (reuse the same `get_mv_matrix` lookat used in task 2).
  - _Requirements: 3.2, 3.3, 4.2, 4.3_

- [ ] 4. Wire gap-fill into `_run_hyface` (default-on, guarded)
  - In [server.py](../../../webapp/server.py) `_run_hyface`, before the final `completed` `_set` (~line 940), if `_GAPFILL_HYFACE`: build a `get_reference` closure over `_gap_reference` seeded with the `face_refs` (nearest already-painted faces), wrapped in try/except (log + keep base GLB on failure).
  - Call `worker.fill_coverage_gaps(uid=job_id, textured_glb_path=textured_path, get_reference=..., max_cams=_GAPFILL_MAX_CAMS, ...)`; reassign `textured_path` to its return.
  - Add `_set(job_id, status="processing_texture", progress=94, message="Covering oblique gaps")` before the call.
  - _Requirements: 3.1, 3.4, 7.2, 7.3_

- [ ] 5. Wire gap-fill into `_run_reface` (default-on, guarded)
  - In `_run_reface`, before the final `completed` `_set` (~line 1003), if `_GAPFILL_REFACE`: build a `get_reference` closure — with refs: `render_textured_view(glb, e, a)` → `restyle_to_references(base, ref_paths)`; without refs: `_gap_reference(...)` (R4 no-ref ladder). Wrap in try/except (non-fatal).
  - Call `worker.fill_coverage_gaps(...)` on the reface result `textured_path`; reassign return.
  - Add a progress `_set` message before the call.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 7.2, 7.3_

- [ ] 6. Observability
  - In `fill_coverage_gaps`, log `print(f"[gapfill] mode-agnostic gaps={n0} cams={k} angles={[...]} remaining={n1}")`; no secrets.
  - Ensure each placed camera emits a `_set(job_id, message=...)`-style update via a passed-in callback or by the caller; keep within existing progress band (94–98).
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 7. Diagnostic probe `webapp/diag_gapfill.py`
  - New script: load a `{uid}_textured.glb`, run the coverage re-probe, dump `gap_uv` as a PNG + print gap-texel count; run `fill_coverage_gaps` with a STUB `get_reference` (returns a solid-colour PIL) and a low cap; dump after-count + result GLB.
  - Assert a known oblique face IS in `gap_uv` and a head-on face is NOT (validates the re-probe sign/visibility without a hand dot-product).
  - _Requirements: 1.1, 2.1 (verification)_

- [ ] 8. Tests (Phase 6)
  - Unit (no GPU): greedy selection respects `max_cams`, `min_texels`, and the 2-strike early-stop (stub coverage counts); empty-gap early return; non-fatal fallback (stub `get_reference` raises → `fill_coverage_gaps` returns input path, texture unchanged).
  - Integration: reface end-to-end with refs + without refs; hyface end-to-end — confirm the screenshot model's oblique walls carry real colour (`cos>0` there) after the stage; run via `diag_gapfill.py` + visual.
  - Regression: `GAPFILL_REFACE=0 GAPFILL_HYFACE=0` → reface/hyface output identical to pre-feature; a non-target mode (`projection`) output unchanged either way.
  - _Requirements: 7.1, 7.2, 7.3; NFR reliability_

- [ ] 9. Documentation (Phase 8)
  - Update `Hunyuan3D-2.1/CHANGELOG.md` with the gap-fill stage + `GAPFILL_*` knobs.
  - Note the env knobs + default-on behavior wherever hyface/reface modes are documented (README / mode hints if present).
  - _Requirements: 6.4 (documented defaults)_

## Verification
- `python -m webapp.diag_gapfill <uid>` on the screenshot model → gap mask shrinks; result GLB shows textured oblique walls.
- reface + hyface jobs end-to-end with stage on; spot-check a non-target mode unchanged.
- `GAPFILL_*=0` → byte-identical legacy output.
- Code review against `codeReview.instructions.md` (esp. best-effort isolation, bounded fallbacks, no secrets in logs).

## Dependencies / ordering
- 1 → 2 → 3 → (4, 5) → 6 → 7 → 8 → 9. Tasks 2 and 3 are independent of each other after 1; 4 depends on 3, 5 depends on 3. 7 needs 2. 8 needs 4+5+7.
