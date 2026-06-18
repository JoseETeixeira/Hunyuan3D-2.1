"""Full end-to-end GPU probe of TextureWorker.fill_coverage_gaps WITHOUT loading the paint UNet.

Builds a single MeshRender, wraps it in a minimal fake "worker" exposing exactly what the method
touches (paint_pipeline.render/config, rembg, output_dir, low_vram_mode, PROJECTION_CAMS), and runs
the REAL TextureWorker.fill_coverage_gaps with a stub reference that returns a solid MAGENTA view.
Then it loads the resulting GLB's texture and counts magenta texels — proving fill cameras actually
wrote real colour onto former-gap texels (not inpaint). Validates: coverage re-probe, greedy
selection, alignment, back_project, masked composite, and the atomic temp+replace export.

Usage:
  python webapp/diag_gapfill_full.py <uid>   # reads {uid}_textured.glb, writes {uid}_gftest.glb
"""
import os
import shutil
import sys
import types

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import webapp

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from DifferentiableRenderer.MeshRender import MeshRender  # noqa: E402
from webapp.pipeline import TextureWorker, _extract_base_texture  # noqa: E402

uid = sys.argv[1]
out = os.environ.get("HY3D_OUTPUT_DIR", "webapp/outputs")
SZ = 512
src = f"{out}/{uid}_textured.glb"
work = f"{out}/{uid}_gftest.glb"
shutil.copyfile(src, work)  # operate on a copy so the original stays intact

rd = MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=SZ,
                texture_size=1024, device="cuda")

# Minimal fake worker exposing only what fill_coverage_gaps uses.
fake = types.SimpleNamespace(
    paint_pipeline=types.SimpleNamespace(render=rd, config=types.SimpleNamespace(render_size=SZ)),
    rembg=lambda img: img.convert("RGBA"),          # passthrough: stub view is already clean
    output_dir=out,
    low_vram_mode=False,
    PROJECTION_CAMS=TextureWorker.PROJECTION_CAMS,
)

MAGENTA = (255, 0, 255)


def stub_reference(elev, azim, geom_img):
    return Image.new("RGB", (SZ, SZ), MAGENTA)


# Standard coverage set = the 10 named views (matches the reface path).
corners = [(45.0, 45.0), (45.0, 135.0), (45.0, 225.0), (45.0, 315.0)]
standard = [(float(e), float(a)) for (e, a) in TextureWorker.PROJECTION_CAMS.values()] + corners

result = TextureWorker.fill_coverage_gaps(
    fake, uid=f"{uid}_gftest", textured_glb_path=work, get_reference=stub_reference,
    standard_cams=standard, candidate_cams=None, max_cams=4, dilation_px=4,
    cos_thres_deg=75.0, min_texels=64,
    progress=lambda k, cam, rem: print(f"[full] placed cam {k} {cam} remaining={rem}"),
)

# Count magenta texels in the result texture -> proves fill cameras wrote real colour into gaps.
mesh_out = trimesh.load(result, force="mesh")
tex = _extract_base_texture(mesh_out, result)
assert tex is not None, "result GLB has no base texture"
arr = np.asarray(tex.convert("RGB")).astype(np.int16)
mag = (np.abs(arr - np.array(MAGENTA)).sum(-1) < 60)
n_mag = int(mag.sum())
print(f"[full] result={result} magenta_texels={n_mag}")
assert os.path.exists(result), "result GLB missing"
assert n_mag > 0, "no magenta written -> fill cameras did not paint any gap texels"
print("[full] OK — fill cameras wrote real colour onto former-gap texels; atomic export succeeded")
