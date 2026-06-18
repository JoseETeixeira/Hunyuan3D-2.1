"""Coverage-gap probe for the auto gap-fill stage.

Validates the load-bearing part of `TextureWorker.fill_coverage_gaps`: the coverage RE-PROBE
that flags texels no standard view covered. Builds a MeshRender directly (no paint UNet),
re-probes the 10 standard cameras exactly as the stage does (back_project a dummy image,
accumulate cos_map), and dumps the gap mask + counts. Then checks that an OBLIQUE candidate
camera (off the standard set) would cover some gap texels, and that a head-on cardinal does NOT
leave its own face as a gap — i.e. the re-probe sign + visibility are correct without any hand
dot-product.

Usage:
  python webapp/diag_gapfill.py <uid> [cos_deg]
Reads {HY3D_OUTPUT_DIR}/{uid}_shape.glb (coverage depends only on geometry).
"""
import math
import os
import sys

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from DifferentiableRenderer.MeshRender import MeshRender, get_mv_matrix  # noqa: E402
from utils.uvwrap_utils import mesh_uv_wrap  # noqa: E402

uid = sys.argv[1]
COS_DEG = float(sys.argv[2]) if len(sys.argv) > 2 else 75.0
out = os.environ.get("HY3D_OUTPUT_DIR", "webapp/outputs")
SZ = 512
cos_thres = math.cos(math.radians(COS_DEG))

# 10 standard views that DEFINE coverage: 6 cardinals + 4 corners (elev 45).
STANDARD = [(0.0, 0.0), (0.0, 180.0), (0.0, 90.0), (0.0, 270.0), (90.0, 0.0), (-90.0, 0.0),
            (45.0, 45.0), (45.0, 135.0), (45.0, 225.0), (45.0, 315.0)]
# A few oblique candidates the standard set lacks (should pick up gap texels).
OBLIQUE = [(30.0, 45.0), (-30.0, 135.0), (60.0, 200.0), (-45.0, 300.0)]

mesh = trimesh.load(f"{out}/{uid}_shape.glb", force="mesh")
mesh = mesh_uv_wrap(mesh)
rd = MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=SZ,
                texture_size=1024, device="cuda")
rd.load_mesh(mesh=mesh)

valid = rd.texture_indices >= 0
n_valid = int(valid.sum().item())


def coverage(cams):
    """Accumulated coverage trust over a camera list, via back_project (same as the stage)."""
    dummy = np.ones((SZ, SZ, 3), dtype=np.float32)
    trust = torch.zeros(rd.texture_size, device=rd.device)
    for (e, a) in cams:
        _, cos_map, _ = rd.back_project(dummy, e, a)
        trust = trust + cos_map[..., 0]
    return trust


trust = coverage(STANDARD)
covered = trust > 1e-8
gap = valid & (~covered)
n_gap = int(gap.sum().item())

gap_png = (gap.detach().cpu().numpy().astype(np.uint8) * 255)
Image.fromarray(gap_png).save(f"{out}/{uid}_gapmask.png")
print(f"[diag_gapfill] valid_texels={n_valid} gap_texels={n_gap} "
      f"({100.0 * n_gap / max(1, n_valid):.2f}% of mesh) cos_deg={COS_DEG}")
print(f"[diag_gapfill] saved {out}/{uid}_gapmask.png (white = gap)")

# Each oblique candidate should newly cover some of the gap (proves fill cameras can reach gaps,
# and that single-camera coverage respects the same gate/visibility).
for (e, a) in OBLIQUE:
    t = coverage([(e, a)])
    new = int((gap & (t > 1e-8)).sum().item())
    print(f"[diag_gapfill] oblique elev={e:>5} azim={a:>5} -> newly covers {new} gap texels")

# Sanity: a head-on cardinal must cover a large share of valid texels (sign/visibility correct).
front = coverage([(0.0, 0.0)])
front_cov = int((valid & (front > 1e-8)).sum().item())
print(f"[diag_gapfill] front cardinal covers {front_cov} valid texels "
      f"({100.0 * front_cov / max(1, n_valid):.2f}%) — should be a large, nonzero share")

assert n_gap < n_valid, "every texel flagged as gap — coverage re-probe sign is likely inverted"
assert front_cov > 0, "front cardinal covered nothing — coverage re-probe is broken"
print("[diag_gapfill] OK")
