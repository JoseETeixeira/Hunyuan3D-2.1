"""Pair each generated view with the mesh's surface-normal render at the SAME bake angle
(H = az + MVADAPTER_AZ_OFFSET) so geometry-vs-content drift is directly visible. Also
shows source + references for appearance comparison. Read-only (no bake, no API)."""
import os
import sys
import json

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

import trimesh  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from DifferentiableRenderer.MeshRender import MeshRender  # noqa: E402
from utils.uvwrap_utils import mesh_uv_wrap  # noqa: E402

uid = sys.argv[1]
off = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
out = os.environ.get("HY3D_OUTPUT_DIR", "webapp/outputs")
S = 300

mesh = trimesh.load(f"{out}/{uid}_shape.glb", force="mesh")
mesh = mesh_uv_wrap(mesh)
rd = MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=S, device="cuda")
rd.load_mesh(mesh=mesh)
angles = json.load(open(f"{out}/{uid}_mvadapter_angles.json"))


def L(p):
    try:
        return Image.open(p).convert("RGB").resize((S, S))
    except Exception:
        return Image.new("RGB", (S, S), (40, 40, 40))


# top strip: source + references
inp = [("source0", f"{out}/{uid}_source0.png")]
for i in range(6):
    p = f"{out}/{uid}_reference{i}.png"
    if os.path.exists(p):
        inp.append((f"ref{i}", p))

n = len(angles)
cols = 6
groups = (n + cols - 1) // cols
sheet = Image.new("RGB", (cols * S, S + groups * 2 * S), (12, 12, 12))
d = ImageDraw.Draw(sheet)
for c, (lbl, p) in enumerate(inp[:cols]):
    sheet.paste(L(p), (c * S, 0))
    d.text((c * S + 4, 4), lbl, fill=(0, 230, 255))

# each view occupies a half-cell: normal (left) | gptview (right) stacked? Use 2 columns per view.
# simpler: one row of normals, one row of gptviews, per group of `cols`.
y = S
for start in range(0, n, cols):
    grp = list(range(start, min(start + cols, n)))
    for j, i in enumerate(grp):
        az, el = angles[i]
        nrm = rd.render_normal(float(el), float(az + off), use_abs_coor=True, return_type="pl").convert("RGB").resize((S, S))
        sheet.paste(nrm, (j * S, y))
        d.text((j * S + 4, y + 4), f"NORM v{i} az{az:.0f} el{el:.0f}", fill=(0, 255, 120))
    for j, i in enumerate(grp):
        gv = L(f"{out}/{uid}_mvadapter_gptview{i}.png")
        sheet.paste(gv, (j * S, y + S))
        d.text((j * S + 4, y + S + 4), f"GEN v{i}", fill=(255, 230, 0))
    y += 2 * S

path = f"{out}/{uid}_geomcheck.png"
sheet.save(path)
print("saved", path)
