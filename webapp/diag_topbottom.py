"""Diagnose top/bottom orientation for the direct elevation bake. Bakes the cardinal
elevations (top at TOP_EL, default -89.99) and renders a vertical sweep so we can SEE
which elevation lands on the GLB roof vs base, and whether the building is upright.

Usage: python webapp/diag_topbottom.py <uid> [TOP_EL]
"""
import os
import sys

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from DifferentiableRenderer.MeshRender import MeshRender, get_mv_matrix, transform_pos  # noqa: E402
from utils.uvwrap_utils import mesh_uv_wrap  # noqa: E402

uid = sys.argv[1]
TOP_EL = float(sys.argv[2]) if len(sys.argv) > 2 else -89.99
out = os.environ.get("HY3D_OUTPUT_DIR", "webapp/outputs")
SZ = 380

mesh = trimesh.load(f"{out}/{uid}_shape.glb", force="mesh")
mesh = mesh_uv_wrap(mesh)
rd = MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=SZ, texture_size=2048, device="cuda")
rd.load_mesh(mesh=mesh)


def rembg_grey(im):
    a = np.asarray(im).astype(int)
    bg = a[0, 0]
    alpha = (np.abs(a - bg).sum(-1) > 36)
    return Image.composite(im, Image.new("RGB", im.size, (127, 127, 127)),
                           Image.fromarray((alpha * 255).astype(np.uint8)))


CARDINAL = [("front", 0.0, 0.0), ("right", 0.0, 90.0), ("back", 0.0, 180.0),
            ("left", 0.0, 270.0), ("top", TOP_EL, 0.0)]
textures, cos_maps = [], []
for side, el, az in CARDINAL:
    p = f"{out}/{uid}_elev_{side}.png"
    if not os.path.exists(p):
        continue
    im = rembg_grey(Image.open(p).convert("RGB").resize((SZ, SZ)))
    tex, cos, _ = rd.back_project(im, el, az)
    textures.append(tex)
    cos_maps.append(cos ** 6.0)
texture, trust = rd.fast_bake_texture(textures, cos_maps)
mask_np = ((trust > 1e-8).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
texture = torch.tensor(rd.uv_inpaint(texture, mask_np) / 255).float().to(texture.device)
rd.set_texture(texture, force_set=True)


def render_color(elev, azim, size=SZ):
    proj = rd.camera_proj_mat
    r_mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=rd.camera_distance, center=None)
    pos_clip = transform_pos(proj, transform_pos(r_mv, rd.vtx_pos, keepdim=True))
    rast_out, _ = rd.raster_rasterize(pos_clip, rd.pos_idx, resolution=(size, size))
    uv, _ = rd.raster_interpolate(rd.vtx_uv[None, ...], rast_out, rd.uv_idx)
    vis = torch.clamp(rast_out[..., -1:], 0, 1)[0]
    tex = rd.tex.to(rd.device).float()
    if tex.max() > 1.5:
        tex = tex / 255.0
    samp = F.grid_sample(tex.permute(2, 0, 1)[None], uv * 2 - 1, align_corners=False)[0].permute(1, 2, 0)
    img = samp * vis + (1 - vis) * 1.0
    return Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype("uint8"))


sweep = [("el=-89.99 (pole A)", -89.99, 0), ("el=-40", -40, 0), ("el=0 front", 0, 0),
         ("el=+40", 40, 0), ("el=+89.99 (pole B)", 89.99, 0)]
sheet = Image.new("RGB", (len(sweep) * SZ, SZ), (12, 12, 12))
d = ImageDraw.Draw(sheet)
for c, (lbl, el, az) in enumerate(sweep):
    sheet.paste(render_color(el, az), (c * SZ, 0))
    d.text((c * SZ + 5, 5), lbl, fill=(0, 255, 120))
path = f"{out}/{uid}_topbottom_TOPEL{TOP_EL:.0f}.png"
sheet.save(path)
print(f"baked with top at el={TOP_EL}; saved {path}")
