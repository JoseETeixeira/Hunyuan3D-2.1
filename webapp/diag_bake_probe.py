"""Decisive end-to-end MV->Hunyuan bake probe (symmetry-proof).

Re-bakes the 4 equator views of a finished combined job under a parametrized azimuth
mapping (H = SIGN*az + OFFSET) with an optional horizontal image mirror, then forward-
renders the textured mesh from the 4 cardinal Hunyuan azimuths (which correspond to the
GLB faces +Z/+X/-Z/-X = viewer front/right/back/left). A magenta bar is painted on the
LEFT edge of every input view so in-plane mirroring is directly visible in the result.

Reads existing gptviews (no MV/GPT regen). Usage:
  python webapp/diag_bake_probe.py <uid> <SIGN> <OFFSET> <MIRROR>
e.g. current mapping:  ... <uid> 1 90 0
"""
import os
import sys

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from DifferentiableRenderer.MeshRender import MeshRender, get_mv_matrix, transform_pos  # noqa: E402
from utils.uvwrap_utils import mesh_uv_wrap  # noqa: E402

uid = sys.argv[1]
SIGN = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OFFSET = int(sys.argv[3]) if len(sys.argv) > 3 else 90
MIRROR = int(sys.argv[4]) if len(sys.argv) > 4 else 0
out = os.environ.get("HY3D_OUTPUT_DIR", "webapp/outputs")
SZ = 512
BAKE_EXP = 8.0

mesh = trimesh.load(f"{out}/{uid}_shape.glb", force="mesh")
mesh = mesh_uv_wrap(mesh)
rd = MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=SZ, texture_size=1024, device="cuda")
rd.load_mesh(mesh=mesh)

angles = json.load(open(f"{out}/{uid}_mvadapter_angles.json"))
EQ = [0, 1, 2, 3]  # MV az -90 LEFT, 0 FRONT, 90 RIGHT, 180 BACK
LBL = {0: "MVleft(az-90)", 1: "MVfront(az0)", 2: "MVright(az90)", 3: "MVback(az180)"}


def load_view(i):
    for suf in ("gptview", "view"):
        p = f"{out}/{uid}_mvadapter_{suf}{i}.png"
        if os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((SZ, SZ))
            if i == 1:  # MVfront only: bake-robust asymmetric marker, LEFT-CENTER disc
                d = ImageDraw.Draw(im)
                cx, cy, r = int(SZ * 0.28), int(SZ * 0.5), int(SZ * 0.1)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 255))
            return im
    raise FileNotFoundError(i)


# --- bake ---
textures, cos_maps = [], []
for i in EQ:
    az, el = angles[i]
    img = load_view(i)
    if MIRROR:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    H = (SIGN * az + OFFSET) % 360
    tex, cos, _ = rd.back_project(img, el, H)
    textures.append(tex)
    cos_maps.append(cos ** BAKE_EXP)
texture, trust = rd.fast_bake_texture(textures, cos_maps)
mask = (trust > 1e-8)
mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
texture_np = rd.uv_inpaint(texture, mask_np)
texture = torch.tensor(texture_np / 255).float().to(texture.device)
rd.set_texture(texture, force_set=True)


def render_color(elev, azim, size=SZ):
    proj = rd.camera_proj_mat
    r_mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=rd.camera_distance, center=None)
    pos_camera = transform_pos(r_mv, rd.vtx_pos, keepdim=True)
    pos_clip = transform_pos(proj, pos_camera)
    rast_out, _ = rd.raster_rasterize(pos_clip, rd.pos_idx, resolution=(size, size))
    uv, _ = rd.raster_interpolate(rd.vtx_uv[None, ...], rast_out, rd.uv_idx)
    vis = torch.clamp(rast_out[..., -1:], 0, 1)[0]
    tex = rd.tex.to(rd.device).float()
    if tex.max() > 1.5:
        tex = tex / 255.0
    grid = uv * 2 - 1  # u->x, v->y; write used row=v,col=u with no flip
    samp = F.grid_sample(tex.permute(2, 0, 1)[None], grid, align_corners=False)[0].permute(1, 2, 0)
    img = samp * vis + (1 - vis) * 1.0
    return Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype("uint8"))


# H -> GLB face label (Th: GLB->Hun (-X,Z,-Y); cam at H sees that GLB face)
FACE = {0: "+Z FRONT(viewer)", 90: "+X side", 180: "-Z BACK", 270: "-X side"}
renders = {h: render_color(0.0, float(h)) for h in [0, 90, 180, 270]}

# contact sheet: row0 = input views, row1 = forward renders
cols = 4
top = [load_view(i) for i in EQ]
sheet = Image.new("RGB", (cols * SZ, 2 * SZ), (12, 12, 12))
d = ImageDraw.Draw(sheet)
for c, i in enumerate(EQ):
    sheet.paste(top[c], (c * SZ, 0))
    d.text((c * SZ + 6, 6), LBL[i], fill=(255, 255, 0))
for c, h in enumerate([0, 90, 180, 270]):
    sheet.paste(renders[h], (c * SZ, SZ))
    d.text((c * SZ + 6, SZ + 6), f"render H{h}={FACE[h]}", fill=(0, 255, 120))
path = f"{out}/{uid}_bakeprobe_s{SIGN}_o{OFFSET}_m{MIRROR}.png"
sheet.save(path)
print("MAPPING H =", f"{SIGN}*az + {OFFSET}", "mirror=", MIRROR)
print("saved", path)
