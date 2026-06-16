"""Bake-only iteration tool for the Hunyuan N-view bake (mvgpt combined).

Re-bakes a finished job's existing per-view images (gptview*, else rawview*) onto its mesh
WITHOUT regenerating anything, under a configurable MV->Hunyuan mapping, then forward-
renders each canonical face next to its source elevation so face-mapping errors are obvious.
Lets us pin azimuth offset / elevation sign / mirror against a known-good output.

Usage:
  python webapp/bake_only.py <uid> [az_offset] [el_sign] [mirror] [corner_weight] [bake_exp]
Defaults reflect current production: az_offset=0 el_sign=-1 mirror=0 corner_weight=0.3 bake_exp=8
"""
import os
import sys
import json

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
AZ_OFF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
EL_SIGN = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
MIRROR = int(sys.argv[4]) if len(sys.argv) > 4 else 0
CORNER_W = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
BAKE_EXP = float(sys.argv[6]) if len(sys.argv) > 6 else 8.0
out = os.environ.get("HY3D_OUTPUT_DIR", "webapp/outputs")
SZ = 512

mesh = trimesh.load(f"{out}/{uid}_shape.glb", force="mesh")
mesh = mesh_uv_wrap(mesh)
rd = MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=SZ, texture_size=2048, device="cuda")
rd.load_mesh(mesh=mesh)
angles = json.load(open(f"{out}/{uid}_mvadapter_angles.json"))


def _is_corner(az, el):
    return abs(el) < 60 and (az % 90) != 0


def load_view(i, az, el):
    # Simulate the production fix: 3/4 corners use the raw MV view (gpt hallucinates them),
    # everything else uses the gpt/recolour output.
    sufs = ("view", "gptview") if _is_corner(az, el) else ("gptview", "view")
    for suf in sufs:
        p = f"{out}/{uid}_mvadapter_{suf}{i}.png"
        if os.path.exists(p):
            return Image.open(p).convert("RGB").resize((SZ, SZ))
    return None


def rembg_thresh(im):
    """Cheap bg removal: pixels close to the corner colour -> transparent. The gpt views
    sit on a plain neutral background, so this isolates the subject for back_project."""
    a = np.asarray(im).astype(int)
    bg = a[0, 0]
    mask = (np.abs(a - bg).sum(-1) > 36).astype(np.uint8) * 255
    rgba = np.dstack([np.asarray(im), mask])
    return Image.fromarray(rgba, "RGBA")


# --- bake all views under the chosen mapping ---
textures, cos_maps = [], []
for i, (az, el) in enumerate(angles):
    v = load_view(i, az, el)
    if v is None:
        continue
    if MIRROR:
        v = v.transpose(Image.FLIP_LEFT_RIGHT)
    H_el = EL_SIGN * el
    H_az = az + AZ_OFF
    w = 1.0 if i < 6 else CORNER_W
    img = rembg_thresh(v)
    # back_project consumes RGBA -> use RGB; alpha-zero the bg by compositing onto mid-grey
    rgb = Image.composite(v, Image.new("RGB", v.size, (127, 127, 127)), img.split()[3])
    tex, cos, _ = rd.back_project(rgb, H_el, H_az)
    textures.append(tex)
    cos_maps.append(w * (cos ** BAKE_EXP))
texture, trust = rd.fast_bake_texture(textures, cos_maps)
mask = (trust > 1e-8)
mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
texture = torch.tensor(rd.uv_inpaint(texture, mask_np) / 255).float().to(texture.device)
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
    samp = F.grid_sample(tex.permute(2, 0, 1)[None], uv * 2 - 1, align_corners=False)[0].permute(1, 2, 0)
    img = samp * vis + (1 - vis) * 1.0
    return Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype("uint8"))


# Canonical faces to view (Hunyuan camera; top viewed from ABOVE = negative elevation).
FACES = [("front", 0.0, 0.0), ("right", 0.0, 90.0), ("back", 0.0, 180.0),
         ("left", 0.0, 270.0), ("top", -89.99, 0.0)]


def elev_img(side):
    p = f"{out}/{uid}_elev_{side}.png"
    return Image.open(p).convert("RGB").resize((SZ, SZ)) if os.path.exists(p) else Image.new("RGB", (SZ, SZ), (40, 40, 40))


sheet = Image.new("RGB", (len(FACES) * SZ, 2 * SZ), (12, 12, 12))
d = ImageDraw.Draw(sheet)
for c, (name, el, az) in enumerate(FACES):
    sheet.paste(elev_img(name), (c * SZ, 0))
    d.text((c * SZ + 5, 5), f"ELEV {name}", fill=(0, 230, 255))
    sheet.paste(render_color(el, az), (c * SZ, SZ))
    d.text((c * SZ + 5, SZ + 5), f"BAKED {name} cam(el={el:.0f},az={az:.0f})", fill=(0, 255, 120))
path = f"{out}/{uid}_bakeonly_az{AZ_OFF:.0f}_els{EL_SIGN:.0f}_m{MIRROR}.png"
sheet.save(path)
print(f"MAPPING az+{AZ_OFF} el*{EL_SIGN} mirror={MIRROR} corner_w={CORNER_W} bake_exp={BAKE_EXP}")
print("saved", path)
