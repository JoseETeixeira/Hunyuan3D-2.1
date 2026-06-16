"""Bake the cardinal elevations (top at TOP_EL) and render the result in the SAVED GLB's
own coordinate frame (what model-viewer shows), from +Y (top), -Y (bottom) and +Z (front),
so top/bottom orientation is settled empirically rather than by algebra.

Usage: python webapp/diag_glbframe.py <uid> [TOP_EL]
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
from DifferentiableRenderer.MeshRender import MeshRender  # noqa: E402
from utils.uvwrap_utils import mesh_uv_wrap  # noqa: E402

uid = sys.argv[1]
TOP_EL = float(sys.argv[2]) if len(sys.argv) > 2 else -89.99
out = os.environ.get("HY3D_OUTPUT_DIR", "webapp/outputs")
SZ = 380
DEV = "cuda"

mesh = trimesh.load(f"{out}/{uid}_shape.glb", force="mesh")
mesh = mesh_uv_wrap(mesh)
rd = MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=SZ, texture_size=2048, device=DEV)
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
texture = torch.tensor(rd.uv_inpaint(texture, mask_np) / 255).float().to(DEV)
rd.set_texture(texture, force_set=True)

# GLB-frame geometry (inverts the render-frame transform); reuse the render-frame UVs that
# render_color samples correctly (UV is intrinsic, unchanged by the coordinate transform).
v_glb = torch.tensor(rd.get_mesh(normalize=True)[0], dtype=torch.float32, device=DEV)
uv = rd.vtx_uv[None, ...]


def _look_at(eye, up):
    eye = np.asarray(eye, np.float32); up = np.asarray(up, np.float32)
    f = -eye / np.linalg.norm(eye)          # look toward origin
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, np.float32)
    M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def _ortho(r=0.7, n=0.01, fr=10.0):
    M = np.eye(4, np.float32)
    M[0, 0] = 1 / r; M[1, 1] = 1 / r; M[2, 2] = -2 / (fr - n); M[2, 3] = -(fr + n) / (fr - n)
    return M


def render_glb(eye, up):
    mvp = torch.tensor(_ortho() @ _look_at(eye, up), device=DEV)
    vh = torch.cat([v_glb, torch.ones(len(v_glb), 1, device=DEV)], 1)
    clip = (vh @ mvp.T)[None].contiguous()
    rast, _ = rd.raster_rasterize(clip, rd.pos_idx, resolution=(SZ, SZ))
    uvm, _ = rd.raster_interpolate(uv, rast, rd.uv_idx)
    vis = torch.clamp(rast[..., -1:], 0, 1)[0]
    tex = rd.tex.to(DEV).float()
    if tex.max() > 1.5:
        tex = tex / 255.0
    samp = F.grid_sample(tex.permute(2, 0, 1)[None], uvm * 2 - 1, align_corners=False)[0].permute(1, 2, 0)
    img = samp * vis + (1 - vis) * 1.0
    return Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype("uint8"))


views = [("GLB +Y (top)", (0, 3, 0), (0, 0, 1)),
         ("GLB -Y (bottom)", (0, -3, 0), (0, 0, 1)),
         ("GLB +Z (front)", (0, 0, 3), (0, 1, 0)),
         ("GLB +X (side)", (3, 0, 0), (0, 1, 0))]
sheet = Image.new("RGB", (len(views) * SZ, SZ), (12, 12, 12))
d = ImageDraw.Draw(sheet)
for c, (lbl, eye, up) in enumerate(views):
    sheet.paste(render_glb(eye, up), (c * SZ, 0))
    d.text((c * SZ + 5, 5), lbl, fill=(0, 255, 120))
path = f"{out}/{uid}_glbframe_TOPEL{TOP_EL:.0f}.png"
sheet.save(path)
print(f"top baked at el={TOP_EL}; GLB-frame render saved {path}")
