"""Render a finished textured GLB from the 5 canonical cameras (front/right/back/left/top)
so face assignment can be compared between jobs. Read-only. Usage:
  python webapp/glb_faces.py <uid> [<uid2> ...]
"""
import os
import sys

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from DifferentiableRenderer.MeshRender import MeshRender, get_mv_matrix, transform_pos  # noqa: E402

out = os.environ.get("HY3D_OUTPUT_DIR", "webapp/outputs")
SZ = 360
FACES = [("front", 0.0, 0.0), ("right", 0.0, 90.0), ("back", 0.0, 180.0),
         ("left", 0.0, 270.0), ("top", -89.99, 0.0)]
uids = sys.argv[1:]


def render_color(rd, elev, azim, size=SZ):
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


def load_textured(glb):
    """Load a textured GLB into a MeshRender (geometry+uv via the renderer util, then the
    embedded baseColor texture, which the util does NOT read). Handles Scene exports."""
    import numpy as np
    loaded = trimesh.load(glb)
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        geom = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(tuple(geoms))
    else:
        geom = loaded
    rd = MeshRender(camera_distance=1.45, camera_type="orth", default_resolution=SZ, texture_size=2048, device="cuda")
    rd.load_mesh(mesh=geom)
    mat = getattr(geom.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is not None:
        rd.set_texture(np.asarray(img.convert("RGB")).astype("float32") / 255.0)
    else:
        print(f"[glb_faces] WARNING: no embedded texture for {glb}")
    return rd


sheet = Image.new("RGB", (len(FACES) * SZ, len(uids) * SZ), (12, 12, 12))
d = ImageDraw.Draw(sheet)
for r, uid in enumerate(uids):
    glb = f"{out}/{uid}_textured.glb"
    if not os.path.exists(glb):
        print(f"[glb_faces] missing {glb}; skipping")
        continue
    rd = load_textured(glb)
    for c, (name, el, az) in enumerate(FACES):
        sheet.paste(render_color(rd, el, az), (c * SZ, r * SZ))
        d.text((c * SZ + 5, r * SZ + 5), f"{uid[:8]} {name}", fill=(0, 255, 120))
path = f"{out}/_glbfaces_{'_'.join(u[:8] for u in uids)}.png"
sheet.save(path)
print("saved", path)
