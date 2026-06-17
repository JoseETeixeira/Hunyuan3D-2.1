"""gpt-image geomatch with DEPTH guidance + a SILHOUETTE MASK (runs in the hunyuan env).

Feeds gpt-image-2 the mesh's per-view DEPTH relief (Image 1, on white) + the clean reference
elevation (Image 2), and an edit MASK whose transparent region is the object's exact silhouette.
gpt may only paint inside the silhouette, so elements fit + scale to the real mesh footprint
(fixing the reframe/corner-clustering of plain gpt). Result is confined to the silhouette on white.

Single:  python -m webapp.gpt_depthmask --uid U --dir D --side front --ref_prefix dealership_
Batch :  python -m webapp.gpt_depthmask --uid U --dir D --sides front,left,right,back,top --ref_prefix dealership_
Outputs D/{uid}_cnmatch_{side}.png (so the existing bake spec picks them up).
"""
import argparse
import os

import numpy as np
from PIL import Image

from webapp.image_edit import CARTOON_STYLE, CONSISTENCY_RULE, edit_image

S = 1024
PROMPT = (
    "Image 1 is a GREY SHADED render of one side of a single object. It is the GROUND TRUTH for "
    "LAYOUT and GEOMETRY: the exact silhouette, and the exact position, size, count, shape and "
    "spacing of EVERY element — the building, its roofline, the recessed entrance, each individual "
    "vehicle, and each individual plant/bush/tree are all exactly where and how big Image 1 shows "
    "them. Your job is to COLOURISE Image 1 in place: paint over the grey while keeping every shape, "
    "position, size, count and spacing EXACTLY as in Image 1. Do NOT move, add, remove, duplicate, "
    "resize, merge or rearrange anything; do NOT redraw the scene. "
    ""
    "Image 2 is a COLOUR/MATERIAL REFERENCE for this side — use it ONLY as a palette: which colours "
    "and materials the walls, trim, roof, signage, glass, vehicles and plants should have. Image 2's "
    "element POSITIONS and COUNTS are NOT authoritative and must be IGNORED where they differ from "
    "Image 1 — if Image 2 puts plants in the corners but Image 1 has them spread across the front, "
    "follow Image 1. Match colours to the corresponding part of Image 1 by what it is (a wall stays a "
    "wall, a car stays a car, a plant stays a plant). "
    ""
    "Image 3 is a depth map (near = bright) as an extra 3D cue for which parts are nearer/farther. "
    "Paint only inside the object silhouette; keep a plain white background. "
    + CARTOON_STYLE + " " + CONSISTENCY_RULE
)


def run_side(uid, d, side, ref_prefix):
    geom_p = os.path.join(d, f"{uid}_geom_{side}.png")
    depth_p = os.path.join(d, f"{uid}_depth_{side}.png")
    elev_p = os.path.join(d, f"{ref_prefix}{side}.png")
    if not (os.path.exists(geom_p) and os.path.exists(depth_p) and os.path.exists(elev_p)):
        miss = "geom" if not os.path.exists(geom_p) else ("depth" if not os.path.exists(depth_p) else "elev")
        print(f"[depthmask] skip {side}: missing {miss}")
        return
    depth = Image.open(depth_p).convert("L").resize((S, S))
    obj = np.asarray(depth) > 8
    sil = Image.fromarray((obj * 255).astype("uint8"))               # object silhouette (white on black)
    white = Image.new("RGB", (S, S), (255, 255, 255))
    base = Image.composite(Image.open(geom_p).convert("RGB").resize((S, S)), white, sil)  # GREY GEOM on white
    depth_rgb = Image.composite(Image.open(depth_p).convert("RGB").resize((S, S)), white, sil)
    mask = Image.new("RGBA", (S, S), (0, 0, 0, 255))
    mask.putalpha(Image.fromarray(((~obj) * 255).astype("uint8")))   # transparent over object = editable
    elev = Image.open(elev_p).convert("RGB").resize((S, S))

    res = edit_image([base, elev, depth_rgb], PROMPT, size=(S, S), mask=mask)
    res = Image.composite(res.resize((S, S)), white, sil)            # confine to silhouette, clean bg
    out = os.path.join(d, f"{uid}_cnmatch_{side}.png")
    res.save(out)
    print("DEPTHMASK_DONE:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--side")
    ap.add_argument("--sides")
    ap.add_argument("--ref_prefix", default="dealership_")
    args = ap.parse_args()
    sides = [args.side] if args.side else [s.strip() for s in (args.sides or "").split(",") if s.strip()]
    for s in sides:
        run_side(args.uid, args.dir, s, args.ref_prefix)


if __name__ == "__main__":
    main()
