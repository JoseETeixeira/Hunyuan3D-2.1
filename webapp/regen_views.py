"""Targeted regen for one existing job: regenerate selected genviews, then re-transfer ALL
elevmatched with style normalisation (the FRONT face is the style anchor for every other face).
Rebake separately (blender_project on the job's proj spec).

Run: python -m webapp.regen_views --uid U --dir D --regen bl,br,right --ref_prefix dealership_
"""
import argparse
import os

from PIL import Image

from webapp.gen_transfer import ADJ, gen_view_paths, transfer

ALL = ["front", "back", "left", "right", "top", "fr", "fl", "br", "bl",
       "fr_hi", "fl_hi", "br_hi", "bl_hi"]
CARD = ["front", "back", "left", "right", "top"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--regen", default="")       # genviews to regenerate
    ap.add_argument("--transfer", default="")    # elevmatched to re-transfer (default: the regen'd sides)
    ap.add_argument("--ref_prefix", default="dealership_")
    a = ap.parse_args()
    d, uid, rp = a.dir, a.uid, a.ref_prefix
    regen = [s.strip() for s in a.regen.split(",") if s.strip()]
    transfer_sides = [s.strip() for s in a.transfer.split(",") if s.strip()] or regen

    front_gv = os.path.join(d, f"{uid}_genview_front.png")
    gv_anchor = [front_gv] if os.path.exists(front_gv) else []

    # 1) regenerate the requested genviews (style-anchored on the front genview)
    for s in regen:
        geom = os.path.join(d, f"{uid}_geom_{s}.png")
        if not os.path.exists(geom):
            print(f"skip genview {s}: no geom"); continue
        faces = ADJ.get(s, [s])
        refs = [os.path.join(d, f"{rp}{f}.png") for f in faces]
        refs = [r for r in refs if os.path.exists(r)]
        cons = [os.path.join(d, f"{rp}{f}.png") for f in CARD if f not in faces]
        cons = [r for r in cons if os.path.exists(r)]
        gv = gen_view_paths(s, refs, geom, consistency_refs=cons,
                            style_anchors=(gv_anchor if s != "front" else []))
        gv.save(os.path.join(d, f"{uid}_genview_{s}.png"))
        print("GENVIEW_DONE", s)

    # 2) re-transfer the requested elevmatched (genview -> geom; no style anchor, to avoid cross-face bleed)
    for s in transfer_sides:
        geom = os.path.join(d, f"{uid}_geom_{s}.png")
        gvp = os.path.join(d, f"{uid}_genview_{s}.png")
        if not (os.path.exists(geom) and os.path.exists(gvp)):
            print(f"skip transfer {s}: missing geom/genview"); continue
        out = transfer(geom, Image.open(gvp).convert("RGB"))
        out.save(os.path.join(d, f"{uid}_elevmatched_{s}.png"))
        print("ELEV_DONE", s)
    print("REGEN_DONE")


if __name__ == "__main__":
    main()
