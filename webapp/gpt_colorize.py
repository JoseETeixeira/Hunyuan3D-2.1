"""gpt-image geomatch by COLOURISING the grey geometry render with the reference elevation.

The grey geom render is the layout/geometry ground truth (every element where the mesh has it);
gpt colourises it using the reference elevation purely as the colour source. Empirically a short
literal prompt preserves the geometry far better than verbose instructions or depth/mask tricks
(gpt-image reproduces a reference's layout when over-instructed, but a plain "colourise this image"
keeps the input's structure).

Batch: python -m webapp.gpt_colorize --uid U --dir D --sides front,left,right,back,top --ref_prefix dealership_
Single: python -m webapp.gpt_colorize --uid U --dir D --side front --ref_prefix dealership_
Outputs D/{uid}_cnmatch_{side}.png (the bake spec picks these up).
"""
import argparse
import os

from PIL import Image

from webapp.image_edit import edit_image

S = 1024
PROMPT = ("Colorize the grayscale image using the reference (colored image). Keep the grayscale "
          "proportions, shape, components and silhouette EXACTLY as-is, only apply the colors, do "
          "not change composition, props, orientation or any geometry present on the grayscale image")


def run_side(uid, d, side, ref_prefix, ref_img=None):
    geom_p = os.path.join(d, f"{uid}_geom_{side}.png")
    elev_p = os.path.join(d, ref_img) if ref_img else os.path.join(d, f"{ref_prefix}{side}.png")
    if not (os.path.exists(geom_p) and os.path.exists(elev_p)):
        print(f"[colorize] skip {side}: missing {'geom' if not os.path.exists(geom_p) else 'elev'}")
        return
    geom = Image.open(geom_p).convert("RGB").resize((S, S))   # image 1 = grayscale geometry
    elev = Image.open(elev_p).convert("RGB").resize((S, S))   # image 2 = colour reference
    res = edit_image([geom, elev], PROMPT, size=(S, S), prefer="gemini")  # Gemini keeps proportions
    out = os.path.join(d, f"{uid}_cnmatch_{side}.png")
    res.save(out)
    print("COLORIZE_DONE:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--side")
    ap.add_argument("--sides")
    ap.add_argument("--ref_prefix", default="dealership_")
    ap.add_argument("--ref_img", help="fixed reference image (in dir) for ALL sides, e.g. the 3/4 source for corners")
    a = ap.parse_args()
    sides = [a.side] if a.side else [s.strip() for s in (a.sides or "").split(",") if s.strip()]
    for s in sides:
        run_side(a.uid, a.dir, s, a.ref_prefix, ref_img=a.ref_img)


if __name__ == "__main__":
    main()
