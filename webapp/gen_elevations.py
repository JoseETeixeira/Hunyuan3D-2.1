"""Standalone preview of the mvgpt elevation step: turn one 3/4 source image into clean
canonical elevations (front/left/right/back/top) WITHOUT running MV-Adapter or a bake.

Usage:
  python webapp/gen_elevations.py <source.png> [out_dir] [uid]
Needs OPENAI_API_KEY and/or GEMINI_API_KEY in the environment. Writes <uid>_elev_<side>.png
and a <uid>_elev_montage.png contact sheet next to them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

from webapp.elevations import ELEVATION_SIDES, generate_elevations  # noqa: E402

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else (os.path.dirname(os.path.abspath(src)) or ".")
uid = sys.argv[3] if len(sys.argv) > 3 else "preview"

res = generate_elevations(src, out_dir=out, uid=uid)
for side, path in res.items():
    print(f"{side}: {path}")

# contact sheet: source + each elevation
S = 320
cells = [("source", src)] + [(s, res[s]) for s in ELEVATION_SIDES if s in res]
sheet = Image.new("RGB", (len(cells) * S, S), (12, 12, 12))
d = ImageDraw.Draw(sheet)
for c, (lbl, p) in enumerate(cells):
    try:
        im = Image.open(p).convert("RGB").resize((S, S))
    except Exception:
        im = Image.new("RGB", (S, S), (40, 40, 40))
    sheet.paste(im, (c * S, 0))
    d.text((c * S + 5, 5), lbl, fill=(255, 230, 0))
montage = os.path.join(out, f"{uid}_elev_montage.png")
sheet.save(montage)
print("montage:", montage)
