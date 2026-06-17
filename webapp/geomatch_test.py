"""Test a structure-locked geomatch prompt on ONE side. Regenerates elevmatched_<side>_v2.png
from the clean elevation + grey geometry render, with a prompt that forbids resizing/reframing.

Run (in repo root):  python -m webapp.geomatch_test <uid> <side>
"""
import sys

from PIL import Image

from webapp.image_edit import CARTOON_STYLE, CONSISTENCY_RULE, edit_image

OUT = "/workspace/Hunyuan3D-2.1/webapp/outputs"

uid = sys.argv[1]
side = sys.argv[2]

# Structure-locked: Image 1 (grey geom) defines the EXACT silhouette/size/position; gpt only
# paints Image 2's appearance onto it, edge-to-edge, with no margin or repositioning.
PROMPT = (
    "Image 1 is a GREYSCALE shape render giving the EXACT silhouette, size, position and surface "
    "relief of one side of an object. Image 2 is the clean coloured art for that same side. Repaint "
    "Image 1 using the colours, materials, panels, windows, doors, signage, vehicles, plants and "
    "details from Image 2. CRITICAL ALIGNMENT RULES: the result MUST match Image 1's silhouette "
    "exactly — identical outline, identical overall size and position on the canvas, filling the "
    "same area edge-to-edge. Do NOT shrink, enlarge, recenter, rotate, or add any margin or padding. "
    "Place every feature on the corresponding part of Image 1's geometry: roofline on the roof, base "
    "at the base, openings where the relief shows recesses, props where the relief bumps out. Keep "
    "Image 2's exact colours and materials; do not recolour, remove or restyle anything. Plain flat "
    "background. " + CARTOON_STYLE + " " + CONSISTENCY_RULE
)

geom = f"{OUT}/{uid}_geom_{side}.png"
elev = f"{OUT}/{uid}_elev_{side}.png"
res = edit_image([Image.open(geom).convert("RGB"), Image.open(elev).convert("RGB")],
                 PROMPT, size=(1024, 1024))
mp = f"{OUT}/{uid}_elevmatched_{side}_v2.png"
res.save(mp)
print("GEOMATCH_V2_DONE:", mp)
