"""Generate clean canonical ELEVATIONS (front/left/right/back/top) of a single object
from one 3/4 source view, using gpt-image (Gemini fallback).

The 3/4 source shows ~3 faces; we synthesise flat head-on views of every side so the
texturing pipeline has an authoritative per-side appearance to transfer onto the mesh.
Views are generated SEQUENTIALLY and each call is fed the source plus every elevation
produced so far — this mimics a chat session, keeping colours/elements consistent
across sides. Sides the caller already has (user uploaded + side-tagged) are used as-is;
only missing sides are generated. Bottom is intentionally skipped (the source carries no
underside information; the mesh's raw view is kept for it).
"""
import os

from PIL import Image

from webapp.image_edit import CARTOON_STYLE, CONSISTENCY_RULE, edit_image

# Order matters: front carries the most signal from a 3/4 view, then the sides, then the
# (unseen) back, then top. Each is generated with all earlier results as context.
ELEVATION_SIDES = ["front", "left", "right", "back", "top"]

_SIDE_DESC = {
    "front": "the FRONT face — the main facade with the primary entrance and signage — viewed "
             "perfectly head-on (orthographic, zero rotation, no perspective foreshortening)",
    "left": "the LEFT side, viewed perfectly head-on (the face 90 degrees to the left of the front)",
    "right": "the RIGHT side, viewed perfectly head-on (the face 90 degrees to the right of the front)",
    "back": "the BACK face, viewed perfectly head-on (infer it consistently from the other sides; "
            "keep the same roofline, colours and materials, just without the front entrance/signage)",
    "top": "the TOP, looking straight down (a roof plan / bird's-eye orthographic view)",
}


def _elev_prompt(side, n_context):
    consistency = (
        "The images after Image 1 are clean elevations of the SAME object already produced — "
        "match them EXACTLY (identical colours, materials, panels, windows, signage, trim, "
        "proportions and art style); this is another side of that same object. "
        if n_context > 1 else ""
    )
    return (
        "Image 1 is a 3/4 reference render of a single stylised building/object. " + consistency +
        f"Generate a CLEAN, FLAT, HEAD-ON elevation of {_SIDE_DESC[side]}. "
        "Output a single centered object on a plain neutral background, orthographic, no ground "
        "plane, no extra props, no scenery. Keep the object's EXACT colours, materials, panels, "
        "windows, signage, trim, proportions and design — do NOT add, remove, restyle or "
        "reinterpret elements; only the camera viewpoint changes versus the references. " +
        CARTOON_STYLE + " " + CONSISTENCY_RULE
    )


def generate_elevations(source_path, provided=None, extra_context=None, out_dir=".", uid="obj",
                        sides=None, size=(1024, 1024)):
    """Return {side: path} of canonical elevations for `sides` (default ELEVATION_SIDES).

    `provided` {side: path} are already-correct elevations (user uploads) — reused, not
    regenerated. `extra_context` are additional reference views of the object fed into
    every generation for consistency. Each generated side is appended to the running
    context so later sides stay consistent (chat-like memory).
    """
    provided = provided or {}
    sides = sides or ELEVATION_SIDES
    src = Image.open(source_path).convert("RGB")
    context = [src] + [Image.open(p).convert("RGB") for p in (extra_context or []) if os.path.exists(p)]
    result = {}
    for side in sides:
        path = os.path.join(out_dir, f"{uid}_elev_{side}.png")
        if provided.get(side) and os.path.exists(provided[side]):
            img = Image.open(provided[side]).convert("RGB")
            img.save(path)
            print(f"[elevations] {side}: using provided reference")
        else:
            img = edit_image(context, _elev_prompt(side, len(context)), size=size)
            img.save(path)
            print(f"[elevations] {side}: generated")
        result[side] = path
        context.append(img)  # later sides see this one -> consistency
    return result
