"""Two-stage corner-view texturing: gpt GENERATES the view from the GEOMETRY render + the colour
references for the faces visible in that view, then Gemini TRANSFERS it onto the geom (exact
proportions).

Stage 1 (gpt-image): the grey GEOMETRY render of the corner is the layout ground truth (exact
composition, proportions, every element's position + size). The colour references are the faces
visible in that corner (e.g. fr -> front + right + top). gpt produces a coloured render that matches
the geometry and is coloured from those faces — it must NOT add/move/resize/omit anything.
Stage 2 (Gemini): colourise the grey geom render USING the gpt view as the colour reference. Gemini
keeps the geom's proportions/layout exactly (gpt drifts ~12%, Gemini does not).

Run: python -m webapp.gen_transfer --uid U --dir D --sides fr,fl,br,bl --ref_prefix dealership_
Outputs D/{uid}_genview_{side}.png (stage 1) and D/{uid}_cnmatch_{side}.png (stage 2, used by the bake).
"""
import argparse
import os

from PIL import Image

from webapp.image_edit import CARTOON_STYLE, edit_image

S = 1024
# Per-view projection + which faces ARE and are NOT visible. Stated explicitly so gpt does not
# bend perspective or paint a face that this view cannot see (e.g. the front showing on a side view).
VIEW_SPEC = {
    "front": ("a straight-on ORTHOGRAPHIC FRONT view (camera level, looking straight at the front). "
              "ONLY the front face is visible. The back, left side, right side, roof/top and underside "
              "are NOT visible and must NOT appear."),
    "back": ("a straight-on ORTHOGRAPHIC BACK view. ONLY the back face is visible. The front, left, "
             "right, top and bottom are NOT visible and must NOT appear."),
    "left": ("a straight-on ORTHOGRAPHIC LEFT-SIDE view. ONLY the left side is visible. The front, "
             "back, right side and top are NOT visible and must NOT appear."),
    "right": ("a straight-on ORTHOGRAPHIC RIGHT-SIDE view. ONLY the right side is visible. The front, "
              "back, left side and top are NOT visible and must NOT appear."),
    "top": ("an ORTHOGRAPHIC ABSOLUTE TOP-DOWN view (nadir, looking straight down from directly above). "
            "ONLY the roof and upward-facing surfaces are visible. The front, back, left and right faces "
            "are NOT visible — show no facade, walls or sides, only the top as seen from straight above."),
    "bottom": ("an ORTHOGRAPHIC ABSOLUTE BOTTOM-UP view (looking straight up from directly below). ONLY "
               "the underside is visible; no walls or roof."),
    "fr": ("an ORTHOGRAPHIC 3/4 FRONT-RIGHT view (camera slightly above). The FRONT face, RIGHT side and "
           "TOP are visible; the back and left side are NOT visible and must NOT appear."),
    "fl": ("an ORTHOGRAPHIC 3/4 FRONT-LEFT view (camera slightly above). The FRONT face, LEFT side and "
           "TOP are visible; the back and right side are NOT visible and must NOT appear."),
    "br": ("an ORTHOGRAPHIC 3/4 BACK-RIGHT view (camera slightly above). The BACK face, RIGHT side and "
           "TOP are visible; the front and left side are NOT visible and must NOT appear."),
    "bl": ("an ORTHOGRAPHIC 3/4 BACK-LEFT view (camera slightly above). The BACK face, LEFT side and "
           "TOP are visible; the front and right side are NOT visible and must NOT appear."),
    "fr_hi": ("a HIGH 3/4 FRONT-RIGHT view from well ABOVE (camera high, looking steeply down ~55deg). "
              "The TOP/roof is DOMINANT (most of the frame), with the FRONT face and RIGHT side seen "
              "foreshortened below it; the back and left side are NOT visible and must NOT appear."),
    "fl_hi": ("a HIGH 3/4 FRONT-LEFT view from well ABOVE (camera high, looking steeply down ~55deg). "
              "The TOP/roof is DOMINANT (most of the frame), with the FRONT face and LEFT side seen "
              "foreshortened below it; the back and right side are NOT visible and must NOT appear."),
    "br_hi": ("a HIGH 3/4 BACK-RIGHT view from well ABOVE (camera high, looking steeply down ~55deg). "
              "The TOP/roof is DOMINANT (most of the frame), with the BACK face and RIGHT side seen "
              "foreshortened below it; the front and left side are NOT visible and must NOT appear."),
    "bl_hi": ("a HIGH 3/4 BACK-LEFT view from well ABOVE (camera high, looking steeply down ~55deg). "
              "The TOP/roof is DOMINANT (most of the frame), with the BACK face and LEFT side seen "
              "foreshortened below it; the front and right side are NOT visible and must NOT appear."),
}
# Faces visible in each view -> feed gpt those exact face references (colour source). Cardinal views
# use their own face; 3/4 corners use the two adjacent sides + top.
ADJ = {
    "front": ["front"], "back": ["back"], "left": ["left"], "right": ["right"], "top": ["top"],
    "bottom": ["bottom"],
    "fr": ["front", "right", "top"], "fl": ["front", "left", "top"],
    "br": ["back", "right", "top"], "bl": ["back", "left", "top"],
    # high tilt: top dominant -> top first as the primary colour reference.
    "fr_hi": ["top", "front", "right"], "fl_hi": ["top", "front", "left"],
    "br_hi": ["top", "back", "right"], "bl_hi": ["top", "back", "left"],
}
TRANSFER = ("Colorize the grayscale image using the reference (colored image). Keep the grayscale "
            "proportions, shape, components and silhouette EXACTLY as-is, only apply the colors, do "
            "not change composition, props, orientation or any geometry present on the grayscale image, try to match colors to elements by position in the reference image "
            "and do not add any new elements or props that are not present in the grayscale image. ")


def _open(x):
    return x if isinstance(x, Image.Image) else Image.open(x).convert("RGB")


def gen_view_paths(side, ref_paths, geom_path, consistency_refs=None, style_anchors=None):
    """gpt-image generates the `side` view FROM THE GEOMETRY render (Image 1 = exact layout/
    proportions) coloured using the face references. Geometry is the ground truth for what exists;
    the references supply colours. `consistency_refs` (other faces) keep shared objects consistent;
    `style_anchors` (previously generated views) normalise the art style across views. Accepts paths
    or PIL images. Returns a PIL image. Reused by the server."""
    faces = ADJ.get(side, [side])
    faces_txt = (" and ".join(f.upper() for f in faces) + " faces") if faces else "this side"
    consistency_refs = list(consistency_refs or [])[:3]
    style_anchors = list(style_anchors or [])[:2]
    parts = [f"the next {len(ref_paths)} image(s) are the {faces_txt} (PRIMARY colour reference for this view)"]
    if consistency_refs:
        parts.append(f"the next {len(consistency_refs)} image(s) are OTHER faces/renders of the SAME asset — use "
                     "them to keep every shared object's identity, colour and shape consistent across views (a blue "
                     "car stays the SAME blue car; a tan car stays tan). This is an ANGLED view: if Image 1's "
                     "geometry shows an object that also appears in these refs — even partially, at an edge, or "
                     "foreshortened by the angle — RENDER it the same way; do NOT drop it and do NOT swap it for a "
                     "different object (never turn a car into a bush/blob). But do NOT add an object Image 1 does not "
                     "show, do NOT copy their layout/viewpoint, and do NOT paint a feature that belongs to a face "
                     "THIS view cannot see (e.g. no front emblem, logo or facade on a side or back view)")
    if style_anchors:
        parts.append(f"the final {len(style_anchors)} image(s) are FINISHED renders of this SAME asset — match their "
                     "exact art style, palette, colour saturation, shading and level of detail so every view looks "
                     "like one consistent set, AND use them to identify shared objects (e.g. the cars) so each keeps "
                     "the SAME colour and shape here as there")
    groups_txt = "After Image 1, " + "; ".join(parts) + "."
    prompt = (
        f"Image 1 is a grey GEOMETRY render of a single 3D asset, shown as {VIEW_SPEC.get(side, 'a view')} "
        f"Produce a COLOURED render that MATCHES IMAGE 1's GEOMETRY EXACTLY: same ORTHOGRAPHIC projection "
        f"(no perspective/foreshortening), identical silhouette, composition and proportions, every element at "
        f"the same position and size. Do NOT stretch, skew, resize, rotate, move, add, remove or duplicate "
        f"anything; Image 1 is the SOLE AUTHORITY for WHAT EXISTS and WHERE. "
        f"Render EVERY object in Image 1 as its ACTUAL type, matched to its silhouette: a vehicle-shaped object "
        f"is a VEHICLE/CAR (NEVER vegetation, bushes, rocks or a grey blob); a bush is a bush; a lamp is a lamp. "
        f"Carefully SCAN the ground/base of Image 1 for VEHICLES and render each one as a vehicle — do NOT replace "
        f"a vehicle with a lamp, bollard or bush, and do NOT omit it. Add NO object (no lamp, bollard, bush, plant "
        f"or prop) that Image 1's geometry does not actually contain. "
        f"Fully colour every object — never leave one grey or omit it. Colour each object from the PRIMARY "
        f"reference for this view; DIFFERENT objects keep their OWN colours (e.g. a blue car on one side and a tan "
        f"car on another are SEPARATE cars — do NOT make all cars the same colour). If an object's colour is "
        f"unclear, infer a plausible one, but always render it fully coloured and recognizable as its type. Do "
        f"NOT add any object Image 1 does not show; where Image 1 shows empty ground, keep it empty. For any prop "
        f"partly occluded or cut off in this view, render the hidden part consistently with its visible part so "
        f"nothing looks broken. {groups_txt} Single object, plain white background, even flat lighting. " + CARTOON_STYLE)
    imgs = ([_open(geom_path)] + [_open(r) for r in ref_paths]
            + [_open(r) for r in consistency_refs] + [_open(r) for r in style_anchors])
    return edit_image(imgs, prompt, size=(S, S), prefer="openai")  # gpt-image generates the angle


def transfer(geom_path, gen_img, style_anchors=None):
    """Gemini RECOLOURS the grey geom render using gen_img (the genview) as the colour reference,
    keeping the geom's EXACT view, silhouette and proportions. NO style anchor: the genview is already
    style-consistent, and a cross-face anchor bleeds the wrong face's content/shape into the output
    (e.g. a side view picking up the front emblem/facade). `style_anchors` is accepted but ignored.
    Accepts paths/PIL."""
    prompt = (
        "COLOURISE Image 1, a grayscale GEOMETRY render — a RECOLOUR of Image 1's exact pixels, NOT a redraw. "
        "Image 1 ALONE defines WHICH view this is and its exact silhouette, composition and proportions. The "
        "output MUST be pixel-aligned to Image 1: identical outline, every element at Image 1's exact position, "
        "shape and size. Do NOT change the viewpoint, do NOT widen, narrow, stretch or reshape, do NOT move, add, "
        "remove, duplicate or invent anything, and do NOT introduce features from a DIFFERENT face — e.g. never put "
        "a front emblem, logo or front facade onto a side/back view. Take the COLOURS, materials and flat "
        "3D-cartoon look from the colour reference (Image 2), which is the SAME view — match each colour to the "
        "element at the same position. Some props may be occluded or partial in Image 2 — wherever Image 1 shows "
        "an object Image 2 hides, still colour that whole object from Image 1's shape (a vehicle stays a vehicle, "
        "never a blob). Do not add elements not in Image 1. " + CARTOON_STYLE)
    imgs = [_open(geom_path).resize((S, S)), _open(gen_img).resize((S, S))]
    return edit_image(imgs, prompt, size=(S, S), prefer="gemini")


def _is_h_mirrored(out_img, base_img, margin=0.03):
    """True if out_img is a horizontal mirror of base_img. The image model occasionally flips a SIDE
    view left<->right while restyling. Compares bbox-normalised silhouettes flipped vs not; the margin
    avoids flipping a near-symmetric (front/back/top) view, where a flip would not matter anyway."""
    import numpy as np

    def _silbox(img, s=160):
        a = np.asarray(img.convert("RGB").resize((256, 256))).astype(np.int16)
        m = np.abs(a - a[0, 0]).sum(-1) > 24          # background = top-left corner colour
        ys, xs = np.where(m)
        if xs.size == 0:
            return None
        crop = (m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255)
        return np.asarray(Image.fromarray(crop).resize((s, s))) > 127

    sb = _silbox(base_img); so = _silbox(out_img)
    if sb is None or so is None:
        return False
    iou = lambda x, y: (x & y).sum() / ((x | y).sum() or 1)
    return iou(so[:, ::-1], sb) > iou(so, sb) + margin


def restyle_to_references(base_render, ref_paths, max_refs=3):
    """Reface restyle: push the reference images' look onto a REAL textured-mesh render while holding
    its geometry EXACTLY.

    `base_render` is the current mesh rendered at the face camera — a COMPLETE colour image with the
    exact geometry and the colours base-texturing already baked. Because Image 1 is a full-colour
    render (not a grey geom), the model keeps ITS outline/proportions and only takes colour/material/
    style from the references; a grey or partial Image 1 instead abandons its shape to a colour
    reference, which is the genview-geometry-drift bug. Single structural image + references as style
    only. Accepts paths/PIL; returns a PIL image."""
    base = _open(base_render)
    refs = [_open(r) for r in (ref_paths or [])][:max_refs]
    prompt = (
        "Image 1 is the asset to KEEP. The remaining image(s) are STYLE/COLOUR references of the SAME "
        "asset from a DIFFERENT viewpoint — their framing, scale, shape and layout are IRRELEVANT and "
        "MUST be ignored. Output Image 1 with its geometry, outline, composition and EVERY element's "
        "position, shape and size reproduced PIXEL-FOR-PIXEL (the result must overlay Image 1 exactly), "
        "only refining its colours, materials and shading toward the reference palette and style. Do NOT "
        "adopt the references' viewpoint, scale or layout; do NOT move, resize, add, remove or reshape "
        "anything from Image 1. Plain pure-white background. " + CARTOON_STYLE)
    out = edit_image([base] + refs, prompt, size=(S, S), prefer="gemini")
    # The model occasionally mirrors a side view (left<->right) while restyling. base is the real mesh
    # render — ground truth and in back_project's handedness — so undo any flip against it, else the
    # bake paints the left face onto the right.
    if _is_h_mirrored(out, base):
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sides", required=True)
    ap.add_argument("--ref_prefix", default="dealership_")
    a = ap.parse_args()
    cardinal = ["front", "back", "left", "right", "top"]
    for s in [x.strip() for x in a.sides.split(",") if x.strip()]:
        geom = os.path.join(a.dir, f"{a.uid}_geom_{s}.png")
        if not os.path.exists(geom):
            print(f"[gen_transfer] skip {s}: no geom"); continue
        faces = ADJ.get(s, [s])
        refs = [os.path.join(a.dir, f"{a.ref_prefix}{f}.png") for f in faces]
        refs = [r for r in refs if os.path.exists(r)]
        cons = [os.path.join(a.dir, f"{a.ref_prefix}{f}.png") for f in cardinal if f not in faces]
        cons = [r for r in cons if os.path.exists(r)]
        gv = gen_view_paths(s, refs, geom, consistency_refs=cons)
        gv.save(os.path.join(a.dir, f"{a.uid}_genview_{s}.png"))
        out = transfer(geom, gv)
        out.save(os.path.join(a.dir, f"{a.uid}_cnmatch_{s}.png"))
        print("GEN_TRANSFER_DONE:", s)


if __name__ == "__main__":
    main()
