"""Staged, mesh-free reference-view generation for the per-model studio.

From a single seed image, synthesise the 10 canonical orthographic reference views with
gpt-image-2 (Gemini fallback) along an imperative dependency graph:

    front              <- seed
    back, top, bottom  <- front (+ seed)
    left, right        <- front + top (+ seed)
    front corners      <- front + (left | right) + top   (fl: front+left+top, fr: front+right+top)
    back corners       <- back  + (left | right) + top   (bl: back+left+top, br: back+right+top)

Each view is generated from the seed plus the already-approved views it depends on (fed in
dependency order so the model keeps colours/identity consistent across sides). NO mesh and NO
geometry render are involved — this runs before any mesh exists. Per-view prompts reuse the
orthographic descriptions and face-visibility vocabulary from `gen_transfer` so each view shows
ONLY the faces visible in it. Uses `image_edit.edit_image` (gpt-image-2 default, Gemini fallback).
"""
import os

from PIL import Image

from webapp.image_edit import CARTOON_STYLE, CONSISTENCY_RULE, edit_image

# Frontend ViewId (hyphenated) <-> backend angle tag (gen_transfer / hyface convention).
VIEW_TO_TAG = {
    "front": "front", "back": "back", "left": "left", "right": "right",
    "top": "top", "bottom": "bottom",
    "front-left": "fl", "front-right": "fr", "back-left": "bl", "back-right": "br",
}
ALL_VIEWS = list(VIEW_TO_TAG.keys())

# Imperative dependency graph (mirrors the frontend lib/views.ts VIEW_INPUTS): which
# already-approved views feed each view's generation (alongside the seed image).
VIEW_INPUTS = {
    "front": [],
    "back": ["front"],
    # left/right need only front + top (the side doesn't show the back): TOP gives the overhead
    # layout, FRONT gives appearance.
    "left": ["front", "top"],
    "right": ["front", "top"],
    "top": ["front"],
    "bottom": ["front"],
    # Each corner gets ONLY the side it shows (not the opposite side) + top, so the wrong-side
    # reference can't leak in and flip left/right.
    "front-left": ["front", "left", "top"],
    "front-right": ["front", "right", "top"],
    "back-left": ["back", "left", "top"],
    "back-right": ["back", "right", "top"],
}

# Plain-language rotation cue per view (so the prompt is explicit about the camera move).
_ROTATION = {
    "front": "looking straight at the front",
    "back": "the front rotated 180 degrees (looking straight at the back)",
    "left": ("the model rotated 90 degrees COUNTER-CLOCKWISE about its vertical axis as seen from "
             "directly above — the front swings away to the right and the object's LEFT side (the side "
             "on the LEFT edge of the front view) turns to face the camera head-on"),
    "right": ("the model rotated 90 degrees CLOCKWISE about its vertical axis as seen from directly "
              "above — the front swings away to the left and the object's RIGHT side (the side on the "
              "RIGHT edge of the front view) turns to face the camera head-on"),
    "top": "looking straight down from directly above (nadir)",
    "bottom": "looking straight up from directly below",
    "front-left": ("the FRONT rotated 45 degrees COUNTER-CLOCKWISE about its vertical axis (seen from "
                   "directly above), camera slightly above — the front-left 3/4 corner"),
    "front-right": ("the FRONT rotated 45 degrees CLOCKWISE about its vertical axis (seen from directly "
                    "above), camera slightly above — the front-right 3/4 corner"),
    "back-left": ("the BACK rotated 45 degrees CLOCKWISE about its vertical axis (seen from directly "
                  "above), camera slightly above — the back-left 3/4 corner"),
    "back-right": ("the BACK rotated 45 degrees COUNTER-CLOCKWISE about its vertical axis (seen from "
                   "directly above), camera slightly above — the back-right 3/4 corner"),
}

# Per-view orthographic framing (scene-preserving: describes the camera, not "isolate the object").
_VIEW_FRAME = {
    "front": "a straight-on, head-on ORTHOGRAPHIC FRONT view (camera at eye level, looking directly at the front of the scene)",
    "back": "a straight-on, head-on ORTHOGRAPHIC BACK view (looking directly at the back of the scene)",
    "left": "a straight-on, head-on ORTHOGRAPHIC LEFT-SIDE view (the scene seen directly from its left)",
    "right": "a straight-on, head-on ORTHOGRAPHIC RIGHT-SIDE view (the scene seen directly from its right)",
    "top": "an ORTHOGRAPHIC TOP-DOWN view (looking straight down from directly above — a bird's-eye plan showing every upper surface)",
    "bottom": "an ORTHOGRAPHIC BOTTOM-UP view (looking straight up from directly below — the underside)",
    "fl": "an ORTHOGRAPHIC 3/4 FRONT-LEFT view (camera slightly above, showing the front and the left side together)",
    "fr": "an ORTHOGRAPHIC 3/4 FRONT-RIGHT view (camera slightly above, showing the front and the right side together)",
    "bl": "an ORTHOGRAPHIC 3/4 BACK-LEFT view (camera slightly above, showing the back and the left side together)",
    "br": "an ORTHOGRAPHIC 3/4 BACK-RIGHT view (camera slightly above, showing the back and the right side together)",
}
# Per-view visibility rules. Any extra elements in the model (ground/base, props, foliage, fixtures or
# other nearby objects) sit in particular places, so each view must show only what that camera can
# actually see, in its REAL position — keep elements where they belong, never relocate them or drag
# them into views that physically cannot see them (e.g. the back or the underside).
_VIEW_RULES = {
    "front": (
        "Show the FRONT of the model head-on: the object's front face together with any surrounding "
        "elements that sit in front of it, all kept in their exact original positions and scale. If the "
        "model is a standalone object with nothing around it, simply show its front."
    ),
    "back": (
        "Show the REAR of the model head-on: the back of the object. Anything that sits IN FRONT of the "
        "object is NOT visible from behind — do NOT carry front-only elements into this view; show only "
        "the back (plus anything that genuinely belongs at the back). The area behind is empty/plain."
    ),
    "left": (
        "Show the object's LEFT side head-on as the main subject — the side on the LEFT edge of the "
        "front view, turned to face the camera by rotating the model 90 degrees COUNTER-CLOCKWISE (seen "
        "from above). The front must NOT face the camera here (it has rotated away to the right). The "
        "reference images after the seed are the already-approved FRONT and TOP views: read the TOP "
        "view as the exact overhead MAP of where every element sits, and the FRONT view for how the "
        "elements look. Preserve EVERY element of the model that this side can see, each in the "
        "RELATIVE POSITION the top view shows (front-side elements toward the front edge of the "
        "frame), with correct OCCLUSION — nearer elements partially hide the ones behind them, and "
        "anything on the far (right) side is hidden by the object. Do NOT relocate, drop, duplicate or "
        "invent elements; only the camera changes."
    ),
    "right": (
        "Show the object's RIGHT side head-on as the main subject — the side on the RIGHT edge of the "
        "front view, turned to face the camera by rotating the model 90 degrees CLOCKWISE (seen from "
        "above). The front must NOT face the camera here (it has rotated away to the left). The "
        "reference images after the seed are the already-approved FRONT and TOP views: read the TOP "
        "view as the exact overhead MAP of where every element sits, and the FRONT view for how the "
        "elements look. Preserve EVERY element of the model that this side can see, each in the "
        "RELATIVE POSITION the top view shows (front-side elements toward the front edge of the "
        "frame), with correct OCCLUSION — nearer elements partially hide the ones behind them, and "
        "anything on the far (left) side is hidden by the object. Do NOT relocate, drop, duplicate or "
        "invent elements; only the camera changes."
    ),
    "top": (
        "Strict orthographic TOP-DOWN (bird's-eye) view looking straight DOWN from directly overhead. "
        "Show ONLY upper surfaces seen from above — the tops of the object and of any surrounding "
        "elements, plus the ground/base from above. Do NOT show any vertical face, side, front or wall "
        "(you are above them); it must read as a flat plan with NO perspective tilt."
    ),
    "bottom": (
        "Strict orthographic BOTTOM-UP view looking straight UP at the underside. Show ONLY the flat "
        "underside / base of the model — no top surfaces, no faces or sides, no surrounding detail."
    ),
    "fl": (
        "3/4 view from the FRONT-LEFT corner (camera slightly above, off the front-left): show the "
        "FRONT face and the LEFT side together, plus the top. Use the approved references — the FRONT "
        "view for the front, the LEFT view for the side, and the TOP view as the overhead map of which "
        "side is left vs right. The visible side MUST be the LEFT side, matching the approved LEFT view "
        "exactly — it is NOT the right side. Match the TOP view's handedness and do NOT mirror the "
        "model. Do NOT show the back or the right side."
    ),
    "fr": (
        "3/4 view from the FRONT-RIGHT corner (camera slightly above, off the front-right): show the "
        "FRONT face and the RIGHT side together, plus the top. Use the approved FRONT and RIGHT views, "
        "and the TOP view as the overhead map of which side is left vs right. The visible side MUST be "
        "the RIGHT side, matching the approved RIGHT view exactly — it is NOT the left side. Match the "
        "TOP view's handedness and do NOT mirror the model. Do NOT show the back or the left side."
    ),
    "bl": (
        "3/4 view from the BACK-LEFT corner (camera slightly above, off the back-left): show the BACK "
        "and the LEFT side together, plus the top. The TOP view is the SOLE authority for which side is "
        "left vs right — do NOT infer handedness from the BACK view, which itself shows left and right "
        "FLIPPED. The visible side MUST be the object's OWN LEFT side: identify it by matching the "
        "approved LEFT view exactly (same wall, same features), NOT by camera-relative left/right. Use "
        "the BACK view for the back face. Match the TOP view's handedness and do NOT mirror the model. "
        "Front-only elements are NOT visible from behind — do NOT include them. Do NOT show the front "
        "or the right side."
    ),
    "br": (
        "3/4 view from the BACK-RIGHT corner (camera slightly above, off the back-right): show the BACK "
        "and the RIGHT side together, plus the top. The TOP view is the SOLE authority for which side "
        "is left vs right — do NOT infer handedness from the BACK view, which itself shows left and "
        "right FLIPPED. The visible side MUST be the object's OWN RIGHT side: identify it by matching "
        "the approved RIGHT view exactly (same wall, same features), NOT by camera-relative left/right. "
        "Use the BACK view for the back face. Match the TOP view's handedness and do NOT mirror the "
        "model. Front-only elements are NOT visible from behind — do NOT include them. Do NOT show the "
        "front or the left side."
    ),
}

# Cardinal (single-face) views: each shows ONLY its one face — never another canonical view or a 3/4
# angle. (The 3/4 corners deliberately show two sides + top, so they are excluded from this rule.)
CARDINAL_FACE = {
    "front": "front face", "back": "back face", "left": "left side", "right": "right side",
    "top": "top (the overhead plan)", "bottom": "underside",
}


def build_prompt(view, n_context, edit_prompt=None):
    """Per-view prompt: orthographic head-on framing + per-view VISIBILITY rules so any extra elements
    in the model (ground, props, foliage, fixtures, nearby objects) stay in their REAL positions and
    appear ONLY in the views that can actually see them — kept where they belong, not relocated and not
    stripped. The four side views use a ground-aligned, untilted camera. Cartoonish + consistency."""
    tag = VIEW_TO_TAG[view]
    consistency = (
        "The image(s) AFTER Image 1 are already-approved clean reference views of this SAME model "
        "(e.g. the approved FRONT) — match their colours, materials, finish, proportions and art style "
        "EXACTLY. Treat Image 1 (the original seed) as the AUTHORITY for the model's content and layout "
        "— which elements exist and WHERE each one sits — so the elements that ARE visible in this view "
        "keep their correct identity, scale and position; do not invent, relocate or duplicate anything. "
        if n_context > 1 else ""
    )
    tweak = f"User adjustment (apply precisely, keep the same view): {edit_prompt.strip()}. " \
        if (edit_prompt and edit_prompt.strip()) else ""
    exclusive = (
        f"Render exactly ONE canonical orthographic view: show ONLY the {CARDINAL_FACE[tag]} and "
        "nothing of the other five canonical views. Do NOT include, reveal or hint at any other "
        "face/side of the model, do NOT show more than one side at once, and use NO 3/4, angled or "
        "perspective framing — strictly this single head-on view. "
        if tag in CARDINAL_FACE else ""
    )
    ground = (
        "The camera is at ground level and perfectly HORIZONTAL — aligned with the ground, with NO "
        "upward or downward tilt — so the floor/ground is seen edge-on as only a thin strip along the "
        "bottom (where things meet the ground), NOT as a wide visible surface; the full floor is only "
        "visible in the top view. "
        if tag in ("front", "back", "left", "right") else ""
    )
    return (
        "Image 1 is the seed reference image of a single stylised 3D model — a standalone object or a "
        "small scene/diorama (an object together with whatever surrounds it, such as ground/base, props, "
        "foliage, fixtures or other nearby objects, if any). " + consistency +
        f"Re-render it as {_VIEW_FRAME[tag]} ({_ROTATION[view]}); STRICTLY orthographic, no perspective "
        "foreshortening and no tilt beyond the described camera. " + ground + _VIEW_RULES[tag] + " " +
        exclusive +
        "Keep every shown object's EXACT colours, materials, proportions, identity and the direction it "
        "faces, as in the seed; do NOT relocate, duplicate, add, remove or restyle anything beyond what "
        "this exact view requires — only the camera viewpoint changes. A plain neutral backdrop may sit "
        "behind the model, but never change the model's own layout. " + tweak +
        CARTOON_STYLE + " " + CONSISTENCY_RULE
    )


def generate_view(view, seed_path, dep_paths, edit_prompt=None, size=(1024, 1024)):
    """Generate one reference view from the seed + its approved dependency views. Returns PIL RGB.

    `dep_paths` is the ordered list of file paths for VIEW_INPUTS[view] (already approved). Raises
    if the seed is missing for a view that needs it (front needs only the seed)."""
    if not (seed_path and os.path.exists(seed_path)):
        raise RuntimeError("reference generation needs a seed image")
    context = [Image.open(seed_path).convert("RGB")]
    for p in dep_paths:
        if p and os.path.exists(p):
            context.append(Image.open(p).convert("RGB"))
    return edit_image(context, build_prompt(view, len(context), edit_prompt), size=size).convert("RGB")


# ── Masked (brush) editing of an existing reference view ─────────────────────────────────────────
def build_edit_prompt(edit_prompt):
    """Prompt for a MASKED inpaint of an existing reference view: only the masked (transparent) region
    changes; everything else stays pixel-identical."""
    change = (edit_prompt or "").strip() or "refine and clean up this region, keeping it consistent"
    return (
        "Image 1 is an existing reference view of a stylised 3D model. A mask is provided: edit ONLY the "
        "masked (transparent) region and keep EVERYTHING outside the mask pixel-identical — do not alter "
        "any colour, shape, framing, lighting or pixel outside the masked area. Inside the masked region "
        f"apply this change: {change}. Blend the edited area seamlessly with its surroundings, matching "
        "the existing colours, materials and style. " + CARTOON_STYLE + " " + CONSISTENCY_RULE
    )


def _openai_mask_from_brush(mask_path, size):
    """Convert the frontend brush mask (painted region = opaque) into an OpenAI edit mask
    (alpha 0 = the region gpt may repaint, alpha 255 = keep)."""
    import numpy as np
    fm = Image.open(mask_path).convert("RGBA").resize(size)
    painted = np.asarray(fm.split()[-1]) > 10
    alpha = np.where(painted, 0, 255).astype("uint8")
    out = Image.new("RGBA", size, (0, 0, 0, 255))
    out.putalpha(Image.fromarray(alpha))
    return out


def edit_view_masked(current_path, mask_path, edit_prompt, size=(1024, 1024)):
    """Inpaint ONLY the brushed region of an existing reference view via gpt-image (OpenAI mask).
    Gemini has no mask API, so this requires OPENAI_API_KEY. Returns PIL RGB."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Masked editing needs OPENAI_API_KEY (gpt-image mask); Gemini has no mask support.")
    from webapp.image_edit import _openai_edit
    img = Image.open(current_path).convert("RGB").resize(size)
    mask = _openai_mask_from_brush(mask_path, size)
    return _openai_edit([img], build_edit_prompt(edit_prompt), size, mask=mask).convert("RGB")
