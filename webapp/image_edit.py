"""Image edit/refine with a provider fallback chain.

Order: gpt-image-2 (OpenAI) first, then Google's "nano banana" (Gemini 2.5 Flash
Image) if OpenAI is unavailable or fails (e.g. billing hard limit). Shared by the
server (gptproject) and the MV-Adapter runner (mvgpt refine) — both do
image(s)+prompt -> image editing.

`images[0]` is the image to edit/refine (its structure/viewpoint should be preserved);
`images[1:]` are reference images (style/material). Returns a PIL.Image (RGB) at `size`.
Set OPENAI_API_KEY and/or GEMINI_API_KEY. Raises if all available providers fail.
"""
import base64
import io
import os

# Shared style directive: keep textures flat/cartoonish, not photoreal. Appended to the
# gptproject + mvgpt refine prompts (both providers otherwise drift toward realism).
CARTOON_STYLE = (
    "ART STYLE (mandatory): stylised, cartoonish, hand-painted low-poly, 3D game asset — flat, "
    "clean, slightly saturated colours, soft simple shading, crisp readable shapes. NOT "
    "photorealistic: no realistic reflections or specular highlights, no PBR or photographic "
    "surface detail, no realistic grime, dirt or weathering, no dramatic lighting. "
    "When borrowing style from the reference image(s), keep this flat 3D cartoon "
    "look. Think mobile/stylised game prop, not a photo."
)

# Per-object consistency: only the camera changes between views, never an object's own
# colour or the direction it faces. Appended to both refine/paint prompts.
CONSISTENCY_RULE = (
    "Object consistency: every object keeps the exact colour, materials, markings and the "
    "real-world direction it faces, as established by Image 1 and the reference image(s). Only "
    "the camera viewpoint differs between views — never rotate, mirror, recolour or reorient an "
    "object itself. Example: a white car facing east stays white and still faces east when the "
    "camera looks from the north (you simply see it from a different side). Do not flip handedness "
    "or swap which side faces which way."
)

# Hand-paint "AI fix": Image 1 is a render of ONE face of the model's CURRENT texture. Clean it up
# WITHOUT restyling — keep the exact palette, base colours, composition and framing, repair only the
# local artefacts a projected texture leaves behind. Pair with CONSISTENCY_RULE + CARTOON_STYLE and,
# optionally, the user's own touch-up note. Prefer Gemini so the layout/proportions stay locked.
HANDPAINT_FIX_PROMPT = (
    "Image 1 is a render of ONE face of a 3D model's CURRENT texture. Clean it up WITHOUT restyling "
    "it. Keep the exact same composition, framing, proportions, subject, palette and BASE COLOURS as "
    "Image 1 — do not recolour, relight, restyle or redraw anything. Only repair local texture "
    "inconsistencies: visible seams, projection smears, stretched or blurry patches, duplicated or "
    "ghosted detail, colour bleeding across edges, and small artefacts. Where a region is garbled, "
    "reconstruct it to match the colours and the style already present in Image 1. Any additional "
    "reference images show the intended clean look for this view — use them ONLY to resolve "
    "ambiguous or garbled areas, never to change Image 1's colours. Output the same view at the same "
    "scale."
)


# Extra rule for the FREE-CAMERA (custom) hand-paint AI fix, where the reference is the nearest
# canonical face — a DIFFERENT camera angle of the same object. Without this, Gemini tends to redraw
# the captured view toward the reference's viewpoint at angles far from any canonical face.
HANDPAINT_CUSTOM_REF_RULE = (
    "IMPORTANT — the reference image is a DIFFERENT camera angle of the SAME object, supplied ONLY as a "
    "colour and material guide. Image 1 is the ABSOLUTE source of truth for the camera angle, "
    "composition, framing, proportions, geometry, silhouette and which surfaces are visible. Do NOT "
    "adopt the reference's viewpoint, layout, framing or proportions; do NOT rotate, re-perspective, "
    "re-frame or otherwise redraw Image 1's shapes toward the reference; do NOT add, remove or move any "
    "object to match it. Reproduce Image 1's exact structure and layout pixel-for-pixel, changing only "
    "local colour/texture to repair garbled spots. If the reference and Image 1 disagree on shape, "
    "position, scale or angle, Image 1 ALWAYS wins."
)


def _guided_filter(guide, src, radius, eps):
    """He et al. guided filter for one channel (float [0,1]), built from box filters so it needs no
    opencv-contrib. Smooths `src` while snapping its transitions to `guide`'s edges."""
    import cv2

    r = (radius, radius)
    mean_g = cv2.boxFilter(guide, -1, r)
    mean_s = cv2.boxFilter(src, -1, r)
    cov_gs = cv2.boxFilter(guide * src, -1, r) - mean_g * mean_s
    var_g = cv2.boxFilter(guide * guide, -1, r) - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return cv2.boxFilter(a, -1, r) * guide + cv2.boxFilter(b, -1, r)


def recolor_preserve_structure(structure, color, chroma_radius=24, chroma_eps=1e-3):
    """Recolour `structure` with `color`'s hues while keeping ALL of `structure`'s geometry.

    A generative edit (Gemini / gpt-image) regenerates pixels and drifts — elements shift, redraw or
    reframe even when the prompt forbids it. This reconstructs the result in CIELAB so its LIGHTNESS L
    (and therefore every element's position, edges, shapes and fine detail) comes from `structure` (the
    original render), and only the colour channels a/b come from `color` (the AI output). The model can
    thus only change colours — it cannot move or redraw anything. The a/b channels are edge-aware
    aligned to `structure` with a guided filter (guide = structure L) so colour transitions snap back to
    the original's edges, suppressing bleed from the AI's spatial drift.
    """
    import cv2
    import numpy as np
    from PIL import Image

    s = np.asarray(structure.convert("RGB"))
    c = np.asarray(color.convert("RGB").resize(structure.size))
    s_lab = cv2.cvtColor(s, cv2.COLOR_RGB2LAB).astype(np.float32)
    c_lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB).astype(np.float32)

    guide = s_lab[..., 0] / 255.0  # structure lightness as the edge guide
    out = s_lab.copy()             # L stays from `structure` -> geometry is locked
    for ch in (1, 2):              # a, b colour from the AI output, realigned to the structure's edges
        out[..., ch] = np.clip(_guided_filter(guide, c_lab[..., ch] / 255.0, chroma_radius, chroma_eps) * 255.0, 0, 255)
    rgb = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def _png_buf(img, name, size, mode="RGB"):
    b = io.BytesIO()
    img.convert(mode).resize(size).save(b, format="PNG")
    b.seek(0)
    b.name = name
    return b


def _openai_edit(images, prompt, size, mask=None):
    from openai import OpenAI
    from PIL import Image

    client = OpenAI()
    model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")
    # 1-indexed filenames so they line up with prompts that call the base "Image 1" and refs
    # "Image 2 and onward". gpt-image binds mainly by array order, but matching the wording is a
    # cheap reinforcement of which image is the geometry authority.
    bufs = [_png_buf(im, f"image-{i + 1}.png", size) for i, im in enumerate(images)]
    kw = {}
    if mask is not None:
        # RGBA mask: transparent (alpha 0) = region gpt may paint; opaque = keep image[0].
        kw["mask"] = _png_buf(mask, "mask.png", size, mode="RGBA")
    res = client.images.edit(model=model, image=bufs, prompt=prompt, size=f"{size[0]}x{size[1]}", n=1, **kw)
    return Image.open(io.BytesIO(base64.b64decode(res.data[0].b64_json))).convert("RGB")


def _gemini_edit(images, prompt, size):
    """Edit via Gemini 2.5 Flash Image ("nano banana")."""
    from google import genai
    from PIL import Image

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # Nano Banana Pro (latest, best quality). Override with GEMINI_IMAGE_MODEL, e.g.
    # gemini-3.1-flash-image (faster/cheaper) or gemini-2.5-flash-image (original).
    model = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image")
    # Interleave "Image N:" labels so the prompt's "Image 1" / "Image 2" references bind to the
    # actual images. Without this the model only sees raw images in order and guesses which is the
    # authority — at off-canonical angles it grabs the cleaner reference and drifts off Image 1.
    contents = [prompt]
    for i, im in enumerate(images, 1):
        contents.append(f"Image {i}:")
        contents.append(im.convert("RGB").resize(size))
    resp = client.models.generate_content(model=model, contents=contents)
    for cand in (getattr(resp, "candidates", None) or []):
        for part in (getattr(getattr(cand, "content", None), "parts", None) or []):
            data = getattr(getattr(part, "inline_data", None), "data", None)
            if data:
                if isinstance(data, str):
                    data = base64.b64decode(data)
                return Image.open(io.BytesIO(data)).convert("RGB").resize(size)
    raise RuntimeError("gemini returned no image part")


def edit_image(images, prompt, size=(1024, 1024), mask=None, prefer="openai"):
    """Refine images[0] (style from images[1:]) with provider fallback. Returns PIL RGB.
    `mask` (PIL RGBA): transparent areas are the only region gpt-image may paint (OpenAI edit
    mask); used to confine output to the object's silhouette. Gemini has no mask API.
    `prefer`: which provider to try FIRST ("openai" or "gemini"). For COLOURISING a geometry
    render, prefer="gemini" — Gemini keeps the input's exact proportions/layout, whereas gpt-image
    drifts (enlarges/reframes), which mis-scales the projected texture."""
    if isinstance(size, int):
        size = (size, size)
    errors = []
    order = ["gemini", "openai"] if prefer == "gemini" else ["openai", "gemini"]
    for prov in order:
        if prov == "openai" and os.environ.get("OPENAI_API_KEY"):
            try:
                return _openai_edit(images, prompt, size, mask=mask)
            except Exception as e:  # noqa: BLE001
                errors.append(f"gpt-image-2: {e}")
        if prov == "gemini" and os.environ.get("GEMINI_API_KEY"):
            try:
                return _gemini_edit(images, prompt, size)
            except Exception as e:  # noqa: BLE001
                errors.append(f"gemini-nano-banana: {e}")
    if errors:
        raise RuntimeError("image edit failed (" + " | ".join(errors) + ")")
    raise RuntimeError("no image API key set (OPENAI_API_KEY or GEMINI_API_KEY)")
