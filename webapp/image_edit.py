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
    bufs = [_png_buf(im, f"img{i}.png", size) for i, im in enumerate(images)]
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
    contents = [prompt] + [im.convert("RGB").resize(size) for im in images]
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
