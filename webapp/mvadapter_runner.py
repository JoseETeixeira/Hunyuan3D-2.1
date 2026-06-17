"""In-repo MV-Adapter SDXL image-to-texture runner.

Runs inside the isolated 'mvadapter' conda env (invoked by mvadapter_texture.py via
`conda run`). It reuses MV-Adapter's own prepare_pipeline / run_pipeline /
TexturePipeline but adds model cpu-offload + tunable resolution so the SDXL variant
fits a 16GB GPU — which the upstream scripts/texture_i2tex.py CLI does not expose.
SD2.1 is gated on HuggingFace, so SDXL (open weights) is the default base.

Imports MV-Adapter from MVADAPTER_DIR (added to sys.path). Prints 'SHADED_GLB: <path>'.
"""
import argparse
import os
import sys


# MV-Adapter canonical view order (camera_azimuth_deg=[-90,0,90,180,90,90], elev=[0,0,0,0,+90,-90]).
_VIEW_LABELS = [
    "the LEFT side (profile)",
    "the FRONT",
    "the RIGHT side (profile)",
    "the BACK",
    "the TOP, looking straight down",
    "the BOTTOM, looking straight up",
]

_AZ_NAMES = {
    0: "the FRONT", 45: "the FRONT-RIGHT 3/4 view", 90: "the RIGHT side",
    135: "the BACK-RIGHT 3/4 view", 180: "the BACK", 225: "the BACK-LEFT 3/4 view",
    270: "the LEFT side", 315: "the FRONT-LEFT 3/4 view",
}


def _angle_label(az, el):
    """Human label for an arbitrary camera angle (used for combined/3-4 view sets)."""
    if el >= 60:
        return "the TOP, looking straight down"
    if el <= -60:
        return "the BOTTOM, looking straight up"
    a = az % 360
    key = min(_AZ_NAMES, key=lambda k: min(abs(a - k), 360 - abs(a - k)))
    return _AZ_NAMES[key]


def _view_sides(az, el):
    """Side tag(s) a given camera view depicts, in priority order. Cardinal views map to
    one side; 3/4 corners map to their own corner tag (fl/fr/bl/br) FIRST, then their two
    adjacent cardinals as fallback; tilted views still resolve by azimuth. Used to pick which
    side-tagged reference(s) feed each view's refine."""
    if el is not None and el >= 60:
        return ["top"]
    if el is not None and el <= -60:
        return ["bottom"]
    cards = {0: "front", 90: "right", 180: "back", 270: "left"}
    corners = {45: "fr", 135: "br", 225: "bl", 315: "fl"}  # MV-Adapter azimuth convention
    a = az % 360
    order = sorted(cards, key=lambda k: min(abs(a - k), 360 - abs(a - k)))
    d0 = min(abs(a - order[0]), 360 - abs(a - order[0]))
    if 25 <= d0 <= 65:  # 3/4 corner: own tag first, then both adjacent cardinals as fallback
        ckey = min(corners, key=lambda k: min(abs(a - k), 360 - abs(a - k)))
        return [corners[ckey], cards[order[0]], cards[order[1]]]
    return [cards[order[0]]]


def _gpt_refine_views(views, ref_paths, save_dir, save_name, azimuths=None, elevations=None, ref_sides=None,
                      transfer=False, recolor=False):
    """Refine each MV-Adapter view with gpt-image-2 in PARALLEL, steered by reference
    image(s). Each refine must REPRODUCE its input view's exact viewpoint (the bake
    relies on it) — references supply appearance/identity, never pose. When references are
    side-tagged (ref_sides parallel to ref_paths, e.g. front/back/left/right/top/bottom),
    each view is fed ONLY the reference(s) matching its own side (corners get both adjacent
    sides); untagged ("any") references go to every view. gpt-image-2's min size is 1024,
    so views are processed at 1024. Falls back to the original view on any per-view failure.
    """
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image as _Image

    # image_edit lives next to this runner; add its dir so we can import it standalone.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from image_edit import CARTOON_STYLE, CONSISTENCY_RULE, edit_image  # gpt-image-2 -> Gemini fallback

    workers = int(os.environ.get("MVADAPTER_REFINE_WORKERS", "6"))
    # Optional: feed the MV draft to the recolour as GREYSCALE so gpt treats it purely as a
    # structural guide and takes ALL colour from the elevation (a colourise task preserves
    # geometry better than a recolour, and avoids gpt merging same-coloured cars into the
    # building). Enable with MVGPT_RECOLOR_GRAYSCALE=1.
    recolor_bw = recolor and os.environ.get("MVGPT_RECOLOR_GRAYSCALE", "0").lower() not in ("0", "false", "no")
    refs = [_Image.open(p).convert("RGB") for p in (ref_paths or [])]
    sides = [(s or "any").strip().lower() for s in (ref_sides or [])]
    # references explicitly tagged to a side, grouped; "any"/untagged feed every view.
    tagged = {}
    any_refs = []
    for idx, img in enumerate(refs):
        s = sides[idx] if idx < len(sides) else "any"
        (any_refs if s in ("", "any") else tagged.setdefault(s, [])).append(img)
    has_tags = bool(tagged)

    _CORNER_TAGS = ("fl", "fr", "bl", "br")

    def _refs_for(i):
        if not has_tags:
            return refs  # legacy: no side tags -> feed all references to every view
        want = _view_sides(azimuths[i], elevations[i]) if (azimuths and i < len(azimuths)) else ["front"]
        matched = []
        for w in want:
            got = tagged.get(w, [])
            # an explicit corner-tagged reference (want[0] for a 3/4 view) wins outright, so the
            # corner view isn't diluted by both adjacent cardinal references.
            if got and w in _CORNER_TAGS:
                return got
            matched += got
        # "Only feed that": when this view HAS a matching-side reference, feed ONLY those
        # (avoids blending heterogeneous side views into a generic, element-losing result).
        if matched:
            return matched
        # No side-specific match: general ("any") refs, else front, else all.
        return any_refs or tagged.get("front", []) or refs

    def _transfer_prompt(i):
        """Strict elevation transfer: shape ONLY from the MV draft, appearance fully from the
        matching-side elevation(s); faces the elevation doesn't show keep the draft's look."""
        label = _angle_label(azimuths[i], elevations[i]) if azimuths and i < len(azimuths) else (
            _VIEW_LABELS[i] if i < len(_VIEW_LABELS) else "this exact view")
        return (
            f"Image 1 is a rough 3D draft render: {label} of a single object. Use Image 1 for ONE "
            "thing only: its exact silhouette/outline, camera viewpoint, orientation, proportions, "
            "scale and crop. IGNORE and DISCARD all of Image 1's colours, materials, lighting and "
            "surface detail — they are wrong and must not appear in the output. "
            ""
            "Images 2 and onward are the AUTHORITATIVE appearance: clean head-on elevation(s) of the "
            "SAME object for this side. Reproduce their colours, materials, panels, windows, doors, "
            "signage/text, trim and every element EXACTLY, mapped onto Image 1's silhouette for this "
            "viewpoint. The finished view must look like the elevation(s) wrapped onto Image 1's shape "
            "— not like Image 1. Do not genericize, simplify or restyle. "
            ""
            "PROPS / VEHICLES (critical): the elevation(s) may include discrete objects — vehicles/cars, "
            "foliage, bushes, bollards, lights, signage. Reproduce EVERY such object that the elevation "
            "shows, at the SAME position, colour, count and facing direction. Do NOT drop or omit an "
            "object the elevation shows (e.g. a car under the canopy must stay). Do NOT add objects the "
            "elevation does not show, do NOT duplicate them, do NOT move them to a different spot, and do "
            "NOT flip the direction a vehicle faces. A blue car parked on the right stays one blue car on "
            "the right facing the same way. Take vehicles/props ONLY from the elevation — never from "
            "Image 1 (Image 1's cars/props are wrong: wrong colour, mixed, mislocated). "
            ""
            "Coverage rule (narrow): you may fall back to Image 1 ONLY for a structural wall/surface "
            "region that the elevation genuinely does not depict (a perpendicular face seen edge-on at a "
            "3/4 angle). NEVER use Image 1 for any object, vehicle, prop, colour or material that the "
            "elevation does show. When two elevations are given (a 3/4 corner), keep each side's content "
            "on its own side, do not merge or duplicate objects across the two sides, and invent nothing. "
            ""
            "Hard constraints: keep Image 1's outline, viewpoint, proportions, scale and crop unchanged; "
            "single object on a plain neutral background; even flat lighting. " + CARTOON_STYLE + " " +
            CONSISTENCY_RULE
        )

    def _recolor_prompt(i):
        """Hybrid: keep the MV draft's LAYOUT/positions exactly (MV-Adapter views are 3D-
        consistent, so car positions agree across views); only correct colours/materials
        toward the matching-side elevation. A recolour pass, not a redraw."""
        label = _angle_label(azimuths[i], elevations[i]) if azimuths and i < len(azimuths) else (
            _VIEW_LABELS[i] if i < len(_VIEW_LABELS) else "this exact view")
        img1 = ("a GREYSCALE structural render" if recolor_bw else "a render")
        task = ("COLOURISE it" if recolor_bw else "Recolour it")
        return (
            f"Image 1 is {img1} ({label}) of a single object. It is the ONLY source of LAYOUT and "
            "SHAPE: keep its exact composition, viewpoint, silhouette, proportions, depth, and the "
            "position, count, size and facing of EVERY element — walls, panels, windows, doors, "
            f"signage, foliage and especially vehicles/cars. {task} without moving, adding, removing, "
            "duplicating, resizing, reshaping, reorienting or restaging anything. This is a COLOUR/"
            "MATERIAL pass, NOT a redraw. "
            ""
            "Images 2 and onward are clean reference elevation(s) of the SAME object for this side. Use "
            "them ONLY as a colour/material guide: correct Image 1's colours, materials, palette and "
            "finish to match the references — fix muddy, washed-out, blended or wrong colours and apply "
            "the references' clean flat-cartoon colour/material blocking. Every object stays exactly "
            "where Image 1 has it: a car in Image 1 remains the same car in the same position and "
            "facing, only its colour/material may be cleaned to match the reference palette. "
            ""
            "Output: Image 1 unchanged in layout, composition and shape, with corrected, reference-"
            "matched colours and materials. " + CARTOON_STYLE + " " + CONSISTENCY_RULE
        )

    def _prompt(i):
       if recolor:
           return _recolor_prompt(i)
       if transfer:
           return _transfer_prompt(i)
       label = _angle_label(azimuths[i], elevations[i]) if azimuths and i < len(azimuths) else (
           _VIEW_LABELS[i] if i < len(_VIEW_LABELS) else "this exact view")
       return (
            f"Image 1 is a rough draft render: {label} of a single 3D object. "
            "Use Image 1 ONLY for geometry: its silhouette, viewpoint, camera angle, perspective, "
            "orientation, pose, proportions, depth, surface layout, framing, scale and crop. Match all "
            "of those exactly — do not rotate, reframe, resize, move, reshape, restage, or reinterpret "
            "the form. "
            ""
            "Image 1's COLOURS, materials and small details are UNRELIABLE (the draft often hallucinates "
            "shapes/colours) — do NOT copy them. Take the object's true appearance from the reference "
            "image(s) and render it onto Image 1's geometry for this exact view. "
            ""
            "Images 2 and onward are reference views of the SAME desired object from different angles. "
            "They are the ground truth for WHAT the object is: its real colours, materials, panels, "
            "parts, signage/markings, windows, trim and overall design — reproduce these faithfully. "
            "HOWEVER their element POSITIONS may be slightly wrong and they are shot from different "
            "cameras, so do NOT copy a reference's camera angle, framing, crop or exact element "
            "placement. Instead: use the reference(s) closest to this view for appearance, and PLACE "
            "every element where Image 1's geometry says it belongs for this view (Image 1 decides "
            "position/scale; the references decide identity/colour). "
            ""
            "Reproduce the references' DISTINCTIVE specific elements exactly — the central panel, "
            "signage/text, doors, the exact window pattern, awnings, trim and colour blocking. Do NOT "
            "genericize, simplify, average or omit them, and do NOT invent elements the references do "
            "not show. If a reference shows a dark central entrance panel, this view must have that same "
            "panel (placed per Image 1's geometry), not a generic wall. "
            ""
            "Priority order: 1) match Image 1's geometry, silhouette, viewpoint, scale and where each "
            "part sits, exactly; 2) give the object the true colours, materials and elements shown in "
            "the references; 3) correct any draft colours/parts to match the references; 4) clean flat "
            "cartoon finish. Geometry/placement/scale always come from Image 1; identity/colour always "
            "comes from the references. "
            ""
            "Forbidden changes: no different outline, pose, viewpoint, proportions, camera or scale vs "
            "Image 1; no background scenery; no shadows or props that change the composition. (You MUST "
            "change Image 1's wrong colours/details to match the references — that is the goal.) "
            ""
            "Output a clean single-object render on a plain background with even lighting, matching the "
            "viewpoint, framing and scale of Image 1. " + CARTOON_STYLE + " " + CONSISTENCY_RULE
        )

    errors = []
    attempted = []

    elev_mode = transfer or recolor  # both use elevations as side references
    def _refine_one(i, v):
        el = elevations[i] if (elevations and i < len(elevations)) else 0.0
        # TOP view in elevation mode: MV-Adapter generates a poor near-vertical draft (often a
        # front-ish view, not a roof plan), so refining/recolouring onto it is unreliable. Use the
        # clean top elevation directly — the bake aligns it to the roof/lot silhouette.
        if elev_mode and el >= 60:
            refs_i = _refs_for(i)
            out = (refs_i[0] if refs_i else v).convert("RGB").resize((1024, 1024))
            out.save(os.path.join(save_dir, f"{save_name}_gptview{i}.png"))
            return out
        # 3/4 CORNER views (a diagonal azimuth that maps to TWO cardinal sides) in elevation
        # mode: gpt reliably hallucinates a fresh 3/4 building here regardless of how strict the
        # prompt is, and that wrong content bakes onto the adjacent faces. Keep the 3D-consistent
        # raw MV view instead — it's down-weighted at bake time, so it only fills occlusion gaps.
        az_i = azimuths[i] if (azimuths and i < len(azimuths)) else 0.0
        if elev_mode and abs(el) < 60 and len(_view_sides(az_i, el)) > 1:
            out = v.convert("RGB").resize((1024, 1024))
            out.save(os.path.join(save_dir, f"{save_name}_gptview{i}.png"))
            return out
        # Which near-vertical views to keep as raw MV (skip gpt). In elevation mode only the
        # bottom is kept raw (the 3/4 source carries no underside info). In the legacy refine
        # BOTH top and bottom are kept raw (gpt paints a facade on the roof / blank on the base).
        skip = (el <= -60) if elev_mode else (abs(el) >= 60)
        if skip:
            out = v.convert("RGB").resize((1024, 1024))
            out.save(os.path.join(save_dir, f"{save_name}_gptview{i}.png"))
            return out
        attempted.append(i)
        try:
            inp = v.convert("L").convert("RGB") if recolor_bw else v  # greyscale -> colourise
            refined = edit_image([inp] + _refs_for(i), _prompt(i), size=(1024, 1024))
            refined.save(os.path.join(save_dir, f"{save_name}_gptview{i}.png"))
            return refined
        except Exception as e:  # noqa: BLE001
            print(f"[mvrunner] refine failed for view {i}: {e}; using original")
            errors.append(str(e))
            return v.convert("RGB").resize((1024, 1024))

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(views)))) as ex:
        out = list(ex.map(lambda iv: _refine_one(*iv), list(enumerate(views))))
    n_try = len(attempted)
    ok = n_try - len(errors)
    print(f"GPT_REFINE_SUMMARY ok={ok}/{n_try} refined ({len(views) - n_try} top/bottom kept raw)")
    # If every ATTEMPTED refine failed, the equator/corner views are un-refined MV output
    # — surface that as a hard error (e.g. OpenAI billing limit) instead of pretending.
    if n_try and ok == 0:
        raise RuntimeError(f"gpt refine failed for all {n_try} views: {errors[0] if errors else 'unknown'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--save_name", required=True)
    ap.add_argument("--text", default="high quality")
    ap.add_argument("--remove_bg", action="store_true")
    ap.add_argument("--offload", action="store_true", help="model cpu-offload (more RAM, less VRAM)")
    ap.add_argument("--upscale", action="store_true", help="RealESRGAN x2 view upscale (RAM-heavy)")
    ap.add_argument("--gpt_refine", action="store_true", help="refine each MV view with gpt-image-2")
    ap.add_argument("--ref", action="append", default=[], help="reference image path(s) for gpt-refine")
    ap.add_argument("--ref_side", action="append", default=[],
                    help="side tag per --ref (front/back/left/right/top/bottom/any), parallel order")
    ap.add_argument("--elev_transfer", action="store_true",
                    help="strict elevation transfer: shape from MV draft, appearance from side elevation")
    ap.add_argument("--recolor", action="store_true",
                    help="hybrid recolour: keep MV draft layout/positions, only recolour toward elevation")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--uv_size", type=int, default=2048)
    ap.add_argument("--num_views", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--azimuths", default="", help="comma-sep azimuth_deg (len=6*k); default canonical")
    ap.add_argument("--elevations", default="", help="comma-sep elevation_deg (len=6*k); default canonical")
    ap.add_argument("--bake", default="full", choices=["full", "none"],
                    help="'none' = generate views only (server bakes with Hunyuan, e.g. combined >6 views)")
    args = ap.parse_args()

    mv_dir = os.environ.get("MVADAPTER_DIR", "/opt/mvadapter/MV-Adapter")
    sys.path.insert(0, mv_dir)
    os.chdir(mv_dir)  # so the TexturePipeline's ./checkpoints/* resolve

    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    from diffusers import AutoencoderKL
    from scripts.inference_ig2mv_sdxl import run_pipeline, remove_bg
    from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
    from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
    from mvadapter.utils import make_image_grid

    device, dtype, nv = "cuda", torch.float16, args.num_views

    # Configurable view set. run_pipeline hardcodes its cameras, so we monkeypatch
    # get_orthogonal_camera to inject our angles (3/4 corners, tilted, etc.). The bake
    # below uses the same angles. Lists must be length num_views (the adapter is 6-view).
    azimuths = [float(x) for x in args.azimuths.split(",") if x.strip()] or [-90, 0, 90, 180, 90, 90]
    elevations = [float(x) for x in args.elevations.split(",") if x.strip()] or [0, 0, 0, 0, 89.99, -89.99]
    import scripts.inference_ig2mv_sdxl as _ig

    _orig_cam = _ig.get_orthogonal_camera
    _cur = {"az": azimuths[:nv], "el": elevations[:nv]}  # angles for the current pass

    def _cam(**kw):
        kw["elevation_deg"] = _cur["el"]
        kw["azimuth_deg"] = _cur["az"]
        return _orig_cam(**kw)

    _ig.get_orthogonal_camera = _cam

    # Memory-efficient load: pull the fp16 variant directly so we don't spike host RAM
    # with a transient fp32 copy. The upstream prepare_pipeline omits torch_dtype/variant
    # and loads fp32 (~26GB peak) which OOM-kills a 32GB box. Fall back if no fp16 variant.
    try:
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16", use_safetensors=True, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[mvrunner] fp16 variant unavailable ({e}); loading default precision")
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype, low_cpu_mem_usage=True,
        )
    pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=None
    )
    pipe.init_custom_adapter(num_views=nv, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
    pipe.load_custom_adapter("huanngzh/mv-adapter", weight_name="mvadapter_ig2mv_sdxl.safetensors")
    pipe.cond_encoder.to(device=device, dtype=dtype)
    if args.offload:
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:  # noqa: BLE001
            print(f"[mvrunner] offload unavailable ({e}); using full GPU")
            pipe.to(device=device, dtype=dtype)
    else:
        pipe.to(device=device, dtype=dtype)
    pipe.enable_vae_slicing()

    remove_bg_fn = None
    if args.remove_bg:
        # Force fp32: BiRefNet loads fp16 weights but remove_bg feeds an fp32 tensor
        # (ToTensor/Normalize), which mismatches its conv bias ("Input float, bias Half").
        biref = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        ).to(device).float().eval()
        tf = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        remove_bg_fn = lambda x: remove_bg(x, biref, tf, device)

    os.makedirs(args.save_dir, exist_ok=True)
    # The reference may be RGBA (uploads are saved RGBA); the bg-removal Normalize wants
    # 3 channels. Feed an RGB copy.
    from PIL import Image as _PILImage

    ref_rgb = os.path.join(args.save_dir, f"{args.save_name}_ref_rgb.png")
    _PILImage.open(args.image).convert("RGB").save(ref_rgb)

    # Generate in passes of num_views (the adapter is fixed at 6 views). A combined view
    # set (>6 angles) runs several passes with different cameras; the bake below merges
    # ALL views by per-texel visibility, so faces occluded in one pass are covered by another.
    nsets = max(1, len(azimuths) // nv)
    images = []
    for s in range(nsets):
        _cur["az"] = azimuths[s * nv:(s + 1) * nv]
        _cur["el"] = elevations[s * nv:(s + 1) * nv]
        print(f"[mvrunner] MV pass {s + 1}/{nsets} azim={_cur['az']}")
        imgs, *_ = run_pipeline(
            pipe,
            mesh_path=args.mesh,
            num_views=nv,
            text=args.text,
            image=ref_rgb,
            height=args.height,
            width=args.height,
            num_inference_steps=args.steps,
            guidance_scale=3.0,
            seed=-1,
            reference_conditioning_scale=1.0,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
            device=device,
            remove_bg_fn=remove_bg_fn,
        )
        images.extend(imgs)
    # Save the raw MV-Adapter views (pre-refine) so geometry/refine drift is diagnosable.
    for i, im in enumerate(images):
        im.convert("RGB").save(os.path.join(args.save_dir, f"{args.save_name}_rawview{i}.png"))
    if args.gpt_refine:
        images = _gpt_refine_views(images, args.ref, args.save_dir, args.save_name, azimuths, elevations,
                                   ref_sides=args.ref_side, transfer=args.elev_transfer, recolor=args.recolor)
    mv_path = os.path.join(args.save_dir, f"{args.save_name}.png")
    make_image_grid(images, rows=1).save(mv_path)
    torch.cuda.empty_cache()

    # Gen-only: hand the views + their camera angles to the server, which bakes them
    # with Hunyuan's N-view baker (supports >6 views, e.g. the combined set).
    if args.bake == "none":
        import json

        for i, im in enumerate(images):
            im.convert("RGB").save(os.path.join(args.save_dir, f"{args.save_name}_view{i}.png"))
        with open(os.path.join(args.save_dir, f"{args.save_name}_angles.json"), "w") as f:
            json.dump([[azimuths[i], elevations[i]] for i in range(len(images))], f)
        print("GEN_DONE", len(images))
        return

    # TexturePipeline hardcodes the camera distance list to length 6; for a combined
    # (>6) view set that mismatches the azimuth/elevation count and crashes. Patch its
    # camera builder to size the distance list to the actual number of views.
    import mvadapter.pipelines.pipeline_texture as _tex

    _orig_tex_cam = _tex.get_orthogonal_camera

    def _tex_cam(**kw):
        az = kw.get("azimuth_deg")
        dist = kw.get("distance")
        if az is not None and isinstance(dist, (list, tuple)) and len(dist) != len(az):
            kw["distance"] = [dist[0]] * len(az)
        return _orig_tex_cam(**kw)

    _tex.get_orthogonal_camera = _tex_cam

    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="./checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="./checkpoints/big-lama.pt",
        device=device,
    )

    # load_packed_images also hardcodes a 6-way grid split; size it to our view count so
    # a combined (>6) grid is split correctly (instance attr shadows the bound method).
    import numpy as _np

    _nv_total = len(azimuths)

    def _load_packed(path, _n=_nv_total):
        if path is None:
            return None
        parts = _np.array_split(_np.array(_PILImage.open(path)), _n, axis=1)
        return [_PILImage.fromarray(p) for p in parts]

    texture_pipe.load_packed_images = _load_packed

    out = texture_pipe(
        mesh_path=args.mesh,
        save_dir=args.save_dir,
        save_name=args.save_name,
        uv_unwarp=True,
        preprocess_mesh=False,
        uv_size=args.uv_size,
        rgb_path=mv_path,
        rgb_process_config=ModProcessConfig(view_upscale=args.upscale, inpaint_mode="view"),
        camera_azimuth_deg=azimuths,
        camera_elevation_deg=elevations,
    )
    print("SHADED_GLB:", out.shaded_model_save_path)


if __name__ == "__main__":
    main()
