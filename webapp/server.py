"""
FastAPI backend for the Hunyuan3D-2.1 meshy-style web app.

Flow (single GPU -> single sequential worker thread):
  upload image(s) -> shape job -> preview untextured mesh -> texture job -> download.

The first uploaded image drives shape generation; all uploaded images are used as
texture references. Export to GLB / FBX / .blend; browse previously generated models.

Run from the repo root:  python -m webapp.server --host 0.0.0.0 --port 8080
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.pipeline import TextureWorker

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
OUTPUT_DIR = Path(os.environ.get("HY3D_OUTPUT_DIR", HERE / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_local_env():
    """Load KEY=VALUE pairs from the repo-root .env so local (non-docker) runs see
    OPENAI_API_KEY. Existing environment variables win (setdefault). Docker already
    injects the key via docker-compose, so this is a no-op there."""
    env_path = HERE.parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except Exception as e:  # noqa: BLE001
        print(f"[server] .env load skipped: {e}")


_load_local_env()

BLENDER_BIN = os.environ.get("BLENDER_BIN", "blender")
BLENDER_SCRIPT = HERE / "blender_convert.py"
BLENDER_PROJECT_SCRIPT = HERE / "blender_project.py"
EXPORT_FORMATS = ("glb", "fbx", "blend")

# OpenAI image model used for projection texturing (missing-view fill + GPT texture mode).
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")
AI_VIEW_PROMPTS = {
    "back": "viewed from directly behind — the back of the object",
    "left": "viewed from its left side, rotated 90 degrees",
    "right": "viewed from its right side, rotated 90 degrees",
    "top": "viewed from directly above, top-down",
    "bottom": "viewed from directly below, bottom-up",
}


# Mirror conventions (empirically tuned). The renderer's back-projection bakes a
# normal PHOTO horizontally mirrored, so real photos are flipped; AI views are painted
# onto the renderer's own normal render (already screen-convention) so they are not.
AI_VIEW_MIRROR = os.environ.get("AI_VIEW_MIRROR", "0").lower() in ("1", "true", "yes")
PROJECTION_PHOTO_MIRROR = os.environ.get("PROJECTION_PHOTO_MIRROR", "1").lower() in ("1", "true", "yes")
# Within-view L-R flip when baking MV-Adapter views with Hunyuan (camera selection is
# validated; this only flips content if text/asymmetry comes out mirrored).
_MV_BAKE_MIRROR = os.environ.get("MVADAPTER_BAKE_MIRROR", "0").lower() in ("1", "true", "yes")
# Combined set: down-weight the 3/4-corner views (off the adapter's trained distribution,
# noisier) so the clean canonical views dominate where both see a face; corners then only
# fill faces canonical can't see — reduces ghosting/placement issues on side/back faces.
_MV_CORNER_WEIGHT = float(os.environ.get("MVADAPTER_CORNER_WEIGHT", "0.3"))
# Cosine sharpness for the combined Hunyuan bake (default config is 4). Higher = each
# texel takes more from its single most head-on view -> less ghosting across 12 views.
_MV_BAKE_EXP = float(os.environ.get("MVADAPTER_BAKE_EXP", "8"))
# How each view is fitted to the silhouette: "fill" stretches subject bbox to the outline
# (no gaps, can distort aspect); "contain" preserves aspect (no stretch, may leave edges
# for inpaint). Try "contain" if some faces look stretched/wrong-scaled.
_MV_ALIGN_FIT = os.environ.get("MVADAPTER_ALIGN_FIT", "fill")
# MV-Adapter camera azimuth -> Hunyuan bake azimuth offset. Empirically verified = 0
# (H = az): MV az=0 (reference front) bakes onto GLB +Z, the model-viewer front; az=180
# -> -Z back; az=90 -> +X (viewer right); az=-90 -> -X (viewer left). The previous +90
# rotated every view a quarter-turn (MV-left landed on the viewer front = "front/left
# swap"). Verified with webapp/diag_bake_probe.py (re-bakes views, forward-renders each
# GLB face). Override only if a future mesh-load convention changes.
_MV_AZ_OFFSET = float(os.environ.get("MVADAPTER_AZ_OFFSET", "0"))
# mvgpt: synthesise canonical elevations from the 3/4 source. Default on; set 0 for old refine.
_MVGPT_ELEVATIONS = os.environ.get("MVGPT_ELEVATIONS", "1").lower() not in ("0", "false", "no")
# mvgpt strategy once elevations exist:
#   "blender"  (default) — project the elevations onto the mesh with Blender ortho cameras and
#               bake (standard conventions: no pole/azimuth guessing; uniform ortho framing fixes
#               scale; smart-UV). Best mapping quality + correct top/bottom. Needs Blender (used
#               for export anyway); falls back to "direct" if Blender errors. Writes
#               blenderproj_cam_<side>.png debug renders alongside the GLB.
#   "direct"   — bake the clean elevations straight onto each cardinal face (in-process). Faces are
#               ALWAYS correct: the elevations are at known orientations and baked at fixed angles,
#               with ZERO dependence on MV-Adapter's view labels. (MV-Adapter does NOT reliably
#               align its generated views to cardinal azimuths — its az=-90 can be the front, etc.,
#               depending on the conditioning image — which silently breaks the recolour/transfer
#               pairing and the Hunyuan-bake face mapping. Verified by inspecting raw MV views.)
#   "hybrid"   — MV-Adapter views + a light gpt recolour toward each matching elevation. Aims for
#               MV's spatial consistency + clean palette, BUT is unreliable: MV's view mislabeling
#               pairs the wrong elevation to a view and can place wrong faces. Experimental.
#   "transfer" — MV draft for shape, appearance regenerated from the elevation. gpt cannot hold a
#               non-front viewpoint, so it puts wrong faces on sides. Fallback only.
_MVGPT_MODE = os.environ.get("MVGPT_MODE", "blender").lower()
# Cosine exponent for the direct elevation bake (MVGPT_MODE=direct).
_MVGPT_DIRECT_BAKE_EXP = float(os.environ.get("MVGPT_DIRECT_BAKE_EXP", "6"))
# Hunyuan elevation that places the TOP elevation on the GLB roof (model-viewer +Y). The
# render frame is flipped vs the saved-GLB frame (get_mesh inverts the load transform), so
# the correct pole is empirical — user reported el=-89.99 lands on the underside, so default
# to +89.99. Flip via MVGPT_TOP_EL=-89.99. Verify with webapp/diag_glbframe.py.
_MVGPT_TOP_EL = float(os.environ.get("MVGPT_TOP_EL", "89.99"))


def _prep_view(worker, img, remove_bg: bool, flip: bool):
    """Isolate the subject (rembg -> clean alpha so placement fits the silhouette) and
    optionally flip horizontally before baking."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if remove_bg:
        img = worker.rembg(img.convert("RGB"))
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _openai_paint_view(geometry_img, reference_imgs, angle: str):
    """Paint one view: colorize the rendered geometry of THIS view, matching the
    style/material of the reference image(s). The surface-normal render is the first
    (structural) image so the output follows the silhouette; references drive
    colors/materials/style. Uses gpt-image-2 with a Gemini nano-banana fallback.
    """
    from webapp.image_edit import CARTOON_STYLE, CONSISTENCY_RULE, edit_image

    view_desc = AI_VIEW_PROMPTS.get(angle, "viewed from the front")
    prompt = (
        f"Image 1 is a surface-normal render of a single 3D object, {view_desc}. "
        "Use Image 1 as the absolute geometry, camera, and composition lock. "
        "The output must match Image 1 exactly in viewpoint, camera angle, perspective, "
        "orientation, silhouette, pose, object proportions, framing, scale, crop, and visible "
        "contours. Do not rotate, reframe, resize, move, reshape, restage, simplify, exaggerate, "
        "reinterpret the object, or add any perspective / 3-quarter tilt. "
        ""
        "Paint this geometry: produce ONE fully-coloured, textured render of THIS exact shape "
        "that overlays the normal render pixel-for-pixel and fills the silhouette edge-to-edge. "
        "Use clean, flat, slightly stylised colours with crisp readable shapes, covering every "
        "part of the silhouette. "
        ""
        "Images 2 and onward are reference images for colour, material, texture style, and finish "
        "only. Borrow their palette, material treatment, surface-detail language, and art style, "
        "but never their geometry, silhouette, camera angle, pose, object design, proportions, "
        "composition, lighting setup, background, or extra elements. "
        ""
        "Priority order: 1) match Image 1 geometry, silhouette and composition exactly; "
        "2) cover the whole silhouette with coherent texture; "
        "3) apply the reference palette/material style. "
        "If any instruction conflicts, matching Image 1 always wins. "
        ""
        "Forbidden changes: no new objects, no removed parts, no altered outline, no different "
        "pose, no different viewpoint, no added geometry, no changed proportions, no changed "
        "camera, no changed scale, no text, no logos unless implied by the references, no "
        "background scenery, no shadows or props that change the composition. "
        ""
        "Output a clean single-object render on a plain solid background with soft even lighting, "
        "matching the framing and scale of Image 1. " + CARTOON_STYLE + " " + CONSISTENCY_RULE
    )
    return edit_image([geometry_img, *reference_imgs], prompt, size=(1024, 1024))

FILE_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_(shape|textured)\.glb$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# ---------------------------------------------------------------------------
# Job store + work queue
# ---------------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()
WORK = Queue()

# Filled lazily by the worker thread on first task so the HTTP server can start
# (and serve the UI) while the heavy model loads.
WORKER = {"obj": None, "ready": False, "error": None, "loaded_once": False}
ARGS = None

_INTERNAL_FIELDS = ("source_paths", "processed_image_path", "shape_path", "textured_path",
                    "reference_paths", "reference_sides")


def _job_ref_paths(job):
    """Reference image paths for style: uploaded reference(s), else the shape source(s)."""
    rps = [p for p in (job.get("reference_paths") or []) if p and os.path.exists(p)]
    return rps or list(job.get("source_paths") or [])


def _job_ref_sides(job):
    """Side tags (front/back/left/right/top/bottom/any) parallel to the uploaded
    references. Empty when no references were uploaded (source-image fallback) so the
    refine keeps its legacy 'feed every reference to every view' behavior."""
    rps = [p for p in (job.get("reference_paths") or []) if p and os.path.exists(p)]
    if not rps:
        return []
    sides = job.get("reference_sides") or []
    return [(sides[i].strip().lower() if i < len(sides) and sides[i] else "any") for i in range(len(rps))]


def _job_ref_images(worker, job, remove_bg=True):
    """Load reference images (subject isolated) for the gpt/projection generators."""
    imgs = []
    for p in _job_ref_paths(job):
        img = Image.open(p).convert("RGBA")
        if remove_bg:
            img = worker.rembg(img.convert("RGB"))
        imgs.append(img)
    return imgs


def _set(job_id, **kw):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(kw)


def _public(job):
    """Strip internal-only fields before returning a job over HTTP."""
    return {k: v for k, v in job.items() if k not in _INTERNAL_FIELDS}


def _ensure_model():
    if WORKER["obj"] is not None:
        return WORKER["obj"]
    WORKER["obj"] = TextureWorker(
        output_dir=str(OUTPUT_DIR),
        model_path=ARGS.model_path,
        subfolder=ARGS.subfolder,
        device=ARGS.device,
        low_vram_mode=ARGS.low_vram_mode,
        enable_flashvdm=ARGS.enable_flashvdm,
        compile=ARGS.compile,
        max_num_view=ARGS.max_num_view,
        tex_resolution=ARGS.tex_resolution,
    )
    WORKER["ready"] = True
    WORKER["loaded_once"] = True
    return WORKER["obj"]


def _unload_worker():
    """Free the Hunyuan shape+paint models from RAM/VRAM so a separate-process job
    (MV-Adapter's SDXL) has headroom. Models reload lazily on the next Hunyuan job."""
    obj = WORKER.get("obj")
    if obj is None:
        return
    try:
        obj.release()
    except Exception:  # noqa: BLE001
        pass
    WORKER["obj"] = None
    WORKER["ready"] = False
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    print("[server] unloaded Hunyuan models to free memory")


def _run_shape(job_id):
    job = JOBS[job_id]
    worker = _ensure_model()
    _set(job_id, status="processing_shape", progress=15, message="Generating 3D shape")
    # The first image drives shape generation.
    image = Image.open(job["source_paths"][0]).convert("RGBA")
    shape_path, processed_path = worker.generate_shape(
        uid=job_id,
        image=image,
        remove_background=job["params"]["remove_background"],
        steps=job["params"]["steps"],
        guidance_scale=job["params"]["guidance_scale"],
        seed=job["params"]["seed"],
        octree_resolution=job["params"]["octree_resolution"],
        num_chunks=job["params"]["num_chunks"],
        face_count=job["params"]["face_count"],
    )
    _set(
        job_id,
        status="shape_ready",
        progress=55,
        message="Shape ready",
        shape_path=shape_path,
        processed_image_path=processed_path,
        shape_url=f"/api/files/{Path(shape_path).name}",
    )


def _run_projection(job_id):
    """Projection texturing. The front view is AI-generated from the mesh geometry using
    the uploaded image as STYLE reference only (not projected directly). Real per-angle
    photos the user uploaded are projected directly (flipped to the renderer's
    convention). Unseen areas are inpainted. Same baker/placement as GPT mode."""
    job = JOBS[job_id]
    worker = _ensure_model()
    rb = job["params"]["remove_background"]
    _set(job_id, status="processing_texture", progress=70, message="Preparing views")

    # Style reference(s): dedicated uploads if any, else the shape source/front.
    style_refs = _job_ref_images(worker, job, remove_bg=True)
    # Real per-angle photos (excluding front, which is style-only — not projected).
    view_images = {}
    for angle, p in job.get("view_paths", {}).items():
        if angle == "front" or not (p and os.path.exists(p)):
            continue
        img = Image.open(p).convert("RGBA")
        view_images[angle] = _prep_view(worker, img, remove_bg=rb, flip=PROJECTION_PHOTO_MIRROR)

    # AI-generate the front (always) + any selected fill angles from geometry + style.
    gen_angles = ["front"] + [a for a in job.get("ai_fill_angles", []) if a not in view_images]
    if os.environ.get("OPENAI_API_KEY") and style_refs:
        _set(job_id, message="Capturing geometry views")
        geom = worker.render_view_geometry(shape_glb_path=job["shape_path"], angles=gen_angles)
        for angle in gen_angles:
            if angle not in geom:
                continue
            try:
                _set(job_id, message=f"Painting {angle} view with {OPENAI_IMAGE_MODEL}")
                painted = _openai_paint_view(geom[angle], style_refs, angle)
                painted.save(OUTPUT_DIR / f"{job_id}_aiview_{angle}.png")
                view_images[angle] = _prep_view(worker, painted, remove_bg=True, flip=AI_VIEW_MIRROR)
            except Exception as e:  # noqa: BLE001
                print(f"[server] AI view gen failed for {angle}: {e}")
    elif style_refs and "front" not in view_images:
        # No OpenAI key: fall back to projecting the first reference as the front.
        view_images["front"] = _prep_view(worker, style_refs[0], remove_bg=False, flip=PROJECTION_PHOTO_MIRROR)

    if not view_images:
        raise RuntimeError("No views to project")

    _set(job_id, status="processing_texture", progress=85, message="Baking texture from views")
    textured_path = worker.project_texture(
        uid=job_id, shape_glb_path=job["shape_path"], view_images=view_images, mirror=False
    )
    _set(
        job_id,
        status="completed",
        progress=100,
        message="Done",
        textured_path=textured_path,
        textured_url=f"/api/files/{Path(textured_path).name}",
    )


def _run_gpt_projection(job_id):
    """Geometry-guided gpt-image-2 texturing (StableProjectorz-style).

    Render each canonical view's surface normals, paint them with gpt-image-2 using the
    uploaded image(s) as the style/material reference, then back-project + bake + inpaint
    through the existing projection path.
    """
    job = JOBS[job_id]
    worker = _ensure_model()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (add it to .env or the environment)")

    # Style/material reference(s): dedicated uploads if any, else the shape source(s).
    refs = _job_ref_images(worker, job, remove_bg=True)
    if not refs:
        raise RuntimeError("No reference image provided")

    angles = job.get("gpt_angles") or ["front", "back", "left", "right", "top"]
    _set(job_id, status="processing_texture", progress=68, message="Capturing geometry views")
    geom = worker.render_view_geometry(shape_glb_path=job["shape_path"], angles=angles)

    view_images = {}
    for angle in angles:
        if angle not in geom:
            continue
        try:
            _set(job_id, message=f"Painting {angle} view with {OPENAI_IMAGE_MODEL}")
            painted = _openai_paint_view(geom[angle], refs, angle)
            painted.save(OUTPUT_DIR / f"{job_id}_gptview_{angle}.png")
            # rembg -> clean alpha so placement fits the silhouette (like projection's
            # photos); AI views are screen-convention so no flip by default.
            view_images[angle] = _prep_view(worker, painted, remove_bg=True, flip=AI_VIEW_MIRROR)
        except Exception as e:  # noqa: BLE001
            print(f"[server] gpt paint failed for {angle}: {e}")
    if not view_images:
        raise RuntimeError("gpt-image generation produced no views")

    _set(job_id, status="processing_texture", progress=88, message="Baking texture from painted views")
    textured_path = worker.project_texture(
        uid=job_id, shape_glb_path=job["shape_path"], view_images=view_images, mirror=False
    )
    _set(
        job_id,
        status="completed",
        progress=100,
        message="Done",
        textured_path=textured_path,
        textured_url=f"/api/files/{Path(textured_path).name}",
    )


def _mv_texture(job_id, gpt_refine):
    """MV-Adapter texturing. Runs MV-Adapter in its isolated env (Hunyuan worker freed
    first for VRAM/RAM headroom). When `gpt_refine`, each view is refined via gpt-image-2
    (Gemini nano-banana fallback). The 'combined' (>6) view set is generated then baked
    with Hunyuan's N-view baker (MV-Adapter's own bake is hardwired to 6 views)."""
    from webapp.mvadapter_texture import generate_mv_views, generate_textured_glb

    job = JOBS[job_id]
    if gpt_refine and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError("GPT refine needs OPENAI_API_KEY or GEMINI_API_KEY (.env / environment)")
    refs = _job_ref_paths(job)
    ref_sides = _job_ref_sides(job)
    _src = (job.get("source_paths") or [None])[0]
    viewset = job.get("mv_viewset")
    rb = job["params"]["remove_background"]
    tag = "MV-Adapter + GPT" if gpt_refine else "MV-Adapter"

    # ---- Elevation-first pipeline (mvgpt default) -----------------------------------
    # Synthesise clean head-on elevations (front/left/right/back/top) from the 3/4 source,
    # then bake them DIRECTLY onto each cardinal face. We tried generating MV-Adapter drafts
    # and gpt-transferring the elevations onto them, but gpt-image cannot preserve a non-front
    # viewpoint (it regenerates a front), so side/corner views came out as the wrong face and
    # the bake ghosted. Projecting the elevations themselves guarantees correct faces and exact
    # reference fidelity. Disable elevations: MVGPT_ELEVATIONS=0. Restore the MV+transfer
    # behaviour: MVGPT_DIRECT=0.
    elev_transfer = False
    elev_recolor = False
    if gpt_refine and _MVGPT_ELEVATIONS and _src and os.path.exists(_src):
        from webapp.elevations import ELEVATION_SIDES, generate_elevations
        provided = {s: p for p, s in zip(refs, ref_sides) if s in ELEVATION_SIDES}
        extra = [p for p, s in zip(refs, ref_sides) if s not in ELEVATION_SIDES]
        _set(job_id, status="processing_texture", progress=64,
             message=f"Generating {len(ELEVATION_SIDES)} canonical elevations from the 3/4 source")
        elevations = generate_elevations(_src, provided=provided, extra_context=extra,
                                         out_dir=str(OUTPUT_DIR), uid=job_id)

        if _MVGPT_MODE == "blender":
            # Camera-projection bake in Blender: standard conventions (no pole/azimuth
            # guessing) + uniform ortho framing (no fill-stretch). Faces + scale fixed at
            # the source. Writes blenderproj_cam_<side>.png debug renders alongside. Falls
            # back to the in-process direct bake if Blender is missing or errors.
            _set(job_id, status="processing_texture", progress=86,
                 message="Projecting elevations onto the mesh with Blender")
            try:
                textured_path = _blender_project_bake(job_id, job["shape_path"], elevations)
                _set(job_id, status="completed", progress=100, message="Done",
                     textured_path=textured_path, textured_url=f"/api/files/{Path(textured_path).name}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[server] Blender projection failed ({e}); falling back to direct bake")

        if _MVGPT_MODE in ("direct", "blender"):
            _set(job_id, status="processing_texture", progress=86,
                 message="Baking elevations directly onto the mesh")
            worker = _ensure_model()
            # Cardinal face -> Hunyuan (elev, azim). Equator faces at el=0; top viewed from
            # ABOVE = negative elevation (Hunyuan's get_mv_matrix flips elevation internally).
            # Bottom is omitted (the 3/4 source has no underside) -> UV-inpaint fills it.
            CARDINAL = [("front", 0.0, 0.0), ("right", 0.0, 90.0), ("back", 0.0, 180.0),
                        ("left", 0.0, 270.0), ("top", _MVGPT_TOP_EL, 0.0)]
            items = [(worker.rembg(Image.open(elevations[s]).convert("RGB")), el, az + _MV_AZ_OFFSET)
                     for s, el, az in CARDINAL if elevations.get(s)]
            textured_path = worker.project_texture_angles(
                uid=job_id, shape_glb_path=job["shape_path"], items=items,
                mirror=_MV_BAKE_MIRROR, bake_exp=_MVGPT_DIRECT_BAKE_EXP, fit=_MV_ALIGN_FIT,
            )
            _set(job_id, status="completed", progress=100, message="Done",
                 textured_path=textured_path, textured_url=f"/api/files/{Path(textured_path).name}")
            return

        # hybrid / transfer: elevations become perfectly side-tagged references for the MV path.
        refs = list(elevations.values())
        ref_sides = list(elevations.keys())
        elev_recolor = (_MVGPT_MODE != "transfer")   # hybrid = recolour, keeps MV layout/positions
        elev_transfer = (_MVGPT_MODE == "transfer")  # strict transfer = appearance from elevation

    # ---- MV-Adapter path (needs VRAM: free the Hunyuan worker first) -----------------
    _unload_worker()
    # Conditioning image must be a clean, geometry-matched view. Prefer the SOURCE (the
    # coherent shape-gen input); MVADAPTER_COND=reference forces refs[0].
    _cond_pref = os.environ.get("MVADAPTER_COND", "source").lower()
    if _cond_pref == "reference" or not (_src and os.path.exists(_src)):
        mv_cond = refs[0]
    else:
        mv_cond = _src

    if viewset == "combined":
        # >6 views: generate with MV-Adapter, bake with Hunyuan (handles any view count).
        _set(job_id, status="processing_texture", progress=68, message=f"{tag}: generating 12 views")
        view_paths, angles = generate_mv_views(
            mesh_path=job["shape_path"], mv_image_path=mv_cond, out_dir=str(OUTPUT_DIR),
            uid=job_id, remove_bg=rb, ref_paths=refs, gpt_refine=gpt_refine, viewset=viewset,
            ref_sides=ref_sides, elev_transfer=elev_transfer, recolor=elev_recolor,
        )
        _set(job_id, status="processing_texture", progress=88, message="Baking views with Hunyuan (N-view)")
        worker = _ensure_model()
        # Map MV-Adapter camera (az, el) -> Hunyuan (elev=el, azim=az+_MV_AZ_OFFSET).
        # Offset verified = 0 (see _MV_AZ_OFFSET). Within-view L-R mirror toggle:
        # MVADAPTER_BAKE_MIRROR (default 0 = no flip, verified correct).
        # rembg each view so _align_photo isolates the subject and scales it to the
        # silhouette (the MV views sit on a grey background; without this the whole frame
        # is fit into the outline -> undersized/offset texture).
        # Elevation must be NEGATED: Hunyuan's get_mv_matrix does `elev=-elev` internally,
        # so its camera is ABOVE for negative elevation, while MV-Adapter's camera is above
        # for POSITIVE elevation. Passing MV elevation unflipped bakes every above-looking
        # view (top el=90, tilted corners el=45) from BELOW -> the lot/cars/foliage smear
        # onto the base/underside and top<->bottom swap. Verified by rendering Hunyuan
        # normals at +/-45 and +/-90 (MV-above == Hunyuan-negative-elev). Equator (el=0)
        # is unchanged. Azimuth offset (_MV_AZ_OFFSET) was verified separately at the equator.
        items = [(worker.rembg(Image.open(p).convert("RGB")), -el, az + _MV_AZ_OFFSET)
                 for p, (az, el) in zip(view_paths, angles)]
        # First 6 = canonical (full weight), rest = 3/4 corners (down-weighted fill).
        weights = [1.0 if k < 6 else _MV_CORNER_WEIGHT for k in range(len(items))]
        textured_path = worker.project_texture_angles(
            uid=job_id, shape_glb_path=job["shape_path"], items=items,
            mirror=_MV_BAKE_MIRROR, weights=weights, bake_exp=_MV_BAKE_EXP, fit=_MV_ALIGN_FIT,
        )
    else:
        _set(job_id, status="processing_texture", progress=70, message=f"{tag}: generating multi-view texture (SDXL)")
        textured_path = generate_textured_glb(
            mesh_path=job["shape_path"], mv_image_path=mv_cond, out_dir=str(OUTPUT_DIR),
            uid=job_id, remove_bg=rb, ref_paths=refs, gpt_refine=gpt_refine, viewset=viewset,
            ref_sides=ref_sides, elev_transfer=elev_transfer, recolor=elev_recolor,
        )

    _set(
        job_id,
        status="completed",
        progress=100,
        message="Done",
        textured_path=textured_path,
        textured_url=f"/api/files/{Path(textured_path).name}",
    )


def _run_mvadapter(job_id):
    return _mv_texture(job_id, gpt_refine=False)


def _run_mvgpt(job_id):
    return _mv_texture(job_id, gpt_refine=True)


def _run_texture(job_id):
    job = JOBS[job_id]
    mode = job.get("texture_mode")
    if mode == "projection":
        return _run_projection(job_id)
    if mode == "gptproject":
        return _run_gpt_projection(job_id)
    if mode == "mvadapter":
        return _run_mvadapter(job_id)
    if mode == "mvgpt":
        return _run_mvgpt(job_id)
    worker = _ensure_model()
    _set(job_id, status="processing_texture", progress=70, message="Generating PBR texture")
    # Primary reference = the bg-removed front (reuse shape's processed image, or
    # process the source front when retexturing an existing model).
    proc = job.get("processed_image_path")
    if proc and os.path.exists(proc):
        front = Image.open(proc).convert("RGBA")
    else:
        front = Image.open(job["source_paths"][0]).convert("RGBA")
        if job["params"]["remove_background"]:
            front = worker.rembg(front.convert("RGB"))
    images = [front]
    # Additional images become extra texture references for the same mesh.
    for p in job["source_paths"][1:]:
        img = Image.open(p).convert("RGBA")
        if job["params"]["remove_background"]:
            img = worker.rembg(img.convert("RGB"))
        images.append(img)
    # The Hunyuan paint model conditions on the FIRST image only (single style reference +
    # per-view GEOMETRY control); it cannot take a different face reference per view without
    # retraining. So if the user uploaded reference(s), condition on the front-tagged one (a
    # clean elevation/photo) instead of the raw source; the rest ride along as extra context.
    _rps = [p for p in (job.get("reference_paths") or []) if os.path.exists(p)]
    _rsd = job.get("reference_sides") or []
    _front_ref = next((p for i, p in enumerate(_rps) if i < len(_rsd) and _rsd[i] == "front"), None) \
        or (_rps[0] if _rps else None)
    if _front_ref:
        _fr = Image.open(_front_ref).convert("RGBA")
        if job["params"]["remove_background"]:
            _fr = worker.rembg(_fr.convert("RGB"))
        images = [_fr] + images
    textured_path = worker.generate_texture(
        uid=job_id,
        shape_glb_path=job["shape_path"],
        images=images,
        face_count=job["params"]["face_count"],
        views=job["params"].get("views"),
        tex_resolution=job["params"].get("tex_resolution"),
        albedo_only=job["params"].get("albedo_only", False),
    )
    _set(
        job_id,
        status="completed",
        progress=100,
        message="Done",
        textured_path=textured_path,
        textured_url=f"/api/files/{Path(textured_path).name}",
    )


def _worker_loop():
    if ARGS.preload:
        try:
            _ensure_model()
        except Exception as e:  # noqa: BLE001
            WORKER["error"] = str(e)
            print(f"[server] model preload failed: {e}")
    while True:
        kind, job_id = WORK.get()
        try:
            if kind == "shape":
                _run_shape(job_id)
                if JOBS[job_id].get("auto_texture"):
                    WORK.put(("texture", job_id))
            elif kind == "texture":
                _run_texture(job_id)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            _set(job_id, status="failed", message="Generation failed", error=str(e))
        finally:
            # Auto-clear GPU memory after every job so it never accumulates.
            if WORKER["obj"] is not None:
                try:
                    WORKER["obj"].release()
                except Exception:  # noqa: BLE001
                    pass
            WORK.task_done()


def _blender_convert(src_glb: str, out_path: str):
    """Convert a GLB to FBX/.blend via headless Blender."""
    if not (shutil.which(BLENDER_BIN) or os.path.exists(BLENDER_BIN)):
        raise HTTPException(status_code=503, detail="Blender is not installed in this container")
    cmd = [BLENDER_BIN, "--background", "--python", str(BLENDER_SCRIPT), "--", src_glb, out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or proc.stdout or "")[-600:]
        raise HTTPException(status_code=500, detail=f"Blender conversion failed: {tail}")


def _blender_project_bake(job_id, shape_glb, elevations):
    """Project the side elevations onto the mesh and bake to a GLB via headless Blender
    (camera-projection bake; standard conventions, uniform ortho scale). Returns GLB path."""
    if not (shutil.which(BLENDER_BIN) or os.path.exists(BLENDER_BIN)):
        raise RuntimeError("Blender is not installed in this container (BLENDER_BIN)")
    out_glb = str(OUTPUT_DIR / f"{job_id}_textured.glb")
    spec = {
        "mesh": os.path.abspath(shape_glb),
        "out": os.path.abspath(out_glb),
        "tex_size": int(os.environ.get("MVGPT_BLENDER_TEX", "2048")),
        "views": [{"side": s, "image": os.path.abspath(p)} for s, p in elevations.items()],
        "debug_dir": str(OUTPUT_DIR),  # writes blenderproj_cam_<side>.png for verification
    }
    spec_path = OUTPUT_DIR / f"{job_id}_blenderproj_spec.json"
    spec_path.write_text(json.dumps(spec))
    cmd = [BLENDER_BIN, "--background", "--python", str(BLENDER_PROJECT_SCRIPT), "--", str(spec_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0 or not os.path.exists(out_glb):
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        raise RuntimeError(f"Blender projection bake failed: {tail}")
    return out_glb


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Hunyuan3D-2.1 Studio")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def _no_cache_ui(request, call_next):
    """Never let the browser cache the UI (html/js/css) — otherwise mode/wiring changes
    silently don't take effect until a hard refresh."""
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.get("/api/health")
def health():
    try:
        from webapp.mvadapter_texture import is_available as _mv_available
        mvadapter = _mv_available()
    except Exception:  # noqa: BLE001
        mvadapter = False
    return {
        # Operational once the model has loaded at least once — an intentional unload
        # (MV-Adapter frees the Hunyuan worker) lazy-reloads on the next job, so don't
        # report it as "warming up".
        "model_ready": bool(WORKER["ready"] or WORKER["loaded_once"]),
        "model_error": WORKER["error"],
        "queue": WORK.qsize(),
        "blender": bool(shutil.which(BLENDER_BIN) or os.path.exists(BLENDER_BIN)),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "mvadapter": mvadapter,
    }


@app.post("/api/generate")
async def generate(
    images: list[UploadFile] = File(...),
    remove_background: bool = Form(True),
    auto_texture: bool = Form(False),
    steps: int = Form(30),
    guidance_scale: float = Form(5.0),
    seed: int = Form(1234),
    octree_resolution: int = Form(256),
    num_chunks: int = Form(8000),
    face_count: int = Form(40000),
    views: int = Form(7),          # 7/512 is the reliable max on 16GB (8 or 768 thrash)
    tex_resolution: int = Form(512),
    albedo_only: bool = Form(False),      # flat colors: drop metallic/roughness
    texture_mode: str = Form("hunyuan"),  # "hunyuan" (AI) | "projection" (your photos) | "gptproject" (gpt-image-2)
    ai_fill_angles: str = Form(""),       # projection: comma-sep angles to synth from front via OpenAI
    gpt_angles: str = Form("front,back,left,right,top"),  # gptproject: comma-sep canonical angles to paint
    mv_viewset: str = Form("canonical"),  # mvadapter/mvgpt: canonical | corners | tilted
    back: UploadFile = File(None),
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    top: UploadFile = File(None),
    bottom: UploadFile = File(None),
    reference: list[UploadFile] = File(None),  # gpt/mvgpt: optional style reference image(s)
    reference_side: list[str] = Form(None),    # per-reference side tag, parallel to `reference`
):
    if not images:
        raise HTTPException(status_code=400, detail="No image provided")

    job_id = str(uuid.uuid4())

    async def _save(up, name):
        raw = await up.read()
        try:
            Image.open(io.BytesIO(raw)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {up.filename}")
        path = OUTPUT_DIR / f"{job_id}_{name}.png"
        Image.open(io.BytesIO(raw)).convert("RGBA").save(path)
        return str(path)

    source_paths = [await _save(up, f"source{idx}") for idx, up in enumerate(images)]

    # Per-angle photos for projection texturing (front = the main image).
    view_paths = {"front": source_paths[0]}
    for angle, up in (("back", back), ("left", left), ("right", right), ("top", top), ("bottom", bottom)):
        if up is not None:
            view_paths[angle] = await _save(up, f"view_{angle}")

    reference_paths = []
    reference_sides = []
    _rsides = reference_side or []
    for idx, up in enumerate(reference or []):
        if up is not None and getattr(up, "filename", ""):
            reference_paths.append(await _save(up, f"reference{idx}"))
            reference_sides.append((_rsides[idx].strip().lower() if idx < len(_rsides) and _rsides[idx] else "any"))

    job = {
        "id": job_id,
        "status": "queued",
        "progress": 5,
        "message": "Queued",
        "error": None,
        "shape_url": None,
        "textured_url": None,
        "auto_texture": bool(auto_texture),
        "texture_mode": texture_mode,
        "ai_fill_angles": [a.strip() for a in ai_fill_angles.split(",") if a.strip()],
        "gpt_angles": [a.strip() for a in gpt_angles.split(",") if a.strip()],
        "mv_viewset": mv_viewset,
        "reference_paths": reference_paths,
        "reference_sides": reference_sides,
        "num_images": len(source_paths),
        "source_paths": source_paths,
        "view_paths": view_paths,
        "created_at": time.time(),
        "params": {
            "remove_background": bool(remove_background),
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
            "octree_resolution": int(octree_resolution),
            "num_chunks": int(num_chunks),
            "face_count": int(face_count),
            "views": int(views),
            "tex_resolution": int(tex_resolution),
            "albedo_only": bool(albedo_only),
        },
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    WORK.put(("shape", job_id))
    return {"id": job_id}


@app.post("/api/jobs/{job_id}/texture")
async def request_texture(
    job_id: str,
    texture_mode: str = Form(None),
    ai_fill_angles: str = Form(None),
    gpt_angles: str = Form(None),
    mv_viewset: str = Form(None),
    remove_background: bool = Form(None),
    reference: list[UploadFile] = File(None),
    reference_side: list[str] = Form(None),
):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    if job["status"] not in ("shape_ready", "completed"):
        raise HTTPException(status_code=409, detail=f"Shape not ready (status={job['status']})")

    # Honor the CURRENT UI selection (mode/refs can change after the shape was generated).
    if texture_mode:
        job["texture_mode"] = texture_mode
    if ai_fill_angles is not None:
        job["ai_fill_angles"] = [a.strip() for a in ai_fill_angles.split(",") if a.strip()]
    if gpt_angles is not None:
        job["gpt_angles"] = [a.strip() for a in gpt_angles.split(",") if a.strip()]
    if mv_viewset:
        job["mv_viewset"] = mv_viewset
    if remove_background is not None:
        job["params"]["remove_background"] = bool(remove_background)
    new_refs = []
    new_sides = []
    _rsides = reference_side or []
    for idx, up in enumerate(reference or []):
        if up is not None and getattr(up, "filename", ""):
            raw = await up.read()
            path = OUTPUT_DIR / f"{job_id}_reference{idx}.png"
            Image.open(io.BytesIO(raw)).convert("RGBA").save(path)
            new_refs.append(str(path))
            new_sides.append((_rsides[idx].strip().lower() if idx < len(_rsides) and _rsides[idx] else "any"))
    if new_refs:
        job["reference_paths"] = new_refs
        job["reference_sides"] = new_sides

    _set(job_id, status="queued_texture", progress=60, message="Queued for texturing", error=None)
    WORK.put(("texture", job_id))
    return {"id": job_id, "status": "queued_texture"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return _public(job)


@app.get("/api/jobs/{job_id}/download/{fmt}")
def download(job_id: str, fmt: str):
    """Export a generated model. glb is native; fbx/blend go through Blender."""
    if not UUID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Bad job id")
    fmt = fmt.lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    src = OUTPUT_DIR / f"{job_id}_textured.glb"
    if not src.exists():
        src = OUTPUT_DIR / f"{job_id}_shape.glb"
    if not src.exists():
        raise HTTPException(status_code=404, detail="No model for this job")

    if fmt == "glb":
        return FileResponse(src, media_type="model/gltf-binary", filename=f"{job_id}.glb")

    out = OUTPUT_DIR / f"{src.stem}.{fmt}"  # cached, e.g. {id}_textured.fbx
    if not out.exists():
        _blender_convert(str(src), str(out))
    return FileResponse(out, media_type="application/octet-stream", filename=f"{job_id}.{fmt}")


@app.get("/api/gallery")
def gallery():
    """Previously generated models, newest first (rebuilt from disk so it survives restarts)."""
    items = {}
    for p in OUTPUT_DIR.glob("*_shape.glb"):
        items.setdefault(p.name[: -len("_shape.glb")], {})["shape"] = p
    for p in OUTPUT_DIR.glob("*_textured.glb"):
        items.setdefault(p.name[: -len("_textured.glb")], {})["textured"] = p

    out = []
    for jid, d in items.items():
        best = d.get("textured") or d.get("shape")
        out.append({
            "id": jid,
            "shape_url": f"/api/files/{jid}_shape.glb" if "shape" in d else None,
            "textured_url": f"/api/files/{jid}_textured.glb" if "textured" in d else None,
            "preview_url": f"/api/files/{best.name}",
            "textured": "textured" in d,
            "mtime": best.stat().st_mtime,
        })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete all files for a generated model."""
    if not UUID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Bad job id")
    removed = 0
    for p in OUTPUT_DIR.glob(f"{job_id}_*"):
        try:
            p.unlink()
            removed += 1
        except Exception:  # noqa: BLE001
            pass
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    return {"deleted": job_id, "files": removed}


@app.post("/api/retexture")
async def retexture(
    source_id: str = Form(...),
    images: list[UploadFile] = File(...),
    texture_mode: str = Form("hunyuan"),
    ai_fill_angles: str = Form(""),
    gpt_angles: str = Form("front,back,left,right,top"),
    mv_viewset: str = Form("canonical"),
    remove_background: bool = Form(True),
    face_count: int = Form(40000),
    views: int = Form(7),
    tex_resolution: int = Form(512),
    albedo_only: bool = Form(False),
    back: UploadFile = File(None),
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    top: UploadFile = File(None),
    bottom: UploadFile = File(None),
    reference: list[UploadFile] = File(None),
    reference_side: list[str] = Form(None),
):
    """Texture/retexture an existing model's mesh without regenerating the shape."""
    if not UUID_RE.match(source_id):
        raise HTTPException(status_code=400, detail="Bad source id")
    src_shape = OUTPUT_DIR / f"{source_id}_shape.glb"
    if not src_shape.exists():
        raise HTTPException(status_code=404, detail="No untextured shape for that model")

    job_id = str(uuid.uuid4())
    shutil.copyfile(src_shape, OUTPUT_DIR / f"{job_id}_shape.glb")

    async def _save(up, name):
        raw = await up.read()
        try:
            Image.open(io.BytesIO(raw)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {up.filename}")
        path = OUTPUT_DIR / f"{job_id}_{name}.png"
        Image.open(io.BytesIO(raw)).convert("RGBA").save(path)
        return str(path)

    source_paths = [await _save(up, f"source{idx}") for idx, up in enumerate(images)]
    view_paths = {"front": source_paths[0]}
    for angle, up in (("back", back), ("left", left), ("right", right), ("top", top), ("bottom", bottom)):
        if up is not None:
            view_paths[angle] = await _save(up, f"view_{angle}")

    reference_paths = []
    reference_sides = []
    _rsides = reference_side or []
    for idx, up in enumerate(reference or []):
        if up is not None and getattr(up, "filename", ""):
            reference_paths.append(await _save(up, f"reference{idx}"))
            reference_sides.append((_rsides[idx].strip().lower() if idx < len(_rsides) and _rsides[idx] else "any"))

    job = {
        "id": job_id, "status": "shape_ready", "progress": 55, "message": "Shape ready (retexture)",
        "error": None, "shape_url": f"/api/files/{job_id}_shape.glb", "textured_url": None,
        "auto_texture": False, "texture_mode": texture_mode,
        "ai_fill_angles": [a.strip() for a in ai_fill_angles.split(",") if a.strip()],
        "gpt_angles": [a.strip() for a in gpt_angles.split(",") if a.strip()],
        "mv_viewset": mv_viewset,
        "reference_paths": reference_paths,
        "reference_sides": reference_sides,
        "num_images": len(source_paths), "source_paths": source_paths, "view_paths": view_paths,
        "shape_path": str(OUTPUT_DIR / f"{job_id}_shape.glb"), "created_at": time.time(),
        "params": {
            "remove_background": bool(remove_background), "steps": 30, "guidance_scale": 5.0,
            "seed": 1234, "octree_resolution": 256, "num_chunks": 8000,
            "face_count": int(face_count), "views": int(views), "tex_resolution": int(tex_resolution),
            "albedo_only": bool(albedo_only),
        },
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    WORK.put(("texture", job_id))
    return {"id": job_id}


@app.get("/api/files/{name}")
def get_file(name: str):
    if not FILE_RE.match(name):
        raise HTTPException(status_code=400, detail="Bad file name")
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=name)


# Static UI mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mc_algo", type=str, default="mc")
    parser.add_argument("--max_num_view", type=int, default=6)
    parser.add_argument("--tex_resolution", type=int, default=512)
    parser.add_argument("--enable_flashvdm", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument("--preload", action="store_true", help="Load model at startup instead of on first job")
    ARGS = parser.parse_args()

    threading.Thread(target=_worker_loop, daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")


if __name__ == "__main__":
    main()
