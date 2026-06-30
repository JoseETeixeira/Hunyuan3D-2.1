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
# Newer Blender used ONLY to read user-uploaded .blend files (a 4.2 binary can't open files saved by
# Blender 5.x). Falls back to BLENDER_BIN when absent.
BLENDER_NEW_BIN = os.environ.get("BLENDER_NEW_BIN", "blender5")
BLENDER_SCRIPT = HERE / "blender_convert.py"
BLENDER_PROJECT_SCRIPT = HERE / "blender_project.py"
BLENDER_BLEND_TO_GLB_SCRIPT = HERE / "blender_blend_to_glb.py"
BLENDER_FILLHOLES_SCRIPT = HERE / "blender_fillholes.py"
BLENDER_DUMP_SKELETON_SCRIPT = HERE / "blender_dump_skeleton.py"
BLENDER_EDIT_SKELETON_SCRIPT = HERE / "blender_edit_skeleton.py"
# Fill boundary holes (via Blender) on every generated shape so meshes are watertight by default.
FILL_HOLES = os.environ.get("HY3D_FILL_HOLES", "1") not in ("0", "false", "False", "")

# UniRig (https://github.com/VAST-AI-Research/UniRig) auto-rigging — runs in its OWN python env as a
# subprocess (heavy, incompatible deps), like Blender. Point UNIRIG_DIR at the checkout and
# UNIRIG_PYTHON at that venv's python; bash runs its launch/inference/*.sh scripts.
UNIRIG_DIR = os.environ.get("UNIRIG_DIR", "")
UNIRIG_PYTHON = os.environ.get("UNIRIG_PYTHON", "python")
UNIRIG_BASH = os.environ.get("UNIRIG_BASH", "bash")
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
# hyface per-face paint — fill views add angular coverage so the oblique/recessed texels
# the 6 cardinal views only graze (low cos -> masked -> inpainted -> look unpainted) get
# actually painted. Corners (3/4 diagonals tilted down) reach recessed tops; tilted
# cardinals thicken each face above/below the equator. Fills are down-weighted so the
# head-on cardinal faces stay crisp; a lower bake_exp spreads contribution into the gaps.
# Cost scales with view count (each view = one diffusion pass) — dial back via these knobs.
_HYFACE_CORNERS = os.environ.get("HYFACE_CORNERS", "1").lower() not in ("0", "false", "no")
_HYFACE_CORNER_ELEV = float(os.environ.get("HYFACE_CORNER_ELEV", "45"))   # down-tilt to see over recessed tops
_HYFACE_TILT = os.environ.get("HYFACE_TILT", "1").lower() not in ("0", "false", "no")
_HYFACE_TILT_ELEV = float(os.environ.get("HYFACE_TILT_ELEV", "20"))       # cardinal tilt magnitude (deg)
_HYFACE_FILL_WEIGHT = float(os.environ.get("HYFACE_FILL_WEIGHT", "0.3"))  # fill weight vs cardinals (1.0)
# Below-horizon outward fills. Every other hyface camera looks horizontally (cardinals elev 0) or DOWN
# (tilts +/-20, corners +45), so a prop's DOWN-AND-OUT lower faces (car sills/rockers/lower bumpers/wheel
# sides) sit >75deg off every view axis -> the cosine bake gate (bake_angle_thres=75) zeros them -> they
# fall to UV inpaint, which (cv2.INPAINT_NS, island-blind) pulls the adjacent tan GROUND colour onto them
# (the "blue car has a tan lower" wash). A camera BELOW the horizon looking up-and-out brings those normals
# inside the gate AND sees the lower car against empty background (not the lot), so it bakes real car
# colour. Env-gated; HYFACE_LOW=0 reverts. Cost: +1 diffusion pass per equatorial cardinal present.
# NOTE: below-horizon views aim UNDER the model and paint the base/underside, not the cars' lower-SIDE
# faces (which face outward, ~horizontal) — so this does not address the side wash. Default OFF.
_HYFACE_LOW = os.environ.get("HYFACE_LOW", "0").lower() not in ("0", "false", "no")
_HYFACE_LOW_ELEV = float(os.environ.get("HYFACE_LOW_ELEV", "-45"))        # below-horizon look-up angle (deg)
# Bake cosine exponent. The bake blends views by weight*cos(view,normal)**exp; a LOW exp blends many
# overlapping views per texel, so on closely-spaced objects (the two cars) each car's inner/side faces
# pick up a soft mix of the front + neighbour-side projections (blue car gets a tan-ish flank, tan car a
# blue one). A HIGH exp drives the bake toward winner-take-all: each texel takes the single most head-on
# view that is NOT occluded there (back_project already zeroes occluded texels), so the correct view wins
# and the cross-car bleed collapses (same single-winner behaviour that keeps the mvgpt blender bake clean).
_HYFACE_BAKE_EXP = float(os.environ.get("HYFACE_BAKE_EXP", "8"))

# reface — depth-aware single-face re-texture of an already-textured mesh. Foreground band = the
# nearest fraction (0..1) of the face's depth range to repaint. Default 1.0 = every visible
# camera-facing surface (back_project skips occluded + grazing texels itself, so "farther" != "behind").
# A smaller value keeps only the nearest slab — set it to repaint just the frontmost object and leave
# farther-but-visible ones untouched — but a small band drops the car roofs on a TOP/BOTTOM view (roof
# near, cars far below), which is why the default repaints all visible surfaces. A user mask overrides it.
_REFACE_DEPTH_BAND = float(os.environ.get("REFACE_DEPTH_BAND", "1.0"))
# Slight front-cardinal weight bump so the front (which sees the whole lot in true colours) wins ties on
# shared front-lot faces; with the high bake exponent above this only needs to break ties, not dominate.
_HYFACE_FRONT_WEIGHT = float(os.environ.get("HYFACE_FRONT_WEIGHT", "1.5"))
# HYBRID bake: keep the Hunyuan cosine bake for the whole model, then re-bake ONLY the below-horizon
# (`_lo`) views through the occlusion-aware single-winner blender projection, overlaying just the cars'
# DOWN-FACING lower faces (sills/rockers/underside) on top of the cosine result. The cosine bake handles
# those faces badly (no camera sees them head-on -> inpaint pulls the tan ground/foliage onto them); the
# single-winner low views give each one exactly its own view's car colour. Scoped to `_lo` only, so the
# main faces (building, roofs, upper car) stay on the cosine bake — avoids the full-blender regression.
# HYFACE_BLENDER_BAKE=0 reverts to plain cosine. OFF by default — it only re-baked the `_lo` views,
# which target the underside, not the side wash.
_HYFACE_BLENDER_BAKE = os.environ.get("HYFACE_BLENDER_BAKE", "0").lower() not in ("0", "false", "no")
# Only the below-horizon fills are re-baked via blender (their cameras carry the down-facing guard).
_HYFACE_BAKE_SIDES = {"front_lo", "back_lo", "left_lo", "right_lo", "fl_lo", "fr_lo", "bl_lo", "br_lo"}


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
        "Keep each DISTINCT object's OWN colour — match a colour to an object by its position/identity "
        "across the references, do NOT wash every similar object into one palette (e.g. a blue car on "
        "one side and a tan car on another are SEPARATE vehicles and KEEP their own colours; never paint "
        "the blue car tan or the tan car blue). "
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


# Canonical faces for per-face paint (order = display + dedup order).
HYFACE_FACES = ["front", "back", "left", "right", "top", "bottom"]

# Corner fill cameras: {label: (azimuth, [adjacent cardinal faces])}. These corner azimuths are
# tuned independently (NOT derived from PROJECTION_CAMS, whose cardinal left/right are 270/90); each
# corner tilts down (+elev) to reach recessed tops. The adjacent faces' references seed the gpt-synth
# corner reference (geometry render is the layout lock).
# Back corners bl/br empirically corrected (135/225, swapped from the old 225/135) so they frame the
# object's OWN back-left / back-right — matching studio.CORNER_AZ. Front corners unchanged.
HYFACE_CORNER_CAMS = {
    "fl": (315.0, ["front", "left"]),
    "bl": (135.0, ["back", "left"]),
    "br": (225.0, ["back", "right"]),
    "fr": (45.0, ["front", "right"]),
}
# Equatorial cardinals that also get tilted (elev +/-) views. Poles (top/bottom) excluded.
HYFACE_TILT_FACES = ["front", "back", "left", "right"]


def _prep_face_ref(worker, img, rb):
    """Background-prep a hyface paint reference. The salient-object remover (rembg) OVER-SEGMENTS the
    multi-element generated/staged refs (building + storefront + lot + cars on a near-black field): it
    keeps only the high-contrast upper facade and cuts the lower storefront/pillars/cars/lot. Those cut
    pixels then composite to WHITE, so Hunyuan paints the lower front blank -> the unpainted white areas.
    When the reference already has a near-uniform DARK background, key out ONLY that background by a tight
    threshold and keep the whole subject; fall back to rembg only for real photos (non-uniform background).
    Returns an RGB image with a clean white background (paint_faces consumes RGB as-is)."""
    import numpy as np
    rgb = np.asarray(img.convert("RGB"))
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0).astype(np.int16)
    dark_uniform = float((border.max(-1) < 30).mean()) > 0.6
    if rb and dark_uniform:
        bg = rgb.max(-1) < 28                       # only the near-pure-black frame, not dark glass/navy
        out = rgb.copy()
        out[bg] = 255
        return Image.fromarray(out)
    if rb:
        return worker.rembg(img.convert("RGB"))
    return img.convert("RGB")


def _run_hyface(job_id):
    """Per-face AI paint (Hunyuan). Each face is painted INDEPENDENTLY with its own
    uploaded reference via single-view Hunyuan paint, then all views bake into one shared
    UV texture (cosine-blend + inpaint). Cardinal faces with no upload are filled by
    gpt-image-2 from their geometry + the other references (when OPENAI_API_KEY is set).
    Down-weighted FILL views add depth coverage: tilted cardinals (elev +/-) and gpt-synth
    3/4 corners reach the oblique/recessed texels the head-on cardinal views only graze
    (otherwise masked -> inpainted -> look unpainted). Albedo-only matte. Additive: leaves
    every other texture mode untouched. Fill set tunable via HYFACE_* env vars."""
    job = JOBS[job_id]
    worker = _ensure_model()
    rb = job["params"]["remove_background"]
    _set(job_id, status="processing_texture", progress=66, message="Preparing per-face references")

    # Per-face uploads: view_paths {angle: path} (front = main image + back/left/right/top/bottom).
    face_refs = {}
    for angle, p in (job.get("view_paths") or {}).items():
        if angle == "front" or worker.PROJECTION_CAMS.get(angle) is None or not (p and os.path.exists(p)):
            continue
        face_refs[angle] = _prep_face_ref(worker, Image.open(p), rb)

    # Front face reference, in priority order:
    #   1) an explicit Front-slot upload (view_paths["front"] saved as *_view_front, i.e.
    #      different from the shape source) — what the user provided for the front;
    #   2) the clean bg-removed front saved at shape gen (processed_image_path), like the
    #      default hunyuan mode;
    #   3) the raw main shape image (source_paths[0]).
    # The per-side loop above skips "front" so this block fully controls it.
    _src0 = (job.get("source_paths") or [None])[0]
    _vp_front = (job.get("view_paths") or {}).get("front")
    if _vp_front and _vp_front != _src0 and os.path.exists(_vp_front):
        face_refs["front"] = _prep_face_ref(worker, Image.open(_vp_front), rb)
    else:
        _front = job.get("processed_image_path")
        if _front and os.path.exists(_front):
            face_refs["front"] = Image.open(_front).convert("RGBA")
        elif _src0 and os.path.exists(_src0):
            face_refs["front"] = _prep_face_ref(worker, Image.open(_src0), rb)

    # Target face set = uploaded faces + requested canonical faces (default gpt_angles).
    requested = [a for a in (job.get("gpt_angles") or HYFACE_FACES) if worker.PROJECTION_CAMS.get(a)]
    target_faces = sorted(set(face_refs) | set(requested), key=HYFACE_FACES.index)

    # gpt-image-2 fills faces with no upload, from their geometry + the uploaded refs.
    empty = [a for a in target_faces if a not in face_refs]
    other_refs = list(face_refs.values())
    if empty and os.environ.get("OPENAI_API_KEY") and other_refs:
        _set(job_id, status="processing_texture", progress=72, message="Capturing geometry for empty faces")
        geom = worker.render_view_geometry(shape_glb_path=job["shape_path"], angles=empty)
        for angle in empty:
            if angle not in geom:
                continue
            try:
                _set(job_id, message=f"Synthesizing {angle} reference with {OPENAI_IMAGE_MODEL}")
                painted = _openai_paint_view(geom[angle], other_refs, angle)
                painted.save(OUTPUT_DIR / f"{job_id}_hyfaceref_{angle}.png")
                # Style reference only (fed back into Hunyuan paint), so no bake flip.
                face_refs[angle] = _prep_view(worker, painted, remove_bg=True, flip=False)
            except Exception as e:  # noqa: BLE001
                print(f"[server] hyface ref synth failed for {angle}: {e}")

    if not face_refs:
        raise RuntimeError("No face references — upload at least one side, or set OPENAI_API_KEY to synthesize them")

    # Cardinal faces head-on; FRONT gets a higher weight so it owns shared front-lot faces (props in
    # true per-object colours) instead of a side view repainting the neighbour car its own colour.
    # `view_labels` (parallel to view_specs) names ONLY the below-horizon `_lo` views, whose painted
    # albedo we save + re-bake via the single-winner blender hybrid; every other view is unlabelled
    # (None) and contributes only to the cosine bake.
    view_specs, view_labels = [], []
    for a, img in face_refs.items():
        if not worker.PROJECTION_CAMS.get(a):
            continue
        view_specs.append((img, worker.PROJECTION_CAMS[a][0], worker.PROJECTION_CAMS[a][1],
                           _HYFACE_FRONT_WEIGHT if a == "front" else 1.0))
        view_labels.append(None)

    # Tilted cardinal fills: each equatorial face also painted at elev +/-tilt (same azimuth,
    # same reference) so coverage extends into oblique surfaces above/below the equator.
    if _HYFACE_TILT:
        for a in HYFACE_TILT_FACES:
            if a not in face_refs:
                continue
            _e, _az = worker.PROJECTION_CAMS[a]
            for _dt in (_HYFACE_TILT_ELEV, -_HYFACE_TILT_ELEV):
                view_specs.append((face_refs[a], _e + _dt, _az, _HYFACE_FILL_WEIGHT))
                view_labels.append(None)

    # Corner fills (3/4 diagonals, tilted down to reach diagonal/recessed texels no cardinal
    # view sees head-on). Reference per corner, in priority order:
    #   1) an explicit corner upload (view_paths[lbl], the fl/fr/bl/br slots) — used directly;
    #   2) gpt-synth from the corner geometry render + adjacent faces (needs OPENAI_API_KEY).
    if _HYFACE_CORNERS:
        corner_refs = {}
        vp = job.get("view_paths") or {}
        for lbl in HYFACE_CORNER_CAMS:
            p = vp.get(lbl)
            if p and os.path.exists(p):
                corner_refs[lbl] = _prep_face_ref(worker, Image.open(p), rb)

        # gpt-synth only the corners the user didn't upload (and that have adjacent face refs).
        to_synth = [lbl for lbl, (az, faces) in HYFACE_CORNER_CAMS.items()
                    if lbl not in corner_refs and any(f in face_refs for f in faces)]
        if to_synth and os.environ.get("OPENAI_API_KEY"):
            cams = [(lbl, _HYFACE_CORNER_ELEV, HYFACE_CORNER_CAMS[lbl][0]) for lbl in to_synth]
            _set(job_id, status="processing_texture", progress=78, message="Capturing corner geometry")
            cgeom = worker.render_geometry_at(job["shape_path"], cams)
            for lbl in to_synth:
                if lbl not in cgeom:
                    continue
                _, faces = HYFACE_CORNER_CAMS[lbl]
                adj = [face_refs[f] for f in faces if f in face_refs]
                if not adj:
                    continue
                if "top" in face_refs:
                    adj = adj + [face_refs["top"]]  # tilted-down corners also see the top
                try:
                    _set(job_id, message=f"Synthesizing {lbl} corner with {OPENAI_IMAGE_MODEL}")
                    painted = _openai_paint_view(cgeom[lbl], adj, lbl)
                    painted.save(OUTPUT_DIR / f"{job_id}_hyfaceref_{lbl}.png")
                    corner_refs[lbl] = _prep_view(worker, painted, remove_bg=True, flip=False)
                except Exception as e:  # noqa: BLE001
                    print(f"[server] hyface corner synth failed for {lbl}: {e}")

        for lbl, cref in corner_refs.items():
            view_labels.append(None)
            az, _ = HYFACE_CORNER_CAMS[lbl]
            view_specs.append((cref, _HYFACE_CORNER_ELEV, az, _HYFACE_FILL_WEIGHT))
            # Diagonal below-horizon corner fill: from the opposite corner BELOW, the camera looks up
            # through the gap between props and catches each car's INNER-lower faces (the ones the
            # cardinal lows can't reach because the neighbouring car occludes the facing cardinal).
            # Labelled `{lbl}_lo` so its albedo is saved + single-winner re-baked onto those lower faces.
            if _HYFACE_LOW:
                view_specs.append((cref, _HYFACE_LOW_ELEV, az, _HYFACE_FILL_WEIGHT))
                view_labels.append(f"{lbl}_lo")

    # Below-horizon outward fills for the equatorial cardinals (same ref + azimuth, NEGATIVE elev) so the
    # cars' down-and-out lower/side faces fall inside the cosine bake's angle gate and bake real car colour
    # instead of being inpainted from the tan ground. From below, the lower car is seen against empty
    # background, so these views do not carry the lot colour the down-tilted fills do.
    if _HYFACE_LOW:
        for a in HYFACE_TILT_FACES:
            if a not in face_refs:
                continue
            _az = worker.PROJECTION_CAMS[a][1]
            view_specs.append((face_refs[a], _HYFACE_LOW_ELEV, _az, _HYFACE_FILL_WEIGHT))
            view_labels.append(f"{a}_lo")  # saved + single-winner re-baked onto the down-facing lower faces

    _set(job_id, status="processing_texture", progress=84,
         message=f"Hunyuan painting {len(view_specs)} views ({len(face_refs)} faces + fills)")
    textured_path = worker.paint_faces(
        uid=job_id, shape_glb_path=job["shape_path"], view_specs=view_specs,
        bake_exp=_HYFACE_BAKE_EXP, tex_resolution=job["params"].get("tex_resolution"),
        albedo_labels=view_labels if _HYFACE_BLENDER_BAKE else None,
    )
    # Single-winner re-bake: project the per-face Hunyuan albedos with the occlusion-aware blender baker
    # (one view per face, no cross-object cosine blend), overlaying the Hunyuan cosine bake as the base
    # so faces no view wins keep a sensible colour. Fixes the closely-spaced cars' flank colour wash.
    if _HYFACE_BLENDER_BAKE:
        try:
            base_glb = OUTPUT_DIR / f"{job_id}_hycosine.glb"
            shutil.copyfile(textured_path, base_glb)
            elevations = {lbl: str(OUTPUT_DIR / f"{job_id}_hyalbedo_{lbl}.png")
                          for lbl in view_labels
                          if lbl in _HYFACE_BAKE_SIDES
                          and (OUTPUT_DIR / f"{job_id}_hyalbedo_{lbl}.png").exists()}
            if elevations:
                _set(job_id, status="processing_texture", progress=92,
                     message="Single-winner projection bake (Blender)")
                textured_path = _blender_project_bake(job_id, job["shape_path"], elevations,
                                                      base_glb=str(base_glb))
        except Exception as e:  # noqa: BLE001
            print(f"[server] hyface blender re-bake failed ({e}); keeping cosine bake")
    _set(job_id, status="completed", progress=100, message="Done",
         textured_path=textured_path, textured_url=f"/api/files/{Path(textured_path).name}")


def _run_reface(job_id):
    """Depth-aware single-face re-texture of an already-textured mesh. Generates the chosen
    face via the gpt geomatch (the mvgpt 'gpt refine' generation from geometry + references),
    then bakes ONLY the nearest depth band (foreground) over the existing texture — farther
    surfaces are left as-is (a car in front of a wall: only the car is repainted)."""
    job = JOBS[job_id]
    worker = _ensure_model()
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError("reface needs OPENAI_API_KEY or GEMINI_API_KEY (.env / environment)")
    textured_glb = job.get("reface_src_glb")
    if not (textured_glb and os.path.exists(textured_glb)):
        raise RuntimeError("reface: source textured mesh not found")
    face = job.get("reface_face", "front")
    # Resolve the view's camera: cardinal faces from PROJECTION_CAMS, 3/4 corners from the
    # corner table (azimuth + the corner down-tilt elevation).
    _cam = worker.PROJECTION_CAMS.get(face)
    if _cam is not None:
        elev, azim = float(_cam[0]), float(_cam[1])
    elif face in HYFACE_CORNER_CAMS:
        azim = float(HYFACE_CORNER_CAMS[face][0])
        elev = float(_HYFACE_CORNER_ELEV)
    else:
        raise RuntimeError(f"reface: unknown view '{face}' (use front/back/left/right/top/bottom or fl/fr/bl/br)")
    ref_paths = _job_ref_paths(job)
    if not ref_paths:
        raise RuntimeError("reface: no reference image provided")

    # Render the CURRENT textured mesh at the face camera (reface always runs after base texturing, so
    # the mesh already carries colours). This is a COMPLETE colour render with the EXACT geometry — the
    # geometry-locked canvas. The old path coloured a GREY geom via gpt+gemini, but the image models
    # copy the colour reference's geometry, so the result followed gpt's drifted genview, not the geom
    # (the "close but not exact" + "white where the texture should be" bugs). A full-colour base render
    # keeps its own geometry instead.
    _set(job_id, status="processing_texture", progress=70, message=f"Rendering {face} from current texture")
    base_render = worker.render_textured_view(textured_glb, elev, azim)
    base_render.save(OUTPUT_DIR / f"{job_id}_baserender_{face}.png")

    # Push the references' look onto that render while holding its geometry. On failure, fall back to
    # the raw mesh render (still geometry-correct, just the current colours).
    _set(job_id, status="processing_texture", progress=80, message=f"Restyling {face} toward references")
    from webapp.gen_transfer import restyle_to_references
    try:
        painted = restyle_to_references(base_render, ref_paths,
                                        extra_prompt=job.get("reface_extra_prompt")).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"[reface] restyle failed ({e}); baking the current mesh render as-is")
        painted = base_render.convert("RGB")
    painted.save(OUTPUT_DIR / f"{job_id}_painted_{face}.png")

    mask_img = None
    _mp = job.get("reface_mask_path")
    if _mp and os.path.exists(_mp):
        mask_img = Image.open(_mp).convert("L")

    _set(job_id, status="processing_texture", progress=88,
         message=f"Depth-aware baking {face} (foreground only)")
    textured_path = worker.reface(
        uid=job_id, textured_glb_path=textured_glb, elev=elev, azim=azim, view_image=painted,
        depth_band=_REFACE_DEPTH_BAND, mask=mask_img, mirror=AI_VIEW_MIRROR,
    )
    _set(job_id, status="completed", progress=100, message="Done",
         textured_path=textured_path, textured_url=f"/api/files/{Path(textured_path).name}")


def _run_texture(job_id):
    """Only per-face AI paint (hyface) and reface remain; other texture modes were removed."""
    mode = JOBS[job_id].get("texture_mode") or "hyface"
    if mode == "hyface":
        return _run_hyface(job_id)
    if mode == "reface":
        return _run_reface(job_id)
    raise ValueError(f"unsupported texture_mode '{mode}' (only 'hyface' and 'reface' remain)")


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
            elif kind.startswith("studio_"):
                # Any studio_* GPU kind (base/mesh/reface/face_edit/face_clear/face_render/handpaint)
                # routes to the studio dispatcher — no per-kind list to keep in sync here.
                from webapp import studio
                studio.run_gpu_job(kind, job_id)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            if kind.startswith("studio_"):
                from webapp import studio
                studio.fail_job(job_id, str(e))
            else:
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


def _blender_blend_to_glb(src_blend: str, out_glb: str):
    """Import a .blend and export it as GLB via headless Blender (adopt as a new shape base). Prefers
    the newer Blender (BLENDER_NEW_BIN) so it can read modern .blend headers; falls back to the
    default Blender when the newer one is absent."""
    bin_ = BLENDER_NEW_BIN if (shutil.which(BLENDER_NEW_BIN) or os.path.exists(BLENDER_NEW_BIN)) else BLENDER_BIN
    if not (shutil.which(bin_) or os.path.exists(bin_)):
        raise HTTPException(status_code=503, detail="Blender is not installed in this container")
    cmd = [bin_, "--background", "--python", str(BLENDER_BLEND_TO_GLB_SCRIPT), "--", src_blend, out_glb]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or "BLEND_TO_GLB_DONE" not in (proc.stdout or "") or not os.path.exists(out_glb):
        tail = (proc.stderr or proc.stdout or "")[-600:]
        hint = ""
        if "not a blend file" in tail.lower():
            hint = (f" — this .blend looks newer than the import Blender ('{bin_}'). Re-save it in a "
                    "compatible Blender version, or update the server's Blender.")
        raise HTTPException(status_code=500, detail=f"Blender .blend import failed: {tail}{hint}")


def _fill_holes_glb(glb_path: str) -> None:
    """Fill boundary holes in a GLB in place via headless Blender. Raises if Blender is missing
    or the pass fails; callers treat hole-filling as best-effort and keep the original mesh."""
    if not (shutil.which(BLENDER_BIN) or os.path.exists(BLENDER_BIN)):
        raise RuntimeError("Blender is not installed in this container (BLENDER_BIN)")
    tmp_out = f"{glb_path}.filled.glb"
    cmd = [BLENDER_BIN, "--background", "--python", str(BLENDER_FILLHOLES_SCRIPT), "--", glb_path, tmp_out]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or "FILL_HOLES_DONE" not in out or not os.path.exists(tmp_out):
        raise RuntimeError(f"Blender fill-holes failed: {out[-1500:]}")
    os.replace(tmp_out, glb_path)


def _blender_python(script_path, args, tag, expect_done):
    """Run a headless Blender python script and require its done-sentinel in the output. Returns the
    completed process. Mirrors _blender_convert/_blender_run for the small skeleton helper scripts."""
    if not (shutil.which(BLENDER_BIN) or os.path.exists(BLENDER_BIN)):
        raise RuntimeError("Blender is not installed in this container (BLENDER_BIN)")
    cmd = [BLENDER_BIN, "--background", "--python", str(script_path), "--", *[str(a) for a in args]]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or expect_done not in out:
        raise RuntimeError(f"Blender {tag} failed: {out[-2000:]}")
    return proc


def unirig_available() -> bool:
    return bool(UNIRIG_DIR and os.path.isdir(UNIRIG_DIR)
                and (shutil.which(UNIRIG_PYTHON) or os.path.exists(UNIRIG_PYTHON)))


def _unirig_run(script_rel: str, args, tag: str, timeout: int = 1800):
    """Run a UniRig launch/inference/*.sh script in its own env. Raises on failure."""
    if not unirig_available():
        raise RuntimeError("UniRig is not installed (set UNIRIG_DIR + UNIRIG_PYTHON)")
    script = os.path.join(UNIRIG_DIR, script_rel)
    if not os.path.exists(script):
        raise RuntimeError(f"UniRig script not found: {script}")
    env = dict(os.environ)
    # Make `python` inside the .sh resolve to the UniRig venv.
    venv_bin = os.path.dirname(os.path.abspath(UNIRIG_PYTHON))
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHON", UNIRIG_PYTHON)
    cmd = [UNIRIG_BASH, script, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=UNIRIG_DIR, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"UniRig {tag} failed: {out[-3000:]}")
    return proc


def _blender_run(spec, job_id, tag, expect_done="BLENDER_PROJECT_DONE"):
    if not (shutil.which(BLENDER_BIN) or os.path.exists(BLENDER_BIN)):
        raise RuntimeError("Blender is not installed in this container (BLENDER_BIN)")
    spec_path = OUTPUT_DIR / f"{job_id}_{tag}_spec.json"
    spec_path.write_text(json.dumps(spec))
    cmd = [BLENDER_BIN, "--background", "--python", str(BLENDER_PROJECT_SCRIPT), "--", str(spec_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or expect_done not in out:
        raise RuntimeError(f"Blender {tag} failed: {out[-3000:]}")
    return proc


def _blender_project_bake(job_id, shape_glb, elevations, base_glb=None):
    """Project the side elevations onto the mesh and bake to a GLB via headless Blender
    (camera-projection bake; standard conventions, uniform ortho scale). When `base_glb` is a
    Hunyuan-PBR-textured GLB, the elevations overlay its base colour (Hunyuan albedo gap-fills
    uncovered faces) and its metallic/roughness are kept. Returns the GLB path."""
    out_glb = str(OUTPUT_DIR / f"{job_id}_textured.glb")
    spec = {
        "mesh": os.path.abspath(shape_glb),
        "out": os.path.abspath(out_glb),
        "tex_size": int(os.environ.get("MVGPT_BLENDER_TEX", "2048")),
        # Min head-on dot for a face to take a cardinal elevation; faces below this (grazing, diagonal
        # corners/bevels, undersides) fall back to the Hunyuan PBR paint instead of a stretched
        # elevation. 0.5 (~60deg) keeps only reasonably head-on faces on the cardinal views now that
        # the 3/4 corner views are off, so the corners/diagonals are painted by Hunyuan.
        "face_dot": float(os.environ.get("MVGPT_FACE_DOT", "0.5")),
        "views": [{"side": s, "image": os.path.abspath(p)} for s, p in elevations.items()],
        "debug_dir": str(OUTPUT_DIR),  # writes blenderproj_cam_<side>.png for verification
    }
    if base_glb:
        spec["base_glb"] = os.path.abspath(base_glb)
    _blender_run(spec, job_id, "blenderproj", "BLENDER_PROJECT_DONE")
    if not os.path.exists(out_glb):
        raise RuntimeError("Blender projection produced no GLB")
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
    return {
        # Operational once the model has loaded at least once; a lazy reload on the next
        # job shouldn't be reported as "warming up".
        "model_ready": bool(WORKER["ready"] or WORKER["loaded_once"]),
        "model_error": WORKER["error"],
        "queue": WORK.qsize(),
        "blender": bool(shutil.which(BLENDER_BIN) or os.path.exists(BLENDER_BIN)),
        "unirig": unirig_available(),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
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
    texture_mode: str = Form("hyface"),   # per-face AI paint (only hyface + reface remain)
    ai_fill_angles: str = Form(""),       # projection: comma-sep angles to synth from front via OpenAI
    gpt_angles: str = Form("front,back,left,right,top"),  # gptproject: comma-sep canonical angles to paint
    front: UploadFile = File(None),       # hyface: explicit front-face reference (else the main image)
    back: UploadFile = File(None),
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    top: UploadFile = File(None),
    bottom: UploadFile = File(None),
    fl: UploadFile = File(None),          # hyface: 3/4 corner references (front-left/-right, back-left/-right)
    fr: UploadFile = File(None),
    bl: UploadFile = File(None),
    br: UploadFile = File(None),
    reference: list[UploadFile] = File(None),  # gpt reface: optional style reference image(s)
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

    # Per-angle photos for projection texturing (front defaults to the main image; an
    # explicit `front` upload — the hyface Front slot — overrides it). fl/fr/bl/br are
    # optional hyface 3/4-corner references.
    view_paths = {"front": source_paths[0]}
    for angle, up in (("front", front), ("back", back), ("left", left), ("right", right), ("top", top), ("bottom", bottom),
                      ("fl", fl), ("fr", fr), ("bl", bl), ("br", br)):
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
    remove_background: bool = Form(None),
    front: UploadFile = File(None),       # hyface: explicit front-face reference (else the main image)
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
    if remove_background is not None:
        job["params"]["remove_background"] = bool(remove_background)
    # Explicit front-face reference (hyface Front slot) overrides the stored front.
    if front is not None and getattr(front, "filename", ""):
        raw = await front.read()
        fpath = OUTPUT_DIR / f"{job_id}_view_front.png"
        Image.open(io.BytesIO(raw)).convert("RGBA").save(fpath)
        job["view_paths"] = {**(job.get("view_paths") or {}), "front": str(fpath)}
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
    from webapp import studio
    if job_id in studio.STUDIO_JOBS:
        return studio.public_job(job_id)
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

    no_store = {"Cache-Control": "no-store"}
    if fmt == "glb":
        return FileResponse(src, media_type="model/gltf-binary", filename=f"{job_id}.glb", headers=no_store)

    # Re-convert when the cached sibling is missing OR older than the source GLB; the source stem is
    # stable across re-textures, so a stale {id}_textured.fbx would otherwise be served forever.
    out = OUTPUT_DIR / f"{src.stem}.{fmt}"  # cached, e.g. {id}_textured.fbx
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        _blender_convert(str(src), str(out))
    return FileResponse(out, media_type="application/octet-stream", filename=f"{job_id}.{fmt}", headers=no_store)


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
    texture_mode: str = Form("hyface"),
    ai_fill_angles: str = Form(""),
    gpt_angles: str = Form("front,back,left,right,top"),
    remove_background: bool = Form(True),
    face_count: int = Form(40000),
    views: int = Form(7),
    tex_resolution: int = Form(512),
    albedo_only: bool = Form(False),
    front: UploadFile = File(None),       # hyface: explicit front-face reference (else the main image)
    back: UploadFile = File(None),
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    top: UploadFile = File(None),
    bottom: UploadFile = File(None),
    fl: UploadFile = File(None),          # hyface: 3/4 corner references
    fr: UploadFile = File(None),
    bl: UploadFile = File(None),
    br: UploadFile = File(None),
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
    for angle, up in (("front", front), ("back", back), ("left", left), ("right", right), ("top", top), ("bottom", bottom),
                      ("fl", fl), ("fr", fr), ("bl", bl), ("br", br)):
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


@app.post("/api/reface")
async def reface(
    source_id: str = Form(...),
    face: str = Form("front"),
    remove_background: bool = Form(True),
    reference: list[UploadFile] = File(None),
    reference_side: list[str] = Form(None),
    mask: UploadFile = File(None),
):
    """Depth-aware single-face re-texture of an already-textured model. Repaints only the
    nearest depth band (foreground) of `face`, leaving farther surfaces as-is."""
    if not UUID_RE.match(source_id):
        raise HTTPException(status_code=400, detail="Bad source id")
    src_tex = OUTPUT_DIR / f"{source_id}_textured.glb"
    if not src_tex.exists():
        raise HTTPException(status_code=404, detail="No textured mesh for that model (reface needs a textured model)")

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

    reference_paths = []
    reference_sides = []
    _rsides = reference_side or []
    for idx, up in enumerate(reference or []):
        if up is not None and getattr(up, "filename", ""):
            reference_paths.append(await _save(up, f"reference{idx}"))
            reference_sides.append((_rsides[idx].strip().lower() if idx < len(_rsides) and _rsides[idx] else "any"))
    if not reference_paths:
        raise HTTPException(status_code=400, detail="reface needs at least one reference image")

    mask_path = await _save(mask, "reface_mask") if (mask is not None and getattr(mask, "filename", "")) else None

    job = {
        "id": job_id, "status": "queued_texture", "progress": 60, "message": "Queued (reface)",
        "error": None, "shape_url": None, "textured_url": None, "auto_texture": False,
        "texture_mode": "reface", "reface_src_glb": str(src_tex), "reface_face": face.strip().lower(),
        "reface_mask_path": mask_path, "reference_paths": reference_paths, "reference_sides": reference_sides,
        "source_paths": [], "view_paths": {}, "created_at": time.time(),
        "params": {"remove_background": bool(remove_background)},
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


# Per-model studio router (the new /api/models/* surface) mounted before the static UI.
from webapp import studio  # noqa: E402

app.include_router(studio.router)

# Front-end mounted last so /api/* wins. Set HY3D_WEBUI_DIR to the built Next.js `out/`
# directory; falls back to the bundled static UI when unset.
_WEBUI_DIR = os.environ.get("HY3D_WEBUI_DIR")
_ui_dir = _WEBUI_DIR if (_WEBUI_DIR and os.path.isdir(_WEBUI_DIR)) else str(STATIC_DIR)
app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="static")


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
    # `python -m webapp.server` runs this file as `__main__`, a SEPARATE module object from
    # `webapp.server` (which studio.py imports). They would hold independent WORK queues / JOBS
    # dicts, so studio GPU jobs (submitted onto webapp.server.WORK) would never reach this
    # module's worker thread. Run main() from the canonical `webapp.server` module so the worker
    # thread, the queue, and the studio submit path all share one module instance.
    import webapp.server as _server
    _server.main()
