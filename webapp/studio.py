"""Per-model 3D Studio layer: registry, persistence, REST router, and the unified job store.

Mounts on the existing FastAPI app (server.py) and reuses its single GPU worker + TextureWorker.
A "model" is a durable, named aggregate that owns its 10 reference views, mesh, and texture; it
persists as a JSON sidecar so it survives a process restart. Async work runs on two lanes:
  - GPU lane: the existing single worker thread (shape + per-face paint + reface), reached by
    enqueuing onto server.WORK as ("studio_base"|"studio_reface"|"studio_face_edit", studio_job_id);
    server._worker_loop calls back into run_gpu_job().
  - Network lane: a ThreadPoolExecutor for the mesh-free gpt-image reference generation.

server is imported lazily inside functions so there is no import cycle (server imports this module).
"""
import io
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

from webapp import reference_views
from webapp.reference_views import ALL_VIEWS, VIEW_INPUTS, VIEW_TO_TAG

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("HY3D_OUTPUT_DIR", HERE / "outputs"))
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

UUID_RE = __import__("re").compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
FACE_MODES = ("paint", "reface")
FORMATS = ("glb", "fbx", "blend")
SCHEMA_VERSION = 1

# 3/4-corner azimuths + down-tilt (kept consistent with server.HYFACE_CORNER_CAMS). Hunyuan azimuth
# is left-handed (see pipeline.PROJECTION_CAMS): left/right cardinals are 270/90. Front corners
# fl=315, fr=45. Back corners bl=135, br=225 — empirically corrected: the back hemisphere reverses
# the diagonal, so the prior geometric values (bl=225/br=135) framed the wrong corner and bl/br
# rendered + baked swapped until this fix.
CORNER_AZ = {"fl": 315.0, "bl": 135.0, "br": 225.0, "fr": 45.0}
CORNER_ELEV = float(os.environ.get("HYFACE_CORNER_ELEV", "45"))

_STORE_LOCK = threading.RLock()
_NET = ThreadPoolExecutor(max_workers=int(os.environ.get("STUDIO_NET_WORKERS", "4")))

STUDIO_JOBS = {}
_JOBS_LOCK = threading.Lock()

router = APIRouter()


# --------------------------------------------------------------------------- validation / paths
def _vid(mid: str) -> str:
    if not UUID_RE.match(mid or ""):
        raise HTTPException(status_code=400, detail="Bad model id")
    return mid


def _vview(view: str) -> str:
    if view not in ALL_VIEWS:
        raise HTTPException(status_code=400, detail=f"Unknown view '{view}'")
    return view


def _vview_any(view: str) -> str:
    # Like _vview but also allows the free-camera "custom" pseudo-view (render/handpaint at an
    # arbitrary elev/azim, not tied to one of the 10 canonical faces). "custom" is a safe literal,
    # so the filename helpers (facerender_custom.png / handpaint_custom.png) stay inside the model dir.
    if view == "custom" or view in ALL_VIEWS:
        return view
    raise HTTPException(status_code=400, detail=f"Unknown view '{view}'")


def _vangles(elev, azim):
    # Validate a custom camera. elev clamped to [-90, 90]; azim wrapped to [0, 360).
    try:
        e, a = float(elev), float(azim)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="A custom view requires numeric elev and azim")
    if not (-90.0 <= e <= 90.0):
        raise HTTPException(status_code=400, detail="elev must be between -90 and 90")
    return e, a % 360.0


def _assert_within(path: Path, root: Path) -> Path:
    rp = path.resolve()
    if not str(rp).startswith(str(root.resolve())):
        raise HTTPException(status_code=400, detail="Path outside the output root")
    return rp


def _model_dir(mid: str) -> Path:
    return _assert_within(MODELS_DIR / mid, MODELS_DIR)


def _json_path(mid: str) -> Path:
    return _model_dir(mid) / "model.json"


def _seed_file(mid: str) -> Path:
    return _model_dir(mid) / "seed.png"


def _ref_file(mid: str, view: str) -> Path:
    return _model_dir(mid) / f"ref_{view}.png"


def _shape_glb(mid: str) -> Path:
    return _assert_within(OUTPUT_DIR / f"{mid}_shape.glb", OUTPUT_DIR)


def _textured_glb(mid: str) -> Path:
    return _assert_within(OUTPUT_DIR / f"{mid}_textured.glb", OUTPUT_DIR)


def _facerender_file(mid: str, view: str) -> Path:
    return _model_dir(mid) / f"facerender_{view}.png"


def _now_ms() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------------------- texture history
def _history_dir(mid: str) -> Path:
    d = _model_dir(mid) / "texture_history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshot_glb(mid: str, seq: int) -> Path:
    return _assert_within(_history_dir(mid) / f"{int(seq)}.glb", _model_dir(mid))


def _push_snapshot(mid: str, label: str) -> None:
    """Snapshot the CURRENT textured GLB into the history stack (called after a completed texture op).
    No-op if there is no current texture. Records the per-face state + stage so a restore can rebuild
    the model row exactly."""
    cur = _textured_glb(mid)
    if not cur.exists():
        return
    with _STORE_LOCK:
        data = _load(mid)
        seq = int(data.get("textureSeq", 0))
        shutil.copy2(cur, _snapshot_glb(mid, seq))
        data.setdefault("textureHistory", []).append({
            "seq": seq, "label": label, "ts": _now_ms(),
            "faces": {v: dict(data["faces"][v]) for v in ALL_VIEWS},
            "stage": data["textureStage"],
        })
        data["textureSeq"] = seq + 1
        _save(mid, data)


def _clear_history(mid: str) -> None:
    """Drop every snapshot (used when the mesh is regenerated — old snapshots belong to a dead UV)."""
    with _STORE_LOCK:
        data = _load(mid)
        try:
            shutil.rmtree(_history_dir(mid))
        except FileNotFoundError:
            pass
        data["textureHistory"] = []
        data["textureSeq"] = 0
        _save(mid, data)


def _base_snapshot_seq(data: dict):
    """Seq of the base snapshot to revert a single face toward: the earliest recorded snapshot
    (base texturing is always the first texture op). Returns None if there is no history."""
    hist = data.get("textureHistory") or []
    return min((e["seq"] for e in hist), default=None)


# --------------------------------------------------------------------------- persistence
def _default_data(mid: str, name: str) -> dict:
    return {
        "id": mid,
        "name": name or "Untitled model",
        "schemaVersion": SCHEMA_VERSION,
        "references": {v: {"status": "empty", "source": None, "editPrompt": None} for v in ALL_VIEWS},
        "faces": {v: {"status": "pending", "mode": None} for v in ALL_VIEWS},
        "textureStage": "none",
        "meshConfig": None,
        "meshSourceView": None,
        # Texture undo history: each completed base/reface/paint snapshots the whole-mesh texture GLB
        # into texture_history/{seq}.glb plus a metadata entry here, so any prior step is restorable.
        "textureHistory": [],
        "textureSeq": 0,
        "createdAt": _now_ms(),
        "updatedAt": _now_ms(),
    }


def _load(mid: str) -> dict:
    p = _json_path(mid)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    with _STORE_LOCK:
        return json.loads(p.read_text(encoding="utf-8"))


def _save(mid: str, data: dict) -> None:
    data["updatedAt"] = _now_ms()
    p = _json_path(mid)
    tmp = p.with_suffix(".json.tmp")
    with _STORE_LOCK:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, p)


def _update(mid: str, mutate):
    """Atomic read-modify-write of model.json under the reentrant store lock so concurrent
    reference generations / edits on the same model can't lose updates. `mutate(data)` may raise
    HTTPException to reject; the lock is released either way."""
    with _STORE_LOCK:
        data = _load(mid)
        mutate(data)
        _save(mid, data)
        return data


def _ref_url(mid, view, ver):
    """Reference image URL with a cache-busting `?v=` so a replaced/edited file (the path is reused)
    gets a NEW URL and the browser refetches it instead of showing the cached image. `ver` is the
    model's `updatedAt` (bumped on every write) — robust regardless of filesystem mtime resolution."""
    return f"/api/models/{mid}/references/{view}/image?v={ver}" if _ref_file(mid, view).exists() else None


# Faces/vertices per GLB version — computed once with trimesh, reused while the file is unchanged so
# assemble_model (called on every poll) never reloads the mesh.
_MESH_STATS_CACHE: dict = {}


def _mesh_stats(mid: str):
    """Face + vertex count of the current GLB (textured preferred, else the untextured shape). Cached
    by (path, mtime, size). Returns None when there's no mesh or it can't be read."""
    glb = _textured_glb(mid)
    if not glb.exists():
        glb = _shape_glb(mid)
    if not glb.exists():
        return None
    try:
        st = glb.stat()
        sig = (str(glb), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    cached = _MESH_STATS_CACHE.get(mid)
    if cached and cached[0] == sig:
        return cached[1]
    try:
        import trimesh
        mesh = trimesh.load(str(glb), force="mesh")
        stats = {"faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))}
    except Exception:  # noqa: BLE001 — a missing/invalid mesh just yields no count
        return None
    _MESH_STATS_CACHE[mid] = (sig, stats)
    return stats


def assemble_model(mid: str) -> dict:
    data = _load(mid)
    ver = data["updatedAt"]
    refs = {}
    for v in ALL_VIEWS:
        rd = data["references"][v]
        refs[v] = {"view": v, "url": _ref_url(mid, v, ver), "status": rd["status"],
                   "source": rd["source"], "editPrompt": rd.get("editPrompt")}
    faces = {v: {"view": v, "status": data["faces"][v]["status"], "mode": data["faces"][v]["mode"]}
             for v in ALL_VIEWS}
    return {
        "id": mid,
        "name": data["name"],
        "seedImageUrl": (f"/api/models/{mid}/seed?v={ver}" if _seed_file(mid).exists() else None),
        "references": refs,
        "faces": faces,
        "meshUrl": (f"/api/files/{mid}_shape.glb?v={ver}" if _shape_glb(mid).exists() else None),
        "texturedUrl": (f"/api/files/{mid}_textured.glb?v={ver}" if _textured_glb(mid).exists() else None),
        "meshStats": _mesh_stats(mid),
        "textureStage": data["textureStage"],
        "meshSourceView": data.get("meshSourceView"),
        "textureHistory": [{"seq": e["seq"], "label": e["label"], "ts": e["ts"], "stage": e["stage"]}
                           for e in data.get("textureHistory", [])],
        "createdAt": data["createdAt"],
        "updatedAt": data["updatedAt"],
    }


def _summary(mid: str) -> dict:
    data = _load(mid)
    textured = data["textureStage"] == "complete"
    preview = None if textured else _ref_url(mid, "front", data["updatedAt"])
    return {"id": mid, "name": data["name"], "previewUrl": preview,
            "textured": textured, "updatedAt": data["updatedAt"]}


def _save_upload_png(raw: bytes, dest: Path) -> None:
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(raw)).convert("RGBA").save(dest)


# --------------------------------------------------------------------------- job store
def new_job(label: str, model_id: str) -> str:
    sjid = str(uuid.uuid4())
    with _JOBS_LOCK:
        STUDIO_JOBS[sjid] = {"id": sjid, "status": "processing", "progress": 5, "label": label,
                             "error": None, "model": None, "modelId": model_id}
    return sjid


def set_job(sjid: str, **kw) -> None:
    with _JOBS_LOCK:
        j = STUDIO_JOBS.get(sjid)
        if j:
            j.update(kw)


def complete_job(sjid: str, mid: str) -> None:
    set_job(sjid, status="completed", progress=100, model=assemble_model(mid), _proxy=None)


def fail_job(sjid: str, error: str) -> None:
    set_job(sjid, status="failed", error=error, _proxy=None)


def public_job(sjid: str) -> dict:
    j = STUDIO_JOBS.get(sjid)
    if not j:
        raise HTTPException(status_code=404, detail="Unknown job")
    progress = j["progress"]
    proxy = j.get("_proxy")
    if j["status"] == "processing" and proxy:
        try:
            from webapp import server
            pj = server.JOBS.get(proxy)
            if pj and isinstance(pj.get("progress"), (int, float)):
                progress = max(progress, int(pj["progress"]))
        except Exception:  # noqa: BLE001
            pass
    return {"id": j["id"], "status": j["status"], "progress": progress, "label": j.get("label"),
            "error": j.get("error"), "model": j.get("model"), "modelId": j.get("modelId")}


def submit_gpu(kind: str, sjid: str) -> None:
    from webapp import server
    server.WORK.put((kind, sjid))


# --------------------------------------------------------------------------- GPU lane handlers
def _cam_for(tag: str):
    from webapp.pipeline import TextureWorker
    pc = TextureWorker.PROJECTION_CAMS.get(tag)
    if pc is not None:
        return float(pc[0]), float(pc[1])
    if tag in CORNER_AZ:
        return CORNER_ELEV, CORNER_AZ[tag]
    raise RuntimeError(f"unknown view tag '{tag}'")


def run_gpu_job(kind: str, sjid: str) -> None:
    """Dispatched from server._worker_loop. Never raises — failures land on the StudioJob."""
    j = STUDIO_JOBS.get(sjid)
    if not j:
        return
    mid = j["modelId"]
    try:
        if kind == "studio_base":
            _gpu_base(sjid, mid, j["_cfg"])
        elif kind == "studio_mesh":
            _gpu_mesh(sjid, mid, j["_cfg"])
        elif kind == "studio_reface":
            _gpu_reface(sjid, mid, j["_view"], j.get("_edit"), face_mode="reface")
        elif kind == "studio_face_edit":
            if j["_mode"] == "paint":
                _gpu_paint_face(sjid, mid, j["_view"], j.get("_edit"), j.get("_image"))
            else:
                _gpu_reface(sjid, mid, j["_view"], j.get("_edit"), face_mode="reface",
                            ref_override=j.get("_image"))
        elif kind == "studio_face_clear":
            _gpu_face_clear(sjid, mid, j["_view"], j["_clear_seq"])
        elif kind == "studio_face_render":
            _gpu_face_render(sjid, mid, j["_view"], j.get("_elev"), j.get("_azim"))
        elif kind == "studio_handpaint":
            _gpu_handpaint(sjid, mid, j["_view"], j.get("_image"), j.get("_elev"), j.get("_azim"))
        complete_job(sjid, mid)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        # Best-effort revert so a failed job never leaves the model stuck mid-state.
        try:
            data = _load(mid)
            if kind == "studio_base" and data.get("textureStage") == "base-running":
                data["textureStage"] = "none"
                _save(mid, data)
            else:
                v = j.get("_view")
                if v and data["faces"].get(v, {}).get("status") == "texturing":
                    data["faces"][v]["status"] = "done" if _textured_glb(mid).exists() else "pending"
                    _save(mid, data)
        except Exception:  # noqa: BLE001
            pass
        fail_job(sjid, str(e))


def _gpu_base(sjid: str, mid: str, cfg) -> None:
    from webapp import server
    data = _load(mid)
    if any(data["references"][v]["status"] != "approved" for v in ALL_VIEWS):
        raise RuntimeError("all 10 references must be approved before texturing")
    front_f = _ref_file(mid, "front")
    seed_f = _seed_file(mid)
    if not front_f.exists():
        raise RuntimeError("missing front reference")
    set_job(sjid, status="processing", progress=15, _proxy=mid)

    # Reuse a mesh already generated via /mesh; otherwise generate one from the front (back-compat).
    shape_glb = _shape_glb(mid)
    if shape_glb.exists():
        shape_path, processed = str(shape_glb), str(front_f)
    else:
        worker = server._ensure_model()
        shape_path, processed = worker.generate_shape(
            uid=mid, image=Image.open(front_f).convert("RGBA"), remove_background=True,
            steps=int(cfg.inference_steps), guidance_scale=float(cfg.guidance_scale),
            seed=int(cfg.seed), octree_resolution=int(cfg.octree_resolution),
            num_chunks=8000, face_count=int(cfg.mesh_faces),
        )

    view_paths = {VIEW_TO_TAG[v]: str(_ref_file(mid, v)) for v in ALL_VIEWS if _ref_file(mid, v).exists()}
    src0 = str(seed_f) if seed_f.exists() else str(front_f)
    server.JOBS[mid] = {
        "id": mid, "status": "processing_texture", "progress": 60, "message": "", "error": None,
        "shape_url": None, "textured_url": None, "auto_texture": False, "texture_mode": "hyface",
        "ai_fill_angles": [], "gpt_angles": [], "mv_viewset": "canonical",
        "reference_paths": [], "reference_sides": [], "num_images": 1,
        "source_paths": [src0], "view_paths": view_paths, "shape_path": shape_path,
        "processed_image_path": processed, "created_at": time.time(),
        "params": {"remove_background": True, "steps": int(cfg.inference_steps),
                   "guidance_scale": float(cfg.guidance_scale), "seed": int(cfg.seed),
                   "octree_resolution": int(cfg.octree_resolution), "num_chunks": 8000,
                   "face_count": int(cfg.mesh_faces), "views": int(cfg.texture_views),
                   "tex_resolution": 512, "albedo_only": True},
    }
    try:
        server._run_hyface(mid)
    finally:
        server.JOBS.pop(mid, None)

    data = _load(mid)
    data["faces"] = {v: {"status": "done", "mode": "paint"} for v in ALL_VIEWS}
    data["textureStage"] = "complete"
    data["meshConfig"] = {"inferenceSteps": int(cfg.inference_steps),
                          "guidanceScale": float(cfg.guidance_scale),
                          "octreeResolution": int(cfg.octree_resolution),
                          "textureViews": int(cfg.texture_views), "seed": int(cfg.seed),
                          "meshFaces": int(cfg.mesh_faces)}
    _save(mid, data)
    _push_snapshot(mid, "Base texture")


def _gpu_mesh(sjid: str, mid: str, cfg) -> None:
    """Generate ONLY the mesh from a chosen reference view, so the user can preview and iterate on the
    shape before texturing. Regenerating the mesh resets any existing texture."""
    from webapp import server
    view = cfg.source_view
    ref_f = _ref_file(mid, view)
    if not ref_f.exists():
        raise RuntimeError(f"the '{view}' reference has no image")
    set_job(sjid, status="processing", progress=20)
    worker = server._ensure_model()
    worker.generate_shape(
        uid=mid, image=Image.open(ref_f).convert("RGBA"), remove_background=True,
        steps=int(cfg.inference_steps), guidance_scale=float(cfg.guidance_scale),
        seed=int(cfg.seed), octree_resolution=int(cfg.octree_resolution),
        num_chunks=8000, face_count=int(cfg.mesh_faces),
    )
    # a new mesh invalidates any existing texture
    try:
        _textured_glb(mid).unlink()
    except FileNotFoundError:
        pass
    data = _load(mid)
    data["faces"] = {v: {"status": "pending", "mode": None} for v in ALL_VIEWS}
    data["textureStage"] = "none"
    data["meshSourceView"] = view
    data["meshConfig"] = {"inferenceSteps": int(cfg.inference_steps),
                          "guidanceScale": float(cfg.guidance_scale),
                          "octreeResolution": int(cfg.octree_resolution),
                          "seed": int(cfg.seed), "meshFaces": int(cfg.mesh_faces), "sourceView": view}
    _save(mid, data)
    _clear_history(mid)  # snapshots belong to the previous mesh's UV; drop them


def _gpu_reface(sjid: str, mid: str, view: str, edit, face_mode="reface", ref_override=None) -> None:
    from webapp import server
    tag = VIEW_TO_TAG[view]
    if not _textured_glb(mid).exists():
        raise RuntimeError("model has no textured mesh; run base texturing first")
    ref = ref_override if (ref_override and os.path.exists(ref_override)) else str(_ref_file(mid, view))
    if not (ref and os.path.exists(ref)):
        raise RuntimeError("no reference image for this view")
    set_job(sjid, status="processing", progress=30, _proxy=mid)
    server.JOBS[mid] = {
        "id": mid, "status": "queued_texture", "progress": 60, "message": "", "error": None,
        "shape_url": None, "textured_url": None, "auto_texture": False, "texture_mode": "reface",
        "reface_src_glb": str(_textured_glb(mid)), "reface_face": tag, "reface_mask_path": None,
        "reface_extra_prompt": (edit or None), "reference_paths": [ref], "reference_sides": [tag],
        "source_paths": [], "view_paths": {}, "created_at": time.time(),
        "params": {"remove_background": True},
    }
    try:
        server._run_reface(mid)
    finally:
        server.JOBS.pop(mid, None)
    data = _load(mid)
    data["faces"][view] = {"status": "done", "mode": "reface"}
    data["textureStage"] = "complete"  # a per-face edit only runs on a textured model
    _save(mid, data)
    _push_snapshot(mid, f"Reface {view}")


def _gpu_paint_face(sjid: str, mid: str, view: str, edit, ref_override) -> None:
    """Localized single-face per-face AI paint composited onto the base (no full re-bake)."""
    from webapp import server
    tag = VIEW_TO_TAG[view]
    if not (_textured_glb(mid).exists() and _shape_glb(mid).exists()):
        raise RuntimeError("model has no mesh/texture; run base texturing first")
    elev, azim = _cam_for(tag)
    set_job(sjid, status="processing", progress=25)

    ref_path = ref_override if (ref_override and os.path.exists(ref_override)) else str(_ref_file(mid, view))
    if not (ref_path and os.path.exists(ref_path)):
        raise RuntimeError("no reference image for this view")
    ref_img = Image.open(ref_path).convert("RGBA")
    if edit and edit.strip():
        try:
            from webapp.image_edit import CARTOON_STYLE, edit_image
            ref_img = edit_image(
                [ref_img.convert("RGB")],
                f"Apply this adjustment to the image while keeping the same object, view, "
                f"orientation and framing: {edit.strip()}. " + CARTOON_STYLE,
                size=(1024, 1024),
            ).convert("RGBA")
        except Exception as e:  # noqa: BLE001
            print(f"[studio] paint tweak skipped ({e})")

    worker = server._ensure_model()
    painted = worker.paint_single_view(str(_shape_glb(mid)), ref_img, elev, azim,
                                       tex_resolution=512)
    set_job(sjid, status="processing", progress=70)
    worker.reface(uid=mid, textured_glb_path=str(_textured_glb(mid)), elev=elev, azim=azim,
                  view_image=painted.convert("RGB"), depth_band=1.0, mask=None,
                  mirror=server.AI_VIEW_MIRROR)
    data = _load(mid)
    data["faces"][view] = {"status": "done", "mode": "paint"}
    data["textureStage"] = "complete"  # a per-face edit only runs on a textured model
    _save(mid, data)
    _push_snapshot(mid, f"Paint {view}")


def _gpu_face_clear(sjid: str, mid: str, view: str, seq: int) -> None:
    """Revert ONE face to a prior snapshot (default: the base) by re-baking that face's region from
    the snapshot GLB onto the current texture. Other faces stay as they are."""
    from webapp import server
    tag = VIEW_TO_TAG[view]
    if not _textured_glb(mid).exists():
        raise RuntimeError("model has no textured mesh")
    snap = _snapshot_glb(mid, seq)
    if not snap.exists():
        raise RuntimeError("the snapshot to revert from is missing")
    elev, azim = _cam_for(tag)
    set_job(sjid, status="processing", progress=40)
    worker = server._ensure_model()
    src_render = worker.render_textured_view(str(snap), elev, azim)
    set_job(sjid, status="processing", progress=70)
    worker.reface(uid=mid, textured_glb_path=str(_textured_glb(mid)), elev=elev, azim=azim,
                  view_image=src_render.convert("RGB"), depth_band=1.0, mask=None, mirror=False)
    data = _load(mid)
    snap_entry = next((e for e in data.get("textureHistory", []) if e["seq"] == seq), None)
    data["faces"][view] = dict(snap_entry["faces"][view]) if snap_entry else {"status": "done", "mode": "paint"}
    data["textureStage"] = "complete"
    _save(mid, data)
    _push_snapshot(mid, f"Clear {view}")


def _gpu_face_render(sjid: str, mid: str, view: str, elev=None, azim=None) -> None:
    """Render the CURRENT textured face at its camera into facerender_{view}.png — the backdrop the
    hand-paint canvas draws on. For a canonical view it's cached (reused while newer than the textured
    GLB). For view="custom" the (elev, azim) vary per request, so it always re-renders."""
    from webapp import server
    glb = _textured_glb(mid)
    if not glb.exists():
        raise RuntimeError("model has no textured mesh")
    out = _facerender_file(mid, view)
    if view == "custom":
        elev, azim = float(elev), float(azim)  # always re-render — the camera changes each request
    else:
        if out.exists() and out.stat().st_mtime >= glb.stat().st_mtime:
            return  # backdrop already current
        elev, azim = _cam_for(VIEW_TO_TAG[view])
    set_job(sjid, status="processing", progress=40)
    worker = server._ensure_model()
    img = worker.render_textured_view(str(glb), elev, azim)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out)


def _gpu_handpaint(sjid: str, mid: str, view: str, overlay_path: str, elev=None, azim=None) -> None:
    """Bake a hand-painted overlay (RGBA, alpha = strokes) onto the mesh at the face's camera. For
    view="custom" the camera is the supplied (elev, azim) and no face slot is updated."""
    from webapp import server
    if not _textured_glb(mid).exists():
        raise RuntimeError("model has no textured mesh")
    if not (overlay_path and os.path.exists(overlay_path)):
        raise RuntimeError("no painted overlay provided")
    if view == "custom":
        elev, azim = float(elev), float(azim)
    else:
        elev, azim = _cam_for(VIEW_TO_TAG[view])
    set_job(sjid, status="processing", progress=40)
    worker = server._ensure_model()
    overlay = Image.open(overlay_path).convert("RGBA")
    set_job(sjid, status="processing", progress=70)
    worker.paint_overlay(uid=mid, textured_glb_path=str(_textured_glb(mid)), elev=elev, azim=azim,
                         overlay=overlay)
    data = _load(mid)
    if view != "custom":
        data["faces"][view] = {"status": "done", "mode": "paint"}
    data["textureStage"] = "complete"
    _save(mid, data)
    label = f"Hand paint custom ({int(round(elev))}°/{int(round(azim))}°)" if view == "custom" else f"Hand paint {view}"
    _push_snapshot(mid, label)
    try:  # face changed -> drop the stale backdrop so the next open re-renders
        _facerender_file(mid, view).unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------- request models
class RenameIn(BaseModel):
    name: str


class GenerateIn(BaseModel):
    edit_prompt: str | None = None


class MeshConfigIn(BaseModel):
    inference_steps: int = 30
    guidance_scale: float = 7.5
    octree_resolution: int = 256
    texture_views: int = 6
    seed: int = 0
    mesh_faces: int = 40000


class MeshGenIn(BaseModel):
    source_view: str = "front"          # which reference drives the 3D shape
    inference_steps: int = 30
    guidance_scale: float = 7.5
    octree_resolution: int = 256
    seed: int = 0
    mesh_faces: int = 40000


# --------------------------------------------------------------------------- routes: models
@router.get("/api/models")
def list_models():
    out = []
    for p in MODELS_DIR.glob("*/model.json"):
        mid = p.parent.name
        if UUID_RE.match(mid):
            try:
                out.append(_summary(mid))
            except Exception:  # noqa: BLE001
                pass
    out.sort(key=lambda m: m["updatedAt"], reverse=True)
    return out


@router.post("/api/models")
async def create_model(name: str = Form("Untitled model"), seed_image: UploadFile = File(None)):
    mid = str(uuid.uuid4())
    _model_dir(mid).mkdir(parents=True, exist_ok=True)
    if seed_image is not None and getattr(seed_image, "filename", ""):
        _save_upload_png(await seed_image.read(), _seed_file(mid))
    _save(mid, _default_data(mid, name))
    return assemble_model(mid)


@router.get("/api/models/{mid}")
def get_model(mid: str):
    return assemble_model(_vid(mid))


@router.patch("/api/models/{mid}")
def rename_model(mid: str, body: RenameIn):
    data = _load(_vid(mid))
    data["name"] = body.name or data["name"]
    _save(mid, data)
    return assemble_model(mid)


@router.delete("/api/models/{mid}")
def delete_model(mid: str):
    _vid(mid)
    d = _model_dir(mid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    for suffix in ("_shape.glb", "_textured.glb"):
        f = OUTPUT_DIR / f"{mid}{suffix}"
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    return Response(status_code=204)


# --------------------------------------------------------------------------- routes: references
@router.post("/api/models/{mid}/references/{view}/generate")
def generate_reference(mid: str, view: str, body: GenerateIn = GenerateIn()):
    _vid(mid); _vview(view)
    if not _seed_file(mid).exists():
        raise HTTPException(status_code=400, detail="Upload a seed image first")

    def _mut(data):
        if data["references"][view]["status"] == "generating":
            raise HTTPException(status_code=409, detail="Already generating this view")
        for dep in VIEW_INPUTS[view]:
            if data["references"][dep]["status"] != "approved":
                raise HTTPException(status_code=409, detail=f"Approve '{dep}' first")
        data["references"][view] = {"status": "generating",
                                    "source": data["references"][view]["source"],
                                    "editPrompt": (body.edit_prompt or None)}

    _update(mid, _mut)
    sjid = new_job(f"Generating {view}", mid)
    _NET.submit(_net_generate, sjid, mid, view, body.edit_prompt)
    return public_job(sjid)


def _net_generate(sjid: str, mid: str, view: str, edit) -> None:
    try:
        seed = str(_seed_file(mid)) if _seed_file(mid).exists() else None
        dep_paths = [str(_ref_file(mid, d)) for d in VIEW_INPUTS[view]]
        img = reference_views.generate_view(view, seed, dep_paths, edit)
        img.save(_ref_file(mid, view))

        def _ok(data):
            data["references"][view] = {"status": "pending", "source": "generated",
                                        "editPrompt": (edit or None)}
        _update(mid, _ok)
        complete_job(sjid, mid)
    except Exception as e:  # noqa: BLE001
        try:
            def _revert(data):
                data["references"][view]["status"] = "pending" if _ref_file(mid, view).exists() else "empty"
            _update(mid, _revert)
        except Exception:  # noqa: BLE001
            pass
        fail_job(sjid, str(e))


@router.post("/api/models/{mid}/references/{view}/edit")
async def edit_reference_masked(mid: str, view: str, edit_prompt: str = Form(None),
                                mask: UploadFile = File(...)):
    """Inpaint ONLY the brushed region of an existing reference view (mask + prompt). Network lane."""
    _vid(mid); _vview(view)
    if not _ref_file(mid, view).exists():
        raise HTTPException(status_code=400, detail="This view has no image to edit")
    mask_path = str(_model_dir(mid) / f"edit_mask_{view}.png")
    _save_upload_png(await mask.read(), Path(mask_path))

    def _mut(data):
        rd = data["references"][view]
        if rd["status"] == "generating":
            raise HTTPException(status_code=409, detail="Already editing this view")
        data["references"][view] = {"status": "generating", "source": rd["source"],
                                    "editPrompt": (edit_prompt or None)}

    _update(mid, _mut)
    sjid = new_job(f"Editing {view}", mid)
    _NET.submit(_net_masked_edit, sjid, mid, view, edit_prompt, mask_path)
    return public_job(sjid)


def _net_masked_edit(sjid: str, mid: str, view: str, edit, mask_path: str) -> None:
    try:
        out = reference_views.edit_view_masked(str(_ref_file(mid, view)), mask_path, edit)
        out.save(_ref_file(mid, view))

        def _ok(data):
            data["references"][view] = {"status": "pending", "source": "generated",
                                        "editPrompt": (edit or None)}
        _update(mid, _ok)
        complete_job(sjid, mid)
    except Exception as e:  # noqa: BLE001
        try:
            def _revert(data):
                data["references"][view]["status"] = "pending" if _ref_file(mid, view).exists() else "empty"
            _update(mid, _revert)
        except Exception:  # noqa: BLE001
            pass
        fail_job(sjid, str(e))


@router.post("/api/models/{mid}/references/{view}/upload")
async def upload_reference(mid: str, view: str, image: UploadFile = File(...)):
    _vid(mid); _vview(view)
    _save_upload_png(await image.read(), _ref_file(mid, view))

    def _mut(data):
        data["references"][view] = {"status": "approved", "source": "uploaded", "editPrompt": None}
    _update(mid, _mut)
    return assemble_model(mid)


@router.post("/api/models/{mid}/references/{view}/approve")
def approve_reference(mid: str, view: str):
    _vid(mid); _vview(view)

    def _mut(data):
        rd = data["references"][view]
        if not _ref_file(mid, view).exists() or rd["status"] in ("empty", "generating"):
            raise HTTPException(status_code=400, detail="Nothing to approve for this view")
        rd["status"] = "approved"
    _update(mid, _mut)
    return assemble_model(mid)


@router.get("/api/models/{mid}/references/{view}/image")
def reference_image(mid: str, view: str):
    _vid(mid); _vview(view)
    f = _ref_file(mid, view)
    if not f.exists():
        raise HTTPException(status_code=404, detail="No image for this view")
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/api/models/{mid}/seed")
def seed_image(mid: str):
    _vid(mid)
    f = _seed_file(mid)
    if not f.exists():
        raise HTTPException(status_code=404, detail="No seed image")
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/api/models/{mid}/seed")
async def replace_seed(mid: str, image: UploadFile = File(...)):
    """Replace the model's seed image (the generation input) — distinct from the front reference."""
    _vid(mid)
    raw = await image.read()
    _load(mid)  # 404 if the model is missing
    _save_upload_png(raw, _seed_file(mid))
    _update(mid, lambda d: None)  # bump updatedAt so the seed URL cache-busts
    return assemble_model(mid)


# --------------------------------------------------------------------------- routes: texture
@router.post("/api/models/{mid}/mesh")
def generate_mesh(mid: str, cfg: MeshGenIn = MeshGenIn()):
    """Generate (or regenerate) ONLY the mesh from the chosen reference view. Resets any texture."""
    _vid(mid)
    _vview(cfg.source_view)
    if not _ref_file(mid, cfg.source_view).exists():
        raise HTTPException(status_code=400, detail=f"The '{cfg.source_view}' reference has no image")
    sjid = new_job(f"Generating mesh from {cfg.source_view}", mid)
    set_job(sjid, _cfg=cfg)
    submit_gpu("studio_mesh", sjid)
    return public_job(sjid)


@router.post("/api/models/{mid}/texture/base")
def texture_base(mid: str, cfg: MeshConfigIn = MeshConfigIn()):
    _vid(mid)
    data = _load(mid)
    if any(data["references"][v]["status"] != "approved" for v in ALL_VIEWS):
        raise HTTPException(status_code=400, detail="Approve all 10 references before texturing")
    sjid = new_job("Per-face AI paint", mid)
    set_job(sjid, _cfg=cfg)
    data["textureStage"] = "base-running"
    _save(mid, data)
    submit_gpu("studio_base", sjid)
    return public_job(sjid)


@router.post("/api/models/{mid}/texture/reface/{view}")
def reface_view(mid: str, view: str, body: GenerateIn = GenerateIn()):
    _vid(mid); _vview(view)
    if not _textured_glb(mid).exists():
        raise HTTPException(status_code=400, detail="Run base texturing first")
    data = _load(mid)
    if not _ref_file(mid, view).exists():
        raise HTTPException(status_code=400, detail="This view has no reference image")
    sjid = new_job(f"Refacing {view}", mid)
    set_job(sjid, _view=view, _edit=body.edit_prompt)
    data["faces"][view]["status"] = "texturing"
    _save(mid, data)
    submit_gpu("studio_reface", sjid)
    return public_job(sjid)


@router.post("/api/models/{mid}/faces/{view}/edit")
async def edit_face(mid: str, view: str, mode: str = Form(...), edit_prompt: str = Form(None),
                    image: UploadFile = File(None)):
    _vid(mid); _vview(view)
    if mode not in FACE_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'")
    if not _textured_glb(mid).exists():
        raise HTTPException(status_code=400, detail="Run base texturing first")
    img_path = None
    if image is not None and getattr(image, "filename", ""):
        img_path = str(_model_dir(mid) / f"edit_{view}.png")
        _save_upload_png(await image.read(), Path(img_path))
    data = _load(mid)
    sjid = new_job(f"Editing {view}", mid)
    set_job(sjid, _view=view, _mode=mode, _edit=edit_prompt, _image=img_path)
    data["faces"][view]["status"] = "texturing"
    _save(mid, data)
    submit_gpu("studio_face_edit", sjid)
    return public_job(sjid)


# --------------------------------------------------------------------------- routes: texture history
@router.post("/api/models/{mid}/texture/restore/{seq}")
def restore_texture(mid: str, seq: int):
    """Roll the whole-mesh texture back to a prior snapshot. Instant (file copy) — re-texture forward
    from here. Later snapshots are kept, so you can jump around the timeline."""
    _vid(mid)
    snap = _snapshot_glb(mid, seq)
    with _STORE_LOCK:
        data = _load(mid)
        entry = next((e for e in data.get("textureHistory", []) if e["seq"] == seq), None)
        if entry is None or not snap.exists():
            raise HTTPException(status_code=404, detail="Snapshot not found")
        shutil.copy2(snap, _textured_glb(mid))
        data["faces"] = {v: dict(entry["faces"][v]) for v in ALL_VIEWS}
        data["textureStage"] = entry["stage"]
        _save(mid, data)
    return assemble_model(mid)


@router.post("/api/models/{mid}/texture/reset")
def reset_texture(mid: str):
    """Delete the current texture back to the untextured mesh. History snapshots are kept (restorable)."""
    _vid(mid)
    with _STORE_LOCK:
        data = _load(mid)
        try:
            _textured_glb(mid).unlink()
        except FileNotFoundError:
            pass
        data["faces"] = {v: {"status": "pending", "mode": None} for v in ALL_VIEWS}
        data["textureStage"] = "none"
        _save(mid, data)
    return assemble_model(mid)


@router.post("/api/models/{mid}/faces/{view}/clear")
def clear_face(mid: str, view: str):
    """Revert ONE face back to the base texture (drops refaces/paints on that face). GPU re-bake."""
    _vid(mid); _vview(view)
    data = _load(mid)
    if not _textured_glb(mid).exists():
        raise HTTPException(status_code=400, detail="No texture to clear")
    base_seq = _base_snapshot_seq(data)
    if base_seq is None:
        raise HTTPException(status_code=400, detail="No base snapshot to revert this face to")
    sjid = new_job(f"Clearing {view}", mid)
    set_job(sjid, _view=view, _clear_seq=base_seq)
    data["faces"][view]["status"] = "texturing"
    _save(mid, data)
    submit_gpu("studio_face_clear", sjid)
    return public_job(sjid)


# --------------------------------------------------------------------------- routes: hand paint
@router.post("/api/models/{mid}/faces/{view}/render")
def render_face(mid: str, view: str, elev: float | None = None, azim: float | None = None):
    """Render the current textured face into a PNG backdrop for the hand-paint canvas. Poll the job;
    on completion fetch GET faces/{view}/render-image. For view="custom" pass elev+azim query params
    to render an arbitrary camera."""
    _vid(mid); _vview_any(view)
    if not _textured_glb(mid).exists():
        raise HTTPException(status_code=400, detail="Run base texturing first")
    sjid = new_job(f"Rendering {view}", mid)
    if view == "custom":
        e, a = _vangles(elev, azim)
        set_job(sjid, _view=view, _elev=e, _azim=a)
    else:
        set_job(sjid, _view=view)
    submit_gpu("studio_face_render", sjid)
    return public_job(sjid)


@router.get("/api/models/{mid}/faces/{view}/render-image")
def face_render_image(mid: str, view: str):
    _vid(mid); _vview_any(view)
    f = _facerender_file(mid, view)
    if not f.exists():
        raise HTTPException(status_code=404, detail="No render for this view")
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/api/models/{mid}/faces/{view}/handpaint")
async def handpaint_face(mid: str, view: str, overlay: UploadFile = File(...),
                         elev: float | None = Form(None), azim: float | None = Form(None)):
    """Bake a hand-painted overlay (transparent except the brushed strokes) onto this face.  → Job
    For view="custom" pass elev+azim form fields to bake at an arbitrary camera (no face slot is
    touched; a history snapshot is still pushed)."""
    _vid(mid); _vview_any(view)
    if not _textured_glb(mid).exists():
        raise HTTPException(status_code=400, detail="Run base texturing first")
    overlay_path = str(_model_dir(mid) / f"handpaint_{view}.png")
    _save_upload_png(await overlay.read(), Path(overlay_path))
    sjid = new_job(f"Hand painting {view}", mid)
    if view == "custom":
        e, a = _vangles(elev, azim)
        set_job(sjid, _view=view, _image=overlay_path, _elev=e, _azim=a)
    else:
        set_job(sjid, _view=view, _image=overlay_path)
        data = _load(mid)
        data["faces"][view]["status"] = "texturing"
        _save(mid, data)
    submit_gpu("studio_handpaint", sjid)
    return public_job(sjid)


# --------------------------------------------------------------------------- routes: jobs / download
@router.get("/api/models/{mid}/download/{fmt}")
def download_model(mid: str, fmt: str):
    _vid(mid)
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")
    src = _textured_glb(mid)
    if not src.exists():
        src = _shape_glb(mid)
    if not src.exists():
        raise HTTPException(status_code=404, detail="No model to download")
    if fmt == "glb":
        return FileResponse(src, media_type="model/gltf-binary", filename=f"{mid}.glb")
    out = OUTPUT_DIR / f"{src.stem}.{fmt}"
    if not out.exists():
        from webapp import server
        server._blender_convert(str(src), str(out))
    return FileResponse(out, media_type="application/octet-stream", filename=f"{mid}.{fmt}")
