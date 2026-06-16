"""
MV-Adapter (SDXL / SD2.1) image-to-texture integration.

MV-Adapter is run in a SEPARATE conda env + checkout so its diffusers / peft / etc.
pins never clash with the Hunyuan paint pipeline running in this process. We shell
out to its `scripts.texture_i2tex` entry point, which renders the mesh's geometry
conditioning internally, generates 6 multi-view-consistent images from a single
reference image, and bakes them onto the mesh UVs -> a textured GLB.

This module only INVOKES an already-installed MV-Adapter. One-time setup (clone +
conda env + ~12GB weights) lives in webapp/setup_mvadapter.sh. Keeping MV-Adapter
out of this process is deliberate: it protects the working Hunyuan environment from
dependency drift.
"""
import os
import shutil
import subprocess
from pathlib import Path

# Where the MV-Adapter checkout + its conda env live. Overridable via env so the
# same image can point at a mounted volume (default: the persisted /opt/mvadapter).
MVADAPTER_DIR = os.environ.get("MVADAPTER_DIR", "/workspace/MV-Adapter")
# Isolated conda env, addressed by prefix path (preferred, lives on the volume) or name.
MVADAPTER_CONDA_PREFIX = os.environ.get("MVADAPTER_CONDA_PREFIX")  # e.g. /opt/mvadapter/env
MVADAPTER_CONDA_ENV = os.environ.get("MVADAPTER_CONDA_ENV", "mvadapter")
# SDXL is the only viable base: SD2.1-base is gated on HuggingFace (401 without a
# license-accepted token). cpu-offload keeps SDXL in CPU RAM which OOM-kills the host
# (137); instead we free the Hunyuan worker (server-side) and run SDXL on the GPU at a
# modest resolution so it fits 16GB VRAM. Offload off by default.
MVADAPTER_OFFLOAD = os.environ.get("MVADAPTER_OFFLOAD", "0").lower() in ("1", "true", "yes")
MVADAPTER_UPSCALE = os.environ.get("MVADAPTER_UPSCALE", "1").lower() in ("1", "true", "yes")
MVADAPTER_HEIGHT = int(os.environ.get("MVADAPTER_HEIGHT", "512"))
MVADAPTER_UV_SIZE = int(os.environ.get("MVADAPTER_UV_SIZE", "2048"))

# View set (6 cameras; the adapter is 6-view). "corners" = 3/4 diagonal views which see
# two faces + edges; tilting them DOWN ~45deg also sees OVER neighbouring objects into
# the gaps + the tops of recessed/occluded features, which flat (elev 0) corners miss.
# Corner elevation is configurable: MVADAPTER_CORNER_ELEV (default 45 = tilt down 45deg;
# set 0 for the old flat 3/4 corners). "tilted" = cardinal views at MVADAPTER_TILT_ELEV.
# Explicit MVADAPTER_AZIMUTHS/ELEVATIONS (comma-sep, len 6) override the preset.
MVADAPTER_VIEWSET = os.environ.get("MVADAPTER_VIEWSET", "canonical")
_CE = os.environ.get("MVADAPTER_CORNER_ELEV", "45")   # 3/4-corner tilt (deg, down-positive)
_TE = os.environ.get("MVADAPTER_TILT_ELEV", "20")     # "tilted" preset cardinal elevation
_VIEWSETS = {
    "canonical": ("-90,0,90,180,90,90", "0,0,0,0,89.99,-89.99"),
    "corners": ("-45,45,135,225,90,90", f"{_CE},{_CE},{_CE},{_CE},89.99,-89.99"),
    "tilted": ("-90,0,90,180,90,90", f"{_TE},{_TE},{_TE},{_TE},89.99,-89.99"),
    # 12 cameras over two MV passes (canonical faces + tilted 3/4 corners). >6 views can't
    # use MV-Adapter's 6-view-hardwired TexturePipeline, so this set is baked with Hunyuan's
    # N-view baker (server-side) — faces occluded in one set are covered by the other. The
    # corner half tilts down ~45deg (MVADAPTER_CORNER_ELEV) to reach inter-object occlusions.
    "combined": (
        "-90,0,90,180,90,90,-45,45,135,225,90,90",
        f"0,0,0,0,89.99,-89.99,{_CE},{_CE},{_CE},{_CE},89.99,-89.99",
    ),
}


def _resolve_viewset(name=None):
    az = os.environ.get("MVADAPTER_AZIMUTHS", "")
    el = os.environ.get("MVADAPTER_ELEVATIONS", "")
    if az and el:
        return az, el
    return _VIEWSETS.get(name or MVADAPTER_VIEWSET, _VIEWSETS["canonical"])
# Our in-repo runner (adds offload + tunable res that the upstream CLI lacks).
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mvadapter_runner.py")
# Timeout for one texture job (multiview diffusion + bake). 30 min default.
MVADAPTER_TIMEOUT = int(os.environ.get("MVADAPTER_TIMEOUT", "1800"))


def _conda_run_prefix():
    """Build the `conda run` prefix, resolving the conda binary robustly and
    targeting the isolated env by prefix path (preferred) or name."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or "/workspace/miniconda3/condabin/conda"
    base = [conda, "run", "--no-capture-output"]
    if MVADAPTER_CONDA_PREFIX:
        return base + ["-p", MVADAPTER_CONDA_PREFIX]
    return base + ["-n", MVADAPTER_CONDA_ENV]


def is_available() -> bool:
    """True if an MV-Adapter checkout with the texture script is present."""
    return (Path(MVADAPTER_DIR) / "scripts" / "texture_i2tex.py").exists()


def _ensure_installed():
    if not is_available():
        raise RuntimeError(
            "MV-Adapter is not installed. Expected "
            f"{Path(MVADAPTER_DIR) / 'scripts' / 'texture_i2tex.py'}. Run "
            "webapp/setup_mvadapter.sh (clones the repo, creates the 'mvadapter' "
            "conda env, downloads ~12GB of weights), or set MVADAPTER_DIR to an "
            "existing checkout."
        )


def _build_cmd(save_name, mesh_path, mv_image_path, out_dir, remove_bg, ref_paths, gpt_refine, viewset, bake,
               ref_sides=None, elev_transfer=False, recolor=False):
    cmd = _conda_run_prefix() + [
        "python", RUNNER,
        "--image", os.path.abspath(mv_image_path),
        "--mesh", os.path.abspath(mesh_path),
        "--save_dir", os.path.abspath(out_dir),
        "--save_name", save_name,
        "--height", str(MVADAPTER_HEIGHT),
        "--uv_size", str(MVADAPTER_UV_SIZE),
        "--bake", bake,
    ]
    if MVADAPTER_OFFLOAD:
        cmd += ["--offload"]
    if MVADAPTER_UPSCALE and bake == "full":
        cmd += ["--upscale"]  # upscale only matters for the MV TexturePipeline bake
    if remove_bg:
        cmd += ["--remove_bg"]
    az, el = _resolve_viewset(viewset)
    # Use --opt=value form: values start with '-' (e.g. -45,...) which argparse would
    # otherwise mistake for a flag.
    cmd += [f"--azimuths={az}", f"--elevations={el}"]
    if gpt_refine:
        cmd += ["--gpt_refine"]
        if elev_transfer:
            cmd += ["--elev_transfer"]
        if recolor:
            cmd += ["--recolor"]
        sides = ref_sides or []
        for i, r in enumerate(ref_paths or []):
            cmd += ["--ref", os.path.abspath(r)]
            # --opt=value form: side tags are plain words but keep it consistent/safe.
            cmd += [f"--ref_side={sides[i] if i < len(sides) else 'any'}"]
    return cmd


def _run_runner(cmd):
    # Pass HF_TOKEN (loaded from .env into this process) so model downloads are
    # authenticated (no throttle); alias the name HF's client also recognizes.
    env = os.environ.copy()
    if env.get("HF_TOKEN") and not env.get("HUGGING_FACE_HUB_TOKEN"):
        env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
    proc = subprocess.run(
        cmd, cwd=MVADAPTER_DIR, capture_output=True, text=True, timeout=MVADAPTER_TIMEOUT, env=env
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-4000:]
        raise RuntimeError(f"MV-Adapter run failed (exit {proc.returncode}): {tail}")
    return proc


def generate_textured_glb(mesh_path: str, mv_image_path: str, out_dir: str,
                          uid: str, remove_bg: bool = True, ref_paths=None,
                          gpt_refine: bool = False, viewset=None, ref_sides=None,
                          elev_transfer: bool = False, recolor: bool = False) -> str:
    """Texture `mesh_path` with MV-Adapter (its own TexturePipeline bake, <=6 views);
    return the GLB path. `mv_image_path` conditions the multiview generation; `gpt_refine`
    refines each view with gpt-image-2/Gemini using `ref_paths` (optionally side-tagged via
    `ref_sides`) before baking. `elev_transfer` switches the refine to strict elevation
    transfer (appearance from the side elevation, shape only from the MV draft)."""
    _ensure_installed()
    save_name = f"{uid}_mvadapter"
    _run_runner(_build_cmd(save_name, mesh_path, mv_image_path, out_dir, remove_bg,
                           ref_paths, gpt_refine, viewset, "full", ref_sides=ref_sides,
                           elev_transfer=elev_transfer, recolor=recolor))
    produced = Path(out_dir) / f"{save_name}_shaded.glb"
    if not produced.exists():
        raise RuntimeError(f"MV-Adapter produced no GLB at {produced}")
    textured_path = Path(out_dir) / f"{uid}_textured.glb"
    shutil.copyfile(produced, textured_path)
    return str(textured_path)


def generate_mv_views(mesh_path: str, mv_image_path: str, out_dir: str, uid: str,
                      remove_bg: bool = True, ref_paths=None, gpt_refine: bool = False,
                      viewset=None, ref_sides=None, elev_transfer: bool = False, recolor: bool = False):
    """Generate MV-Adapter views WITHOUT baking (for the Hunyuan N-view baker, e.g. the
    >6-view 'combined' set). Returns (view_paths, angles) where angles = [(az, el), ...].
    `ref_sides` (parallel to `ref_paths`) tags each reference with the side it depicts so
    the gpt refine feeds each view only its matching-side reference(s). `elev_transfer`
    switches the refine to strict elevation transfer (appearance from elevation, shape from MV)."""
    import json

    _ensure_installed()
    save_name = f"{uid}_mvadapter"
    _run_runner(_build_cmd(save_name, mesh_path, mv_image_path, out_dir, remove_bg,
                           ref_paths, gpt_refine, viewset, "none", ref_sides=ref_sides,
                           elev_transfer=elev_transfer, recolor=recolor))
    angles_path = Path(out_dir) / f"{save_name}_angles.json"
    if not angles_path.exists():
        raise RuntimeError("MV-Adapter gen produced no angles file")
    angles = json.loads(angles_path.read_text())
    view_paths = [str(Path(out_dir) / f"{save_name}_view{i}.png") for i in range(len(angles))]
    return view_paths, angles
