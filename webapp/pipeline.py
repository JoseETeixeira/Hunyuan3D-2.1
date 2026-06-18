"""
Two-step Hunyuan3D-2.1 pipeline wrapper for the web app.

Splits the upstream monolithic generate() into independent steps so the UI can
do the meshy.ai flow: generate shape -> preview -> generate texture.

Construction + call signatures mirror the authoritative usage in gradio_app.py
(use_safetensors=False, output_type='mesh', export_to_trimesh, FaceReducer) so
behavior matches the official demo rather than the param-less api_server path.
"""
import os
import sys

# hy3dshape / hy3dpaint are imported as top-level packages from the repo root.
sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

# Torchvision compatibility shim (same as demo.py / model_worker.py).
try:
    from torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:  # noqa: BLE001
    print(f"Warning: Failed to apply torchvision fix: {e}")

import torch

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


def _convert_obj_to_glb(obj_path: str, glb_path: str) -> bool:
    """Convert a textured OBJ (+ PBR maps) to a GLB. Mirrors model_worker."""
    textures = {
        "albedo": obj_path.replace(".obj", ".jpg"),
        "metallic": obj_path.replace(".obj", "_metallic.jpg"),
        "roughness": obj_path.replace(".obj", "_roughness.jpg"),
    }
    return create_glb_with_pbr_materials(obj_path, textures, glb_path)


def _force_matte(mesh):
    """Render a textured mesh as flat matte color (no metallic/specular chrome)."""
    import trimesh
    from trimesh.visual.material import PBRMaterial

    def apply(g):
        mat = getattr(getattr(g, "visual", None), "material", None)
        img = None
        if mat is not None:
            img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
        if img is not None:
            g.visual.material = PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=1.0)

    if isinstance(mesh, trimesh.Scene):
        for g in mesh.geometry.values():
            apply(g)
    else:
        apply(mesh)


def _silhouette_bbox(normal_img, size):
    """Bounding box of the mesh silhouette in a rendered normal map (corner = bg)."""
    import numpy as np

    arr = np.asarray(normal_img.convert("RGB").resize((size, size)), dtype=np.int16)
    bg = arr[0, 0]
    mask = np.abs(arr - bg).sum(axis=-1) > 24
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, size, size)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _align_photo(rgba, mesh_bbox, size, fit="fill", mirror=False, with_alpha=False):
    """Place a photo's subject (from its alpha) onto the mesh silhouette bbox.

    fit="fill" (default): map the subject's bbox exactly onto the mesh's projected
    bbox in both axes, so the subject reaches the silhouette edges — what projection
    texturing wants (no bare borders, subject edges land on the outline). Minor aspect
    distortion is acceptable because the bake only samples where the mesh projects.
    fit="contain": preserve aspect inside the bbox (the old behavior; can leave gaps).
    mirror=True: horizontally flip the subject content before placing. The renderer's
    back-projection bakes views horizontally mirrored vs a normal photo, so flipping
    the source un-mirrors the result (the face/azimuth is unchanged — only content).
    with_alpha=True: return an RGBA canvas whose alpha is the subject's coverage (0 where
    the subject does not paint) so the caller can bake ONLY where the subject covers,
    instead of letting the white background canvas leak into the bake.
    """
    from PIL import Image

    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    if mirror:
        rgba = rgba.transpose(Image.FLIP_LEFT_RIGHT)
    sub = rgba.getchannel("A").getbbox() or (0, 0, rgba.width, rgba.height)
    subject = rgba.crop(sub)
    mx0, my0, mx1, my1 = mesh_bbox
    tw, th = max(1, mx1 - mx0), max(1, my1 - my0)
    sw, sh = subject.size
    if fit == "contain":
        scale = min(tw / sw, th / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    else:  # fill — subject bbox -> silhouette bbox in both axes
        nw, nh = tw, th
    subject = subject.resize((nw, nh))
    pos = (mx0 + (tw - nw) // 2, my0 + (th - nh) // 2)
    sub_alpha = subject.getchannel("A")
    if with_alpha:
        canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        canvas.paste(subject, pos, sub_alpha)
        return canvas
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(subject.convert("RGB"), pos, sub_alpha)
    return canvas


def _best_silhouette_fit(moving, target, size):
    """Refine a roughly-aligned mask onto the TRUE mesh silhouette with a small per-axis scale
    (about the silhouette centroid) + translation that maximises IoU. Returns a 2x3 affine, or
    None when the identity is already best. The returned affine is in `size`-space, so it applies
    directly to the full-resolution aligned image.

    `_align_photo` fits the painted subject's bbox to the silhouette bbox, but the generator
    (gpt/gemini) reframes the asset (~1.4x zoom + a few % aspect change it cannot pixel-lock) and
    thin outliers like trees skew the painted bbox, so that fit leaves a residual global scale/
    offset — the "close but not exact" drift. This mops it up. Monotonic: seeded at the identity
    IoU, it only takes a better fit, so an already-aligned view (e.g. the 3/4 corners) is a no-op
    and never degraded."""
    import cv2
    import numpy as np

    # Search at a fixed low resolution so cost is independent of the render/texture size; the
    # per-axis scale is resolution-free and translations scale linearly back to `size`-space.
    S = 256 if size > 256 else size
    k = size / S
    tgt = cv2.resize(target.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
    mov = cv2.resize(moving.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(tgt)
    if xs.size == 0 or not mov.any():
        return None
    cx = float(xs.min() + xs.max()) / 2.0
    cy = float(ys.min() + ys.max()) / 2.0

    def _iou(a):
        uni = np.logical_or(a, tgt).sum()
        return np.logical_and(a, tgt).sum() / uni if uni else 0.0

    best_v = _iou(mov > 0)
    best = None
    shifts = (-8, -4, 0, 4, 8)
    for ax in np.linspace(0.92, 1.08, 9):
        for ay in np.linspace(0.92, 1.08, 9):
            for dx in shifts:
                for dy in shifts:
                    M = np.array([[ax, 0.0, cx - cx * ax + dx],
                                  [0.0, ay, cy - cy * ay + dy]], np.float32)
                    w = cv2.warpAffine(mov, M, (S, S), flags=cv2.INTER_NEAREST) > 0
                    v = _iou(w)
                    if v > best_v:
                        best_v = v
                        best = (ax, ay, dx, dy)
    if best is None:
        return None
    ax, ay, dx, dy = best
    cxf, cyf = cx * k, cy * k
    return np.array([[ax, 0.0, cxf - cxf * ax + dx * k],
                     [0.0, ay, cyf - cyf * ay + dy * k]], np.float32)


def _extract_base_texture(mesh, glb_path):
    """Pull the base-color texture image out of a textured mesh/GLB. The renderer's load_mesh
    reads UVs but hardcodes texture_data=None, so reface must seed the existing texture itself.
    Returns a PIL image or None."""
    import trimesh

    def _from(visual):
        mat = getattr(visual, "material", None)
        if mat is None:
            return None
        for attr in ("baseColorTexture", "image"):
            img = getattr(mat, attr, None)
            if img is not None:
                return img
        return None

    img = _from(getattr(mesh, "visual", None))
    if img is not None:
        return img
    # force="mesh" can drop the material on concatenation -> reload as a scene and scan geometries.
    sc = trimesh.load(glb_path)
    geoms = list(sc.geometry.values()) if isinstance(sc, trimesh.Scene) else [sc]
    for g in geoms:
        img = _from(getattr(g, "visual", None))
        if img is not None:
            return img
    return None


class TextureWorker:
    """Loads the shape + paint pipelines once and exposes step-wise generation."""

    def __init__(
        self,
        output_dir: str,
        model_path: str = "tencent/Hunyuan3D-2.1",
        subfolder: str = "hunyuan3d-dit-v2-1",
        device: str = "cuda",
        low_vram_mode: bool = False,
        enable_flashvdm: bool = False,
        compile: bool = False,
        mc_algo: str = "mc",
        max_num_view: int = 6,
        tex_resolution: int = 512,
    ):
        self.output_dir = output_dir
        self.device = device
        self.low_vram_mode = low_vram_mode
        os.makedirs(output_dir, exist_ok=True)

        print(f"[TextureWorker] loading shape model {model_path}/{subfolder} ...")
        self.rembg = BackgroundRemover()
        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=False,
            device=device,
        )
        if enable_flashvdm:
            algo = "mc" if device in ("cpu", "mps") else mc_algo
            self.shape_pipeline.enable_flashvdm(mc_algo=algo)
        if compile:
            self.shape_pipeline.compile()

        # Sequential VRAM: on limited VRAM (e.g. 16GB) the shape and paint models
        # can't co-reside. The pipeline's diffusers-style enable_model_cpu_offload
        # is broken here (no `.components`), so we move the shape model explicitly via
        # .to(): park it on CPU now, pull it to GPU only for shape gen, push it back.
        self.sequential_vram = low_vram_mode
        if self.sequential_vram:
            self.shape_pipeline.to("cpu")
            torch.cuda.empty_cache()
            print("[TextureWorker] shape model parked on CPU (sequential VRAM)")

        print(f"[TextureWorker] loading paint model (views={max_num_view}, res={tex_resolution}) ...")
        conf = Hunyuan3DPaintConfig(max_num_view=max_num_view, resolution=tex_resolution)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        if self.sequential_vram:
            # Park the big paint UNet on CPU too, so it never co-resides with the
            # shape model (which would peg a 16GB card at ~15.9GB and thrash).
            self._move_multiview("cpu")
        print("[TextureWorker] ready.")

    def _move_multiview(self, device):
        """Move the heavy multiview diffusion UNet (+DINO) on/off the GPU."""
        mv = self.paint_pipeline.models.get("multiview_model")
        if mv is None:
            return
        if getattr(mv, "pipeline", None) is not None:
            mv.pipeline.to(device)
        if getattr(mv, "dino_v2", None) is not None:
            mv.dino_v2.to(device)
        if hasattr(mv, "device"):
            mv.device = device
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_shape(
        self,
        uid: str,
        image,
        remove_background: bool = True,
        steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 1234,
        octree_resolution: int = 256,
        num_chunks: int = 8000,
        face_count: int = 40000,
    ):
        """Run image -> untextured mesh. Returns (shape_glb_path, processed_image).

        face_count is the mesh polygon budget, applied to the shape itself so the
        preview/download and the eventual textured mesh all respect it.
        """
        if remove_background or image.mode == "RGB":
            image = self.rembg(image.convert("RGB"))

        if self.sequential_vram:
            self.shape_pipeline.to(self.device)  # bring shape model onto the GPU

        generator = torch.Generator().manual_seed(int(seed))
        outputs = self.shape_pipeline(
            image=image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            octree_resolution=int(octree_resolution),
            num_chunks=int(num_chunks),
            output_type="mesh",
        )
        mesh = export_to_trimesh(outputs)[0]

        # Apply the mesh polygon budget to the shape itself (quadric decimation).
        if face_count and len(mesh.faces) > int(face_count):
            mesh = mesh.simplify_quadric_decimation(int(face_count))

        shape_path = os.path.join(self.output_dir, f"{uid}_shape.glb")
        mesh.export(shape_path)

        # Persist the processed (background-removed) image so the texture step
        # can run later without re-uploading.
        processed_path = os.path.join(self.output_dir, f"{uid}_input.png")
        image.save(processed_path)

        if self.sequential_vram:
            self.shape_pipeline.to("cpu")  # free GPU for the paint pipeline
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return shape_path, processed_path

    @torch.inference_mode()
    def generate_texture(self, uid: str, shape_glb_path: str, images, face_count: int = 40000,
                         views: int = None, tex_resolution: int = None, albedo_only: bool = False):
        """Run untextured mesh + reference image(s) -> textured GLB.

        images may be a single PIL image or a list of reference images for the
        same mesh. face_count is the polygon budget. views (e.g. 6-9) and
        tex_resolution (512/768) are read at call time, so they can be tuned per
        request for higher fidelity to the reference image (VRAM permitting).
        """
        if views:
            self.paint_pipeline.config.max_selected_view_num = int(views)
        if tex_resolution:
            self.paint_pipeline.config.resolution = int(tex_resolution)
        if self.sequential_vram:
            self._move_multiview("cuda")  # bring the paint UNet onto the GPU
        obj_path = os.path.join(self.output_dir, f"{uid}_texturing.obj")
        textured_obj = self.paint_pipeline(
            mesh_path=shape_glb_path,
            image_path=images,
            output_mesh_path=obj_path,
            save_glb=False,
            target_face_count=int(face_count),
        )

        textured_path = os.path.join(self.output_dir, f"{uid}_textured.glb")
        if albedo_only:
            # Flat colors: keep only the albedo (map_Kd), drop metallic/roughness, matte.
            import trimesh

            mesh_out = trimesh.load(textured_obj)
            _force_matte(mesh_out)
            mesh_out.export(textured_path)
        else:
            tmp_glb = os.path.join(self.output_dir, f"{uid}_texturing.glb")
            _convert_obj_to_glb(textured_obj, tmp_glb)
            os.replace(tmp_glb, textured_path)

        if self.sequential_vram:
            self._move_multiview("cpu")  # park it again so shape gen has headroom
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return textured_path

    # Canonical camera angle (elevation, azimuth) per labelled view.
    PROJECTION_CAMS = {
        "front": (0, 0),
        "back": (0, 180),
        "left": (0, 90),
        "right": (0, 270),
        "top": (90, 0),
        "bottom": (-90, 0),
    }

    @torch.inference_mode()
    def project_texture(self, uid: str, shape_glb_path: str, view_images: dict, mirror: bool = False):
        """Project view images onto the mesh UV from canonical camera angles.

        view_images maps angle name (front/back/left/right/top/bottom) -> PIL image.
        Reuses Hunyuan's back-projection baker + UV inpaint (no diffusion). Callers
        pass already background-removed (and, for real photos, pre-flipped) images;
        mirror=True additionally flips every view here (kept for flexibility).
        """
        import numpy as np
        import trimesh
        from utils.uvwrap_utils import mesh_uv_wrap

        pp = self.paint_pipeline
        render_size = pp.config.render_size

        items = [(a, img) for a, img in view_images.items()
                 if self.PROJECTION_CAMS.get(a) is not None and img is not None]
        if not items:
            raise ValueError("No view images provided for projection")
        elevs = [self.PROJECTION_CAMS[a][0] for a, _ in items]
        azims = [self.PROJECTION_CAMS[a][1] for a, _ in items]

        mesh = trimesh.load(shape_glb_path, force="mesh")
        mesh = mesh_uv_wrap(mesh)
        pp.render.load_mesh(mesh=mesh)

        # Auto-align: render the mesh silhouette at each camera, then scale + center
        # the photo's subject to that silhouette so projection lands in the right
        # place and scale (no manual nudging).
        normals = pp.view_processor.render_normal_multiview(elevs, azims)
        views = [_align_photo(img, _silhouette_bbox(nrm, render_size), render_size, mirror=mirror)
                 for (_, img), nrm in zip(items, normals)]
        weights = [1.0] * len(views)

        texture, mask = pp.view_processor.bake_from_multiview(views, elevs, azims, weights)
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = pp.view_processor.texture_inpaint(texture, mask_np)
        pp.render.set_texture(texture, force_set=True)

        obj_path = os.path.join(self.output_dir, f"{uid}_proj.obj")
        pp.render.save_mesh(obj_path, downsample=True)

        # Projection is albedo-only (OBJ + MTL map_Kd). Convert straight to GLB with
        # trimesh; create_glb_with_pbr_materials would require metallic/roughness maps.
        textured_path = os.path.join(self.output_dir, f"{uid}_textured.glb")
        mesh_out = trimesh.load(obj_path)
        _force_matte(mesh_out)  # flat colors, not chrome
        mesh_out.export(textured_path)

        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return textured_path

    @torch.inference_mode()
    def project_texture_angles(self, uid: str, shape_glb_path: str, items, mirror: bool = False, weights=None, bake_exp=None, fit: str = "fill"):
        """Bake views given at explicit (elev, azim) angles onto the mesh UV using the
        Hunyuan N-view baker. items = list of (PIL image, elev_deg, azim_deg). Handles
        any number of views (e.g. 12 for the combined set) with cosine/visibility
        weighting + UV inpaint — used to combine MV-Adapter view sets the MV bake can't.
        `weights` (per-view, optional) lets callers down-weight less-reliable views
        (e.g. 3/4 corners) so cleaner views dominate where both are visible."""
        import numpy as np
        import trimesh
        from utils.uvwrap_utils import mesh_uv_wrap

        pp = self.paint_pipeline
        render_size = pp.config.render_size
        keep = [(img, float(e), float(a), w) for (img, e, a), w in
                zip(items, (weights or [1.0] * len(items))) if img is not None]
        if not keep:
            raise ValueError("No views to project")
        elevs = [e for _, e, _, _ in keep]
        azims = [a for _, _, a, _ in keep]
        view_weights = [w for _, _, _, w in keep]

        mesh = trimesh.load(shape_glb_path, force="mesh")
        mesh = mesh_uv_wrap(mesh)
        pp.render.load_mesh(mesh=mesh)

        normals = pp.view_processor.render_normal_multiview(elevs, azims)
        views = [_align_photo(img, _silhouette_bbox(nrm, render_size), render_size, fit=fit, mirror=mirror)
                 for (img, _, _, _), nrm in zip(keep, normals)]

        # Sharper cosine (higher bake_exp) -> each texel takes more from its single most
        # head-on view, reducing ghosting where many (e.g. 12) views overlap.
        prev_exp = pp.config.bake_exp
        if bake_exp:
            pp.config.bake_exp = float(bake_exp)
        try:
            texture, mask = pp.view_processor.bake_from_multiview(views, elevs, azims, view_weights)
        finally:
            pp.config.bake_exp = prev_exp
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = pp.view_processor.texture_inpaint(texture, mask_np)
        pp.render.set_texture(texture, force_set=True)

        obj_path = os.path.join(self.output_dir, f"{uid}_proj.obj")
        pp.render.save_mesh(obj_path, downsample=True)
        textured_path = os.path.join(self.output_dir, f"{uid}_textured.glb")
        mesh_out = trimesh.load(obj_path)
        _force_matte(mesh_out)
        mesh_out.export(textured_path)
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return textured_path

    @torch.inference_mode()
    def paint_faces(self, uid: str, shape_glb_path: str, view_specs, bake_exp=None, tex_resolution: int = None,
                    albedo_labels=None):
        """Per-view single-shot Hunyuan paint. `view_specs` = list of
        (PIL ref, elev_deg, azim_deg, weight): run Hunyuan paint on ONE view per spec, each
        conditioned on its OWN reference, then bake every painted view into one shared UV
        texture with cosine/visibility weighting + inpaint. Single-view paint works because
        the multiview net derives num_view from the control-image count (one normal+position
        map -> num_view=1). Down-weighted fill views (corners/tilts) cover the oblique and
        recessed texels the head-on cardinal views graze without overriding them; a lower
        `bake_exp` spreads each view's contribution to reduce bare patches. Albedo-only matte.

        `albedo_labels` (optional, parallel to view_specs): when given, each painted view's albedo
        is saved to {output_dir}/{uid}_hyalbedo_{label}.png on a white background. The caller can then
        re-bake those via the occlusion-aware single-winner blender projection (no cross-object blend).
        """
        import numpy as np
        import trimesh
        from PIL import Image
        from utils.uvwrap_utils import mesh_uv_wrap

        if tex_resolution:
            self.paint_pipeline.config.resolution = int(tex_resolution)

        labels = list(albedo_labels) if albedo_labels else None
        specs = [(img, float(e), float(a), float(w), (labels[i] if labels and i < len(labels) else None))
                 for i, (img, e, a, w) in enumerate(view_specs) if img is not None]
        if not specs:
            raise ValueError("No view specs provided")

        pp = self.paint_pipeline
        render_size = pp.config.render_size
        res = pp.config.resolution

        mesh = trimesh.load(shape_glb_path, force="mesh")
        mesh = mesh_uv_wrap(mesh)
        pp.render.load_mesh(mesh=mesh)

        if self.sequential_vram:
            self._move_multiview("cuda")  # bring the paint UNet onto the GPU

        mv = pp.models["multiview_model"]
        elevs, azims, albedos, weights = [], [], [], []
        for ref, elev, azim, weight, label in specs:
            # 1-view geometry control for THIS camera -> num_view=1 in the multiview net.
            normal = pp.view_processor.render_normal_multiview([elev], [azim], use_abs_coor=True)[0]
            position = pp.view_processor.render_position_multiview([elev], [azim])[0]
            # Composite the reference onto white (avoid black halos), like generate_texture.
            if ref.mode == "RGBA":
                white = Image.new("RGB", ref.size, (255, 255, 255))
                white.paste(ref, mask=ref.getchannel("A"))
                ref_rgb = white
            else:
                ref_rgb = ref.convert("RGB")
            pbr = mv([ref_rgb], [normal, position], prompt="high quality",
                     custom_view_size=res, resize_input=True)
            albedo = pp.models["super_model"](pbr["albedo"][0]).resize((render_size, render_size))
            if label:
                albedo.convert("RGB").save(os.path.join(self.output_dir, f"{uid}_hyalbedo_{label}.png"))
            elevs.append(elev); azims.append(azim); albedos.append(albedo); weights.append(weight)

        prev_exp = pp.config.bake_exp
        if bake_exp:
            pp.config.bake_exp = float(bake_exp)
        try:
            texture, mask = pp.view_processor.bake_from_multiview(albedos, elevs, azims, weights)
        finally:
            pp.config.bake_exp = prev_exp
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = pp.view_processor.texture_inpaint(texture, mask_np)
        pp.render.set_texture(texture, force_set=True)

        if self.sequential_vram:
            self._move_multiview("cpu")  # park it again so shape gen has headroom

        obj_path = os.path.join(self.output_dir, f"{uid}_hyface.obj")
        pp.render.save_mesh(obj_path, downsample=True)

        # Albedo-only matte (OBJ + map_Kd) straight to GLB, like project_texture.
        textured_path = os.path.join(self.output_dir, f"{uid}_textured.glb")
        mesh_out = trimesh.load(obj_path)
        _force_matte(mesh_out)
        mesh_out.export(textured_path)
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return textured_path

    @torch.inference_mode()
    def reface(self, uid: str, textured_glb_path: str, elev: float, azim: float, view_image,
               depth_band: float = 1.0, mask=None, mirror: bool = False):
        """Depth-aware single-view re-texture of an ALREADY-textured mesh.

        Loads the mesh preserving its existing UVs + texture (the base). The generated view is
        GEOMETRY-MATCHED to the mesh first (rembg the painted subject, fit it to the rendered
        silhouette — same alignment projection/gptproject use, so scale + position follow the
        geometry, not gpt's drift), then baked at the (elev, azim) camera and composited over the
        base. `depth_band` (0..1) limits the bake to the nearest fraction of the face's depth range;
        the default 1.0 repaints EVERY visible camera-facing surface (back_project skips occluded +
        grazing texels itself). Set it < 1 to repaint only the frontmost slab (a car in front of a
        wall: just the car) — but note a small band drops separate visible objects that merely sit
        farther, e.g. the car roofs on a TOP view. Works for any camera: the 6 cardinal faces and the
        3/4 corners (fl/fr/bl/br). `mask` (PIL, white = repaint) overrides the band. Albedo-only matte.
        """
        import math
        import cv2
        import numpy as np
        import trimesh
        from PIL import Image
        from DifferentiableRenderer.camera_utils import get_mv_matrix

        elev, azim = float(elev), float(azim)
        pp = self.paint_pipeline
        render = pp.render
        rs = pp.config.render_size

        # 1) Load the textured mesh PRESERVING its UVs (no mesh_uv_wrap — re-unwrapping would
        #    discard the existing texture's UV layout). The renderer's load_mesh reads UVs but NOT
        #    the texture image, so seed the existing texture explicitly as the base to composite over.
        mesh = trimesh.load(textured_glb_path, force="mesh")
        render.load_mesh(mesh=mesh)
        tex_img = _extract_base_texture(mesh, textured_glb_path)
        if tex_img is None:
            raise RuntimeError("reface: the source GLB has no readable base-color texture")
        render.set_texture(tex_img.convert("RGB"))  # force_set=False -> resized to texture_size, [0,1]
        base = torch.from_numpy(render.get_texture()).float().to(render.device)  # (Ht,Wt,3) in [0,1]

        # 2) Screen-space depth at the face camera. POSITION encodes each vertex as
        #    (0.5 - vtx_pos/scale_factor) in [0,1] — NOT raw world — so recover the world
        #    position before measuring depth. Validity = the alpha silhouette (POSITION fills
        #    background white). Camera position = -R^T t from the view matrix (same normalized
        #    frame as vtx_pos); euclidean distance is sign-safe (near = small).
        pos = render.render_position(elev, azim, resolution=rs, return_type="th").detach().cpu().numpy().reshape(rs, rs, 3)
        alpha = render.render_alpha(elev, azim, resolution=rs, return_type="th").detach().cpu().numpy().reshape(rs, rs)
        valid = alpha > 0
        world = (0.5 - pos) * float(render.scale_factor)
        w2c = np.asarray(get_mv_matrix(elev, azim, render.camera_distance), dtype=np.float32)
        cam = -w2c[:3, :3].T @ w2c[:3, 3]
        depth = np.linalg.norm(world - cam[None, None, :], axis=-1)

        # 3) Foreground mask: explicit mask overrides; else keep texels within the nearest depth band.
        #    depth_band=1.0 (default) -> thr = max depth -> fg = every visible texel. Repainting ALL
        #    visible surfaces is correct: back_project only writes visible, camera-facing texels (it
        #    skips occluded ones per-pixel and grazing ones via the cos threshold), so "farther" does
        #    NOT mean "behind". A SMALL band keeps only the nearest slab — useful to repaint just the
        #    frontmost object — but it wrongly drops separate visible objects that merely sit farther:
        #    on a TOP view the roof is near and the cars sit far below it, so a small band leaves the
        #    car roofs unpainted. Hence the default repaints everything visible.
        if mask is not None:
            fg = (np.asarray(mask.convert("L").resize((rs, rs))) > 127) & valid
        else:
            dv = depth[valid]
            if dv.size and float(dv.max()) > float(dv.min()) and float(depth_band) < 1.0:
                thr = float(dv.min()) + float(depth_band) * (float(dv.max()) - float(dv.min()))
                fg = valid & (depth <= thr)
            else:
                fg = valid  # band >= 1.0 or flat face -> repaint all visible

        # 4) Geometry-match the generated view to the mesh, then bake the foreground band.
        #    rembg the painted subject + fit it to the rendered silhouette bbox so its scale and
        #    position follow the geometry (not gpt's ~12% drift) — exactly how projection/
        #    gptproject align before baking. The foreground∧coverage mask rides in the alpha so
        #    back_project only updates the nearest band where the subject paints. View and mask
        #    are flipped together when mirror is set, so they stay aligned through the renderer's
        #    mirror convention.
        normal = render.render_normal(elev, azim, return_type="pl")
        try:
            subject = self.rembg(view_image.convert("RGB"))
        except Exception:  # noqa: BLE001
            subject = view_image.convert("RGBA")
        aligned = _align_photo(subject, _silhouette_bbox(normal, rs), rs, with_alpha=True)
        # Refine that bbox fit against the ACTUAL mesh silhouette (`valid`). The generator can't
        # pixel-lock the geometry (it reframes the asset ~1.4x + a few % aspect) and thin outliers
        # like trees skew the painted bbox, so the bbox fit leaves a residual global scale/offset
        # — the "close but not exact" drift. A small per-axis scale + shift that maximises overlap
        # with the silhouette removes most of it; monotonic, so an already-aligned view is a no-op.
        aligned_np = np.asarray(aligned)
        refine_M = _best_silhouette_fit(aligned_np[..., 3] > 127, valid, rs)
        if refine_M is not None:
            aligned_np = cv2.warpAffine(aligned_np, refine_M, (rs, rs), flags=cv2.INTER_LINEAR)
        # Bake ONLY where the subject actually paints AND the depth band says foreground.
        # _align_photo sits the subject on a WHITE canvas; gating by the depth band alone bakes
        # that white wherever the aligned subject doesn't cover a foreground texel (silhouette
        # edges + gpt/gemini shape drift) -> the "white where the texture should be" holes. The
        # subject's own coverage (its alpha) closes them: uncovered foreground keeps the base.
        cover = aligned_np[..., 3] > 127
        bake_alpha = ((fg & cover).astype(np.uint8) * 255)
        rgba = np.dstack([aligned_np[..., :3], bake_alpha]).astype(np.float32) / 255.0
        if mirror:
            rgba = rgba[:, ::-1, :].copy()
        new_tex, cos_map, _ = render.back_project(rgba, elev, azim)
        fg_uv = (cos_map[..., 0] > 1e-4) & (new_tex[..., 3] > 0.5)

        try:
            _dv = depth[valid]
            _lo, _hi = (float(_dv.min()), float(_dv.max())) if _dv.size else (0.0, 0.0)
            print(f"[reface] silhouette={int(valid.sum())}px depth=[{_lo:.3f},{_hi:.3f}] "
                  f"foreground={int(fg.sum())}px visible_texels={int((cos_map[..., 0] > 1e-4).sum().item())} "
                  f"repainted_texels={int(fg_uv.sum().item())}")
        except Exception as _e:  # noqa: BLE001
            print(f"[reface] diag failed: {_e}")

        # 5) Composite foreground texels over the existing base; everything else untouched.
        out = base.clone()
        out[fg_uv] = new_tex[..., :3][fg_uv]
        render.set_texture(out, force_set=True)

        obj_path = os.path.join(self.output_dir, f"{uid}_reface.obj")
        render.save_mesh(obj_path, downsample=True)
        textured_path = os.path.join(self.output_dir, f"{uid}_textured.glb")
        mesh_out = trimesh.load(obj_path)
        _force_matte(mesh_out)
        mesh_out.export(textured_path)
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return textured_path

    @torch.inference_mode()
    def fill_coverage_gaps(self, uid: str, textured_glb_path: str, get_reference,
                           standard_cams=None, candidate_cams=None, max_cams: int = 6,
                           dilation_px: int = 4, cos_thres_deg: float = 75.0, min_texels: int = 64,
                           progress=None):
        """Auto coverage-gap fill: texture the oblique/recessed surfaces NO standard view covered.

        Runs as an auto-targeted reface on an ALREADY-textured GLB (same entry shape as reface), so
        reface and hyface share this one path. Steps:
          1. Reload the mesh PRESERVING its UVs and seed the existing texture as the base to composite over.
          2. Coverage re-probe: back_project a dummy at each `standard_cams` camera and accumulate cos_map.
             A texel is a GAP if it is a valid mesh texel but no standard view covered it (trust<=eps).
             back_project applies BOTH the 75deg cos gate AND per-pixel visibility, so this catches
             grazing-angle gaps AND occlusion gaps in one reused primitive.
          3. Dilate the gap mask a few texels so new paint blends over the inpaint seam.
          4. Greedy set-cover: rank `candidate_cams` by predicted coverage of the remaining gap normals
             (ranking uses tex_normal vs the camera's lookat with the SAME sign as the renderer's bake);
             for each chosen camera fetch a reference, bake it, and measure its REAL newly-covered texels.
             Stop at `max_cams` or after 2 low-yield (< `min_texels`) cameras.
          5. Composite each fill camera's paint ONLY onto the gap (dilated) texels; covered texels untouched.
        `get_reference(elev, azim) -> PIL|None`: caller supplies the per-camera reference (reface restyle
        or gpt-synth ladder); returning None skips that camera. Non-fatal: any failure returns the input
        path unchanged so the base bake still ships. Albedo-only matte output.
        """
        import math
        import cv2
        import numpy as np
        import trimesh
        from DifferentiableRenderer.camera_utils import get_mv_matrix
        try:
            from webapp.gapfill_logic import best_candidate
        except ImportError:  # when webapp is not a package on the path
            from gapfill_logic import best_candidate

        pp = self.paint_pipeline
        render = pp.render
        rs = pp.config.render_size

        # Default camera sets (caller usually overrides from env). Standard = the 10 named views that
        # define "covered"; candidates = oblique grid the standard set lacks (+ the standard angles).
        if standard_cams is None:
            corners = [(45.0, 45.0), (45.0, 135.0), (45.0, 225.0), (45.0, 315.0)]
            standard_cams = [(float(e), float(a)) for (e, a) in self.PROJECTION_CAMS.values()] + corners
        if candidate_cams is None:
            elevs = [-60.0, -30.0, 0.0, 30.0, 60.0]
            candidate_cams = [(e, float(a)) for e in elevs for a in range(0, 360, 30)]
            candidate_cams += [c for c in standard_cams if c not in candidate_cams]

        cos_thres = math.cos(math.radians(cos_thres_deg))

        def _lookat_world(elev, azim):
            # World-space direction matching the renderer's camera-space lookat=[0,0,-1]:
            # cos_image = cos([0,0,-1], R @ n_world) = (-R[2,:]) . n_world. So rank with L = -R[2,:].
            r_mv = np.asarray(get_mv_matrix(elev=elev, azim=azim, camera_distance=render.camera_distance,
                                            center=None), dtype=np.float32)
            return -r_mv[2, :3]

        try:
            # 1) Load preserving UVs + seed base texture (mirror reface step 1).
            mesh = trimesh.load(textured_glb_path, force="mesh")
            render.load_mesh(mesh=mesh)
            tex_img = _extract_base_texture(mesh, textured_glb_path)
            if tex_img is None:
                print("[gapfill] no readable base texture; skipping gap-fill")
                return textured_glb_path
            render.set_texture(tex_img.convert("RGB"))
            base = torch.from_numpy(render.get_texture()).float().to(render.device)  # (Ht,Wt,3) in [0,1]

            valid = render.texture_indices >= 0                          # (H,W) valid mesh texels
            idx_map = render.texture_indices                             # (H,W) long, -1 invalid
            tex_normal = render.tex_normal                               # (Nvalid,3) world normals

            # 2) Coverage re-probe with a dummy image (only cos_map matters).
            dummy = np.ones((rs, rs, 3), dtype=np.float32)
            trust = torch.zeros(render.texture_size, device=render.device)
            for (e, a) in standard_cams:
                _, cos_map, _ = render.back_project(dummy, e, a)
                trust = trust + cos_map[..., 0]
            covered = trust > 1e-8
            gap = valid & (~covered)
            gap_np = gap.detach().cpu().numpy().astype(np.uint8)
            n0 = int(gap_np.sum())
            if n0 < min_texels:
                print(f"[gapfill] gaps={n0} < min_texels={min_texels}; nothing to fill")
                if self.low_vram_mode:
                    torch.cuda.empty_cache()
                return textured_glb_path

            # 3) Dilate the WRITE mask; track real coverage on the UNdilated gap.
            k = max(1, int(dilation_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            gap_dilated = torch.from_numpy(cv2.dilate(gap_np, kernel) > 0).to(render.device)
            remaining = gap.clone()

            # Precompute candidate view directions once (renderer convention; distance-independent).
            cand_lookats = [((e, a), _lookat_world(e, a)) for (e, a) in candidate_cams]

            out = base.clone()
            cams_used, tried, strikes = [], set(), 0

            while len(cams_used) < max_cams and strikes < 2:
                if int(remaining.sum()) < min_texels:
                    break
                rem_ids = idx_map[remaining]
                rem_ids = rem_ids[rem_ids >= 0]
                if rem_ids.numel() == 0:
                    break
                rn_np = tex_normal[rem_ids].detach().cpu().numpy()           # (M,3) world normals
                # Rank untried candidates by predicted coverage of the remaining gap normals (pure).
                best, best_score = best_candidate(
                    rn_np, [(ck, L) for (ck, L) in cand_lookats if ck not in tried], cos_thres)
                if best is None or best_score < min_texels:
                    break
                e, a = best
                tried.add(best)

                # Geometry render for THIS camera: the structural lock for synth AND the silhouette
                # for alignment. Computed here (on the already-loaded mesh, no reload) and handed to
                # get_reference so the callback never touches the shared renderer mid-loop.
                normal_pl = render.render_normal(e, a, return_type="pl")
                ref = None
                try:
                    ref = get_reference(e, a, normal_pl)
                except Exception as ex:  # noqa: BLE001
                    print(f"[gapfill] reference failed for elev={e} azim={a}: {ex}")
                    ref = None
                if ref is None:
                    continue  # no usable reference -> skip this camera (not a strike)

                # Geometry-match the reference to the silhouette + bake (mirror reface step 4).
                try:
                    subject = self.rembg(ref.convert("RGB"))
                except Exception:  # noqa: BLE001
                    subject = ref.convert("RGBA")
                aligned = _align_photo(subject, _silhouette_bbox(normal_pl, rs), rs, with_alpha=True)
                aligned_np = np.asarray(aligned)
                scr_alpha = render.render_alpha(e, a, resolution=rs, return_type="th").detach().cpu().numpy().reshape(rs, rs) > 0
                refine_M = _best_silhouette_fit(aligned_np[..., 3] > 127, scr_alpha, rs)
                if refine_M is not None:
                    aligned_np = cv2.warpAffine(aligned_np, refine_M, (rs, rs), flags=cv2.INTER_LINEAR)
                cover = (aligned_np[..., 3] > 127).astype(np.uint8) * 255
                rgba = np.dstack([aligned_np[..., :3], cover]).astype(np.float32) / 255.0
                new_tex, cos_map, _ = render.back_project(rgba, e, a)

                writ = gap_dilated & (cos_map[..., 0] > 1e-4) & (new_tex[..., 3] > 0.5)
                real = remaining & (cos_map[..., 0] > 1e-4) & (new_tex[..., 3] > 0.5)
                n_real = int(real.sum().item())
                if n_real < min_texels:
                    strikes += 1
                    print(f"[gapfill] low-yield cam elev={e} azim={a} real={n_real}; strike {strikes}")
                    continue
                strikes = 0
                out[writ] = new_tex[..., :3][writ]
                remaining = remaining & (~real)
                cams_used.append((e, a))
                if progress is not None:
                    try:
                        progress(len(cams_used), (e, a), int(remaining.sum().item()))
                    except Exception:  # noqa: BLE001
                        pass

            n1 = int(remaining.sum().item())
            print(f"[gapfill] gaps={n0} cams={len(cams_used)} angles={cams_used} remaining={n1}")
            if not cams_used:
                return textured_glb_path  # nothing painted -> leave base untouched

            render.set_texture(out, force_set=True)
            obj_path = os.path.join(self.output_dir, f"{uid}_gapfill.obj")
            render.save_mesh(obj_path, downsample=True)
            mesh_out = trimesh.load(obj_path)
            _force_matte(mesh_out)
            # Export to a temp file then atomically replace, so a failure mid-write never corrupts
            # the base bake (the input GLB) — the stage must be non-fatal end to end.
            tmp_glb = os.path.join(self.output_dir, f"{uid}_gapfill.glb")
            mesh_out.export(tmp_glb)
            os.replace(tmp_glb, textured_glb_path)
            if self.low_vram_mode:
                torch.cuda.empty_cache()
            return textured_glb_path
        except Exception as ex:  # noqa: BLE001
            # Best-effort stage: never break the job; keep the base bake.
            print(f"[gapfill] failed ({ex}); keeping base texture")
            if self.low_vram_mode:
                torch.cuda.empty_cache()
            return textured_glb_path

    @torch.inference_mode()
    def render_geom_shaded(self, shape_glb_path: str, elev: float, azim: float):
        """Grey shaded geometry render at one camera — the 'grey GEOMETRY render' gen_transfer's
        gpt/Gemini prompts expect, but from the Hunyuan camera so a geometry-locked gen+transfer
        output aligns to reface's back_project. Lambert from the camera-facing normal component
        (bright head-on, dark at grazing edges), white background outside the silhouette."""
        import numpy as np
        import trimesh
        from PIL import Image
        from utils.uvwrap_utils import mesh_uv_wrap

        pp = self.paint_pipeline
        render = pp.render
        rs = pp.config.render_size
        mesh = trimesh.load(shape_glb_path, force="mesh")
        mesh = mesh_uv_wrap(mesh)
        render.load_mesh(mesh=mesh)
        nrm = render.render_normal(elev, azim, use_abs_coor=False, return_type="th").detach().cpu().numpy().reshape(rs, rs, 3)
        vis = render.render_alpha(elev, azim, return_type="th").detach().cpu().numpy().reshape(rs, rs) > 0
        n = nrm * 2.0 - 1.0
        facing = np.clip(np.abs(n[..., 2]), 0.0, 1.0)  # camera-axis component (head-on = 1)
        # Map to a MID-GREY band that never reaches white: a head-on surface at shade=1.0 -> 255
        # would vanish into the white background, and the gen+transfer step would paint it white
        # (that was the "plain white where textures should be" bug). Head-on lighter, grazing darker,
        # but always clearly grey vs the white bg.
        shade = 0.30 + 0.42 * facing                   # [0.30, 0.72] -> grey [76, 184]
        grey = (shade * 255.0).astype(np.uint8)
        img = np.repeat(grey[..., None], 3, axis=2)
        img[~vis] = 255                                # white background outside the silhouette
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return Image.fromarray(img)

    @torch.inference_mode()
    def render_textured_view(self, textured_glb_path: str, elev: float, azim: float):
        """Forward-render the ALREADY-textured mesh at one camera: EXACT geometry + the existing baked
        colours, white background. This is the geometry-locked canvas reface restyles toward the
        references. Because it is a COMPLETE colour render (not a grey geom), the image model keeps ITS
        geometry and only takes colour/style from the references — a grey/partial image instead loses
        its shape to a full-colour reference (the genview-drift bug). Same camera path as back_project
        so the result aligns to the bake. Convention mirrors webapp/glb_faces.py."""
        import numpy as np
        import trimesh
        import torch.nn.functional as F
        from PIL import Image
        from DifferentiableRenderer.MeshRender import get_mv_matrix, transform_pos

        pp = self.paint_pipeline
        render = pp.render
        rs = pp.config.render_size
        mesh = trimesh.load(textured_glb_path, force="mesh")
        render.load_mesh(mesh=mesh)
        tex_img = _extract_base_texture(mesh, textured_glb_path)
        if tex_img is None:
            raise RuntimeError("render_textured_view: the source GLB has no readable base-color texture")
        render.set_texture(np.asarray(tex_img.convert("RGB")).astype(np.float32) / 255.0)

        proj = render.camera_proj_mat
        r_mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=render.camera_distance, center=None)
        pos_clip = transform_pos(proj, transform_pos(r_mv, render.vtx_pos, keepdim=True))
        rast_out, _ = render.raster_rasterize(pos_clip, render.pos_idx, resolution=(rs, rs))
        uv, _ = render.raster_interpolate(render.vtx_uv[None, ...], rast_out, render.uv_idx)
        vis = torch.clamp(rast_out[..., -1:], 0, 1)[0]
        tex = render.tex.to(render.device).float()
        if tex.max() > 1.5:
            tex = tex / 255.0
        samp = F.grid_sample(tex.permute(2, 0, 1)[None], uv * 2 - 1, align_corners=False)[0].permute(1, 2, 0)
        img = samp * vis + (1 - vis) * 1.0             # white background outside the silhouette
        out = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return Image.fromarray(out)

    @torch.inference_mode()
    def render_geometry_at(self, shape_glb_path: str, cams):
        """Render surface-normal maps at explicit labeled cameras. `cams` = list of
        (label, elev_deg, azim_deg). Returns {label: PIL normal}. Generalizes
        render_view_geometry (limited to the 6 canonical PROJECTION_CAMS) for fill/corner
        cameras at arbitrary angles — used to seed gpt-synth corner references."""
        import trimesh
        from utils.uvwrap_utils import mesh_uv_wrap

        if not cams:
            return {}
        pp = self.paint_pipeline
        mesh = trimesh.load(shape_glb_path, force="mesh")
        mesh = mesh_uv_wrap(mesh)
        pp.render.load_mesh(mesh=mesh)
        elevs = [float(e) for _, e, _ in cams]
        azims = [float(a) for _, _, a in cams]
        normals = pp.view_processor.render_normal_multiview(elevs, azims)
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return {lbl: nrm for (lbl, _, _), nrm in zip(cams, normals)}

    @torch.inference_mode()
    def render_view_geometry(self, shape_glb_path: str, angles):
        """Render the mesh's surface-normal map from each canonical camera angle.

        Returns {angle: PIL normal map}. These geometry captures seed the gpt-image
        generator so each painted view follows the real silhouette/surface — the
        StableProjectorz depth/normal-guidance idea, adapted to a generator that has
        no ControlNet. Residual misalignment is corrected downstream by
        project_texture's silhouette auto-align + cosine-weighted bake + UV inpaint.
        """
        import trimesh
        from utils.uvwrap_utils import mesh_uv_wrap

        pp = self.paint_pipeline
        items = [(a, self.PROJECTION_CAMS[a]) for a in angles if self.PROJECTION_CAMS.get(a) is not None]
        if not items:
            raise ValueError("No valid angles for geometry capture")
        elevs = [c[0] for _, c in items]
        azims = [c[1] for _, c in items]

        mesh = trimesh.load(shape_glb_path, force="mesh")
        mesh = mesh_uv_wrap(mesh)
        pp.render.load_mesh(mesh=mesh)
        normals = pp.view_processor.render_normal_multiview(elevs, azims)
        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return {a: nrm for (a, _), nrm in zip(items, normals)}

    def release(self):
        """Free GPU memory after a job (auto-clear)."""
        import gc

        gc.collect()
        torch.cuda.empty_cache()
