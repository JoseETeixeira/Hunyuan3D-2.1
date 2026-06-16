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


def _align_photo(rgba, mesh_bbox, size, fit="fill", mirror=False):
    """Place a photo's subject (from its alpha) onto the mesh silhouette bbox.

    fit="fill" (default): map the subject's bbox exactly onto the mesh's projected
    bbox in both axes, so the subject reaches the silhouette edges — what projection
    texturing wants (no bare borders, subject edges land on the outline). Minor aspect
    distortion is acceptable because the bake only samples where the mesh projects.
    fit="contain": preserve aspect inside the bbox (the old behavior; can leave gaps).
    mirror=True: horizontally flip the subject content before placing. The renderer's
    back-projection bakes views horizontally mirrored vs a normal photo, so flipping
    the source un-mirrors the result (the face/azimuth is unchanged — only content).
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
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(subject.convert("RGB"), (mx0 + (tw - nw) // 2, my0 + (th - nh) // 2), subject.getchannel("A"))
    return canvas


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
