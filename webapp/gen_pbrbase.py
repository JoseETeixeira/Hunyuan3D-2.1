"""Standalone Hunyuan PBR base generator (runs in the hunyuan env).

Paints a Hunyuan PBR base (albedo + metallic/roughness) for a shape GLB, conditioned on one
reference image. Used to produce the base_glb for an OVERLAY projection bake when testing outside
the server. low_vram_mode so it fits alongside a running server.

Run: python -m webapp.gen_pbrbase <uid> <cond_image_name>   (cond relative to outputs dir)
"""
import sys

from PIL import Image

from webapp.pipeline import TextureWorker

OUT = "/workspace/Hunyuan3D-2.1/webapp/outputs"
uid = sys.argv[1]
cond_name = sys.argv[2] if len(sys.argv) > 2 else "dealership_front.png"

worker = TextureWorker(output_dir=OUT, low_vram_mode=True)
cond = Image.open(f"{OUT}/{cond_name}").convert("RGBA")
out = worker.generate_texture(
    uid=f"{uid}_pbrbase", shape_glb_path=f"{OUT}/{uid}_shape.glb",
    images=[cond], face_count=40000, albedo_only=False,
)
print("PBRBASE_DONE:", out)
