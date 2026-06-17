"""SDXL structure-preserving geomatch (runs in the isolated /opt/mvadapter env).

Paint3D-style stage-1: condition SDXL on the mesh's per-view DEPTH (ControlNet-depth) so the
generated image follows the real 3D relief, and drive appearance from a reference elevation via
IP-adapter. Output is a geometry-aligned, flat-cartoon, (near) lighting-free per-view image that
the Blender occlusion bake fuses onto the mesh.

Modes:
  --mode depth   (default): ControlNet-DEPTH (full relief) + IP-adapter. control = {uid}_depth_{side}.png
  --mode canny           : ControlNet-canny on the grey geom (edges only) + IP-adapter (img2img)
  --mode img2img         : init = geom, IP-adapter; grey init resists colour

Single:  --geom CONTROL.png --elev E.png --out O.png    (CONTROL = depth map for depth mode, else geom)
Batch :  --uid U --dir D --sides front,left,right,back,top --ref_prefix dealership_
         (depth mode: control = D/{uid}_depth_{side}.png; canny/img2img: D/{uid}_geom_{side}.png;
          elev = D/{ref_prefix}{side}.png; out = D/{uid}_cnmatch_{side}.png)

Run: /opt/mvadapter/env/bin/python webapp/sdxl_geomatch.py --mode depth --uid U --dir D \
       --sides front,left,right,back,top --ref_prefix dealership_ --cn 0.85 --ip 0.85 --offload
"""
import argparse
import os

import torch
from diffusers import AutoencoderKL
from PIL import Image

STYLE = ("stylized 3D game asset, clean flat cartoon colours faithful to the reference, crisp solid "
         "materials, even soft lighting, plain neutral background, high quality")
NEG = ("blurry, muddy, washed out, desaturated, grayscale, teal cast, deformed, distorted, extra "
       "objects, duplicated objects, text, watermark, photorealistic, busy background, harsh shadows")


def _canny(img, lo=80, hi=180):
    import numpy as np
    try:
        import cv2
        e = cv2.Canny(np.array(img.convert("RGB")), lo, hi)
    except Exception:
        from PIL import ImageFilter
        g = img.convert("L").filter(ImageFilter.FIND_EDGES)
        e = (np.array(g) > 28).astype("uint8") * 255
    return Image.fromarray(np.stack([e, e, e], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom"); ap.add_argument("--elev"); ap.add_argument("--out")
    ap.add_argument("--uid"); ap.add_argument("--dir"); ap.add_argument("--sides", default="front,left,right,back,top")
    ap.add_argument("--ref_prefix", default="dealership_")
    ap.add_argument("--mode", default="depth", choices=["depth", "canny", "img2img"])
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--ip", type=float, default=0.85)
    ap.add_argument("--cn", type=float, default=0.85)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offload", action="store_true")
    args = ap.parse_args()

    ctrl_suffix = "depth" if args.mode == "depth" else "geom"
    jobs = []
    if args.uid and args.dir:
        for s in [x.strip() for x in args.sides.split(",") if x.strip()]:
            c = os.path.join(args.dir, f"{args.uid}_{ctrl_suffix}_{s}.png")
            e = os.path.join(args.dir, f"{args.ref_prefix}{s}.png")
            o = os.path.join(args.dir, f"{args.uid}_cnmatch_{s}.png")
            if os.path.exists(c) and os.path.exists(e):
                jobs.append((c, e, o))
            else:
                print(f"[geomatch] skip {s}: missing {'control' if not os.path.exists(c) else 'elev'}")
    else:
        jobs.append((args.geom, args.elev, args.out))
    for c, e, _o in jobs:
        Image.open(c); Image.open(e)

    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
    if args.mode in ("depth", "canny"):
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
        cn_id = ("diffusers/controlnet-depth-sdxl-1.0" if args.mode == "depth"
                 else "diffusers/controlnet-canny-sdxl-1.0")
        cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, vae=vae,
            variant="fp16", use_safetensors=True, torch_dtype=dtype)
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
            variant="fp16", use_safetensors=True, torch_dtype=dtype)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(args.ip)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload() if args.offload else pipe.to("cuda")

    prompt = (args.prompt + " " if args.prompt else "") + STYLE
    for c, e, o in jobs:
        ctrl = Image.open(c).convert("RGB").resize((args.size, args.size))
        elev = Image.open(e).convert("RGB").resize((args.size, args.size))
        gen = torch.Generator("cuda").manual_seed(args.seed)
        kw = dict(prompt=prompt, negative_prompt=NEG, ip_adapter_image=elev,
                  num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen)
        if args.mode == "depth":
            # init = elevation (exact colours/content), ControlNet = depth (reshape to the mesh's
            # 3D relief). strength high enough to move structure to the geometry while the elevation
            # colour blocking carries through; IP-adapter(elev) reinforces palette.
            res = pipe(image=elev, control_image=ctrl, strength=args.strength,
                       controlnet_conditioning_scale=args.cn, **kw).images[0]
            # Clean the bg: keep only where the depth has the object (depth > 0).
            import numpy as np
            dm = np.asarray(ctrl.convert("L"))
            mask = Image.fromarray(((dm > 8) * 255).astype("uint8")).resize(res.size)
            res = Image.composite(res, Image.new("RGB", res.size, (245, 245, 245)), mask)
        elif args.mode == "canny":
            res = pipe(image=ctrl, control_image=_canny(ctrl), strength=1.0,
                       controlnet_conditioning_scale=args.cn, **kw).images[0]
        else:
            res = pipe(image=ctrl, strength=args.strength, **kw).images[0]
        res.save(o)
        print("SDXL_GEOMATCH_DONE:", o)


if __name__ == "__main__":
    main()
