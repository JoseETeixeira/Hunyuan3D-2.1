"""
Headless Blender GLB -> FBX/.blend converter.

Run by webapp.server via:
  blender --background --python blender_convert.py -- <input.glb> <output.fbx|output.blend>

This uses Blender's own bundled Python `bpy`, which is unrelated to the (removed)
pip `bpy` module in the app environment.
"""
import sys

import bpy
import numpy as np

# This Blender (Debian build) runs against the env's numpy >=1.24, which removed the
# deprecated aliases np.bool/int/float/object/str that Blender's bundled glTF importer
# and FBX exporter still reference. Restore them so import/export don't crash.
for _name, _ty in (("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _name):
        setattr(np, _name, _ty)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) < 2:
        raise SystemExit("usage: -- <input.glb> <output.fbx|.blend>")
    src, out = argv[0], argv[1]

    # Empty scene, then import the GLB (textures included).
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)

    lower = out.lower()
    if lower.endswith(".fbx"):
        bpy.ops.export_scene.fbx(filepath=out, path_mode="COPY", embed_textures=True)
    elif lower.endswith(".blend"):
        try:
            bpy.ops.file.pack_all()  # embed textures into the .blend
        except Exception as exc:  # noqa: BLE001
            print(f"pack_all skipped: {exc}")
        bpy.ops.wm.save_as_mainfile(filepath=out)
    else:
        raise SystemExit(f"unsupported output format: {out}")


if __name__ == "__main__":
    main()
