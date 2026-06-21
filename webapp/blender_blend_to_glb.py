"""
Headless Blender .blend -> GLB importer.

Run by webapp.server via:
  blender --background --python blender_blend_to_glb.py -- <input.blend> <output.glb>

Opens the uploaded .blend as the active session and exports its meshes to GLB so the studio
can adopt it as a new (untextured) shape base. Uses Blender's own bundled `bpy`.
"""
import sys

import bpy
import numpy as np

# Restore numpy aliases the bundled glTF exporter still references (see blender_convert.py).
for _name, _ty in (("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _name):
        setattr(np, _name, _ty)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) < 2:
        raise SystemExit("usage: -- <input.blend> <output.glb>")
    src, out = argv[0], argv[1]
    if not src.lower().endswith(".blend"):
        raise SystemExit(f"expected a .blend input, got: {src}")

    # Load the uploaded file as the whole session, then export its meshes to GLB.
    bpy.ops.wm.open_mainfile(filepath=src)

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in .blend")
    # Best-effort unhide so hidden meshes still export.
    for o in meshes:
        try:
            o.hide_set(False)
        except RuntimeError:
            pass  # excluded from the active view layer; export still reaches it
        o.hide_viewport = False
        o.hide_render = False

    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=False)
    print("BLEND_TO_GLB_DONE")


if __name__ == "__main__":
    main()
