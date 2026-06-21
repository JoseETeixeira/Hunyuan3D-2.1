"""
Headless Blender hole-fill pass for a generated mesh.

Run by webapp.server via:
  blender --background --python blender_fillholes.py -- <input.glb> <output.glb>

Imports the GLB, fills boundary holes so the mesh is watertight, recomputes normals,
triangulates, and re-exports GLB. Used right after shape generation. Uses bundled `bpy`.
"""
import sys

import bpy
import numpy as np

for _name, _ty in (("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _name):
        setattr(np, _name, _ty)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) < 2:
        raise SystemExit("usage: -- <input.glb> <output.glb>")
    src, out = argv[0], argv[1]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in GLB")

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # sides=0 closes every boundary loop regardless of size.
    bpy.ops.mesh.fill_holes(sides=0)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=False)
    print("FILL_HOLES_DONE")


if __name__ == "__main__":
    main()
