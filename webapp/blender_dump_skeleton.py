"""
Dump a rigged GLB/FBX armature's bones (name + world-space head/tail + parent) to JSON.

Run by webapp.server via:
  blender --background --python blender_dump_skeleton.py -- <rigged.glb|.fbx> <out.json>

Used after UniRig produces a rig so the studio can map predicted joints to the named markers and
show them in the 3D viewer (positions in the rigged file's own coordinate space). Uses bundled bpy.
"""
import json
import sys

import bpy
import numpy as np

for _n, _t in (("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) < 2:
        raise SystemExit("usage: -- <rigged.glb|.fbx> <out.json>")
    src, out = argv[0], argv[1]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    low = src.lower()
    if low.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=src)
    else:
        bpy.ops.import_scene.gltf(filepath=src)

    arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise SystemExit("no armature found")

    mw = arm.matrix_world

    # Blender is Z-up; glTF (what model-viewer + trimesh use downstream) is Y-up. The glTF importer
    # applies +90deg about X (glTF +Y -> Blender +Z), so to report coords in glTF space we invert it:
    # gltf = (bx, bz, -by). Keeps markers consistent with the viewer + the trimesh recenter ray.
    def to_gltf(v):
        return [v.x, v.z, -v.y]

    bones = []
    for b in arm.data.bones:
        bones.append({
            "name": b.name,
            "head": to_gltf(mw @ b.head_local),
            "tail": to_gltf(mw @ b.tail_local),
            "parent": b.parent.name if b.parent else None,
        })
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"bones": bones}, f)
    print("SKELETON_DUMP_DONE")


if __name__ == "__main__":
    main()
