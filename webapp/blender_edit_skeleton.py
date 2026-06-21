"""
Move named bones of a skeleton FBX to new world-space head positions, then re-export the skeleton.

Run by webapp.server via:
  blender --background --python blender_edit_skeleton.py -- <spec.json>

spec.json = { "in": "<skeleton.fbx>", "out": "<edited.fbx>", "moves": { "<boneName>": [x,y,z], ... } }

Each move sets that bone's head to the given world position (translating the bone by the delta) and,
when a child is connected, the connection keeps the chain attached. Used by the "Apply rig changes"
re-skin: the edited skeleton is fed back into UniRig generate_skin. Uses bundled bpy.
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Vector

for _n, _t in (("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) < 1:
        raise SystemExit("usage: -- <spec.json>")
    spec = json.loads(open(argv[0], encoding="utf-8").read())
    src, out, moves = spec["in"], spec["out"], spec.get("moves", {})

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=src)
    arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise SystemExit("no armature in skeleton FBX")

    bpy.context.view_layer.objects.active = arm
    inv = arm.matrix_world.inverted()  # world -> armature-local for edit_bone coords
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for name, pos in moves.items():
        bone = eb.get(name)
        if bone is None:
            continue
        # Incoming positions are glTF Y-up (viewer/trimesh space); convert to Blender Z-up world
        # (blender = (gx, -gz, gy)), then into armature-local.
        gx, gy, gz = float(pos[0]), float(pos[1]), float(pos[2])
        target = inv @ Vector((gx, -gz, gy))
        delta = target - bone.head
        # Translate the whole bone so its head lands on the target; connected children follow.
        bone.head = target
        bone.tail = bone.tail + delta
        # The parent ends at this joint when connected — keep it attached.
        if bone.parent is not None and bone.use_connect:
            bone.parent.tail = target
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.export_scene.fbx(filepath=out, add_leaf_bones=False)
    print("SKELETON_EDIT_DONE")


if __name__ == "__main__":
    main()
