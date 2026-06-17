"""Render a textured GLB from cardinal + 3/4 cameras into one montage PNG, to verify
projection/bake alignment. Uses the SAME ortho camera convention as blender_project.py, so a
front render shows exactly how the front elevation landed on the mesh.

Run: blender --background --python webapp/diag_render_glb.py -- <glb> <out.png> [res]
"""
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
GLB = argv[0]
OUT = argv[1]
RES = int(argv[2]) if len(argv) > 2 else 512

SIDE_DIR = {
    "front": (Vector((0, -1, 0)), Vector((0, 0, 1))),
    "left": (Vector((-1, 0, 0)), Vector((0, 0, 1))),
    "right": (Vector((1, 0, 0)), Vector((0, 0, 1))),
    "back": (Vector((0, 1, 0)), Vector((0, 0, 1))),
    "top": (Vector((0, 0, 1)), Vector((0, 1, 0))),
}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
mx = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
center = (mn + mx) / 2
size = max((mx - mn).x, (mx - mn).y, (mx - mn).z)


def aim(eye, c, up):
    fwd = (c - eye).normalized()
    up = up.normalized()
    right = fwd.cross(up)
    if right.length < 1e-6:
        up = Vector((0, 1, 0)) if abs(fwd.z) > 0.9 else Vector((0, 0, 1))
        right = fwd.cross(up)
    right.normalize()
    tu = right.cross(fwd).normalized()
    return Matrix(((right.x, tu.x, -fwd.x), (right.y, tu.y, -fwd.y), (right.z, tu.z, -fwd.z))).to_euler()


def cam_for(name, d, up, scale_mult):
    cd = bpy.data.cameras.new(name)
    cd.type = "ORTHO"
    cd.ortho_scale = size * scale_mult
    cd.clip_start = 0.001
    cd.clip_end = size * 10 + 10
    cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = center + d.normalized() * (size * 3 + 1)
    cam.rotation_euler = aim(cam.location, center, up)
    return cam


scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8
if scene.world is None:
    scene.world = bpy.data.worlds.new("w")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Strength"].default_value = float(os.environ.get("BG_STR", "1.0"))
sun_d = bpy.data.lights.new("sun", "SUN")
sun_d.energy = 2.5
sun = bpy.data.objects.new("sun", sun_d)
scene.collection.objects.link(sun)
sun.rotation_euler = (0.5, 0.1, 0.4)
if os.environ.get("STUDIO"):
    # bright area "softboxes" -> metallic/glossy surfaces show white highlights (like model-viewer IBL)
    for i, off in enumerate([Vector((2, -2, 2)), Vector((-2, -1.5, 2.2)), Vector((0.5, -3, 1.5))]):
        ld = bpy.data.lights.new(f"area{i}", "AREA")
        ld.energy = size * size * 800
        ld.size = size * 1.5
        la = bpy.data.objects.new(f"area{i}", ld)
        scene.collection.objects.link(la)
        la.location = center + off * size
        la.rotation_euler = aim(la.location, center, Vector((0, 0, 1)))
try:
    scene.view_settings.view_transform = "Standard"
except Exception:
    pass
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = RES

order = ["front", "left", "right", "back", "top"]
seq = [(s, cam_for(f"cam_{s}", *SIDE_DIR[s], 1.02)) for s in order]
seq.append(("34", cam_for("cam_34", Vector((1, -1.4, 0.9)), Vector((0, 0, 1)), 1.25)))
tmp = []
for nm, cam in seq:
    scene.camera = cam
    p = OUT.replace(".png", f"_{nm}.png")
    scene.render.filepath = p
    bpy.ops.render.render(write_still=True)
    tmp.append((nm, p))

try:
    from PIL import Image, ImageDraw
    tiles = [(nm, Image.open(p).convert("RGB")) for nm, p in tmp]
    W = sum(t[1].width for t in tiles)
    H = max(t[1].height for t in tiles)
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    x = 0
    for nm, im in tiles:
        sheet.paste(im, (x, 0))
        d.text((x + 5, 5), nm, fill=(0, 255, 120))
        x += im.width
    sheet.save(OUT)
    print("RENDER_MONTAGE:", OUT)
except Exception as e:  # noqa: BLE001
    print("montage skipped:", e)
print("RENDER_GLB_DONE")
