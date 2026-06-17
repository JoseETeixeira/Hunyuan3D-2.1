"""Close-up render of the front-lot props (cars) of a textured GLB, to inspect per-object
colour bleed/wash the whole-model renders are too small to show.
Run: blender --background --python webapp/render_cars.py -- <glb> <out.png> [res]
"""
import sys
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
GLB, OUT = argv[0], argv[1]
RES = int(argv[2]) if len(argv) > 2 else 768

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
aimp = Vector((center.x, mn.y + (mx.y - mn.y) * 0.22, mn.z + (mx.z - mn.z) * 0.18))


def aim(eye, c, up):
    fwd = (c - eye).normalized()
    up = up.normalized()
    right = fwd.cross(up)
    right.normalize()
    tu = right.cross(fwd).normalized()
    return Matrix(((right.x, tu.x, -fwd.x), (right.y, tu.y, -fwd.y), (right.z, tu.z, -fwd.z))).to_euler()


scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.world = bpy.data.worlds.new("w")
scene.world.use_nodes = True
scene.world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 1.3
sun_d = bpy.data.lights.new("sun", "SUN"); sun_d.energy = 3.0
sun = bpy.data.objects.new("sun", sun_d); scene.collection.objects.link(sun)
sun.rotation_euler = (0.5, 0.1, 0.4)
try:
    scene.view_settings.view_transform = "Standard"
except Exception:
    pass
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = RES

# LOW/level cameras (slightly below horizon, looking UP at the cars' lower/side faces — where the wash
# is). Aim at the lower body. Four directions so each car's outer AND inner (gap-facing) sides are seen.
lot = Vector((center.x, mn.y + (mx.y - mn.y) * 0.28, mn.z + (mx.z - mn.z) * 0.10))
dirs = [
    ("front_l", Vector((-0.7, -1, -0.05))),
    ("front_r", Vector((0.7, -1, -0.05))),
    ("side_l", Vector((-1, -0.15, -0.05))),
    ("side_r", Vector((1, -0.15, -0.05))),
]
for nm, d in dirs:
    cd = bpy.data.cameras.new(nm); cd.type = "ORTHO"; cd.ortho_scale = size * 0.42
    cd.clip_start = 0.001; cd.clip_end = size * 10 + 10
    cam = bpy.data.objects.new(nm, cd); scene.collection.objects.link(cam)
    cam.location = lot + d.normalized() * (size * 3 + 1)
    cam.rotation_euler = aim(cam.location, lot, Vector((0, 0, 1)))
    scene.camera = cam
    scene.render.filepath = OUT.replace(".png", f"_{nm}.png")
    bpy.ops.render.render(write_still=True)
print("CARZOOM_DONE", OUT)
