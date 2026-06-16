"""Headless Blender camera-projection texture bake.

Projects clean per-side ELEVATION images (front/back/left/right/top[/bottom]) onto a mesh
using standard orthographic cameras, then bakes to a single smart-UV texture and exports a
GLB. This replaces the custom Hunyuan back_project for mvgpt: Blender's camera/coordinate
conventions are standard and predictable, so projection scale is uniform (no bbox fill-
stretch) and there is no elevation/azimuth-sign guessing.

Run by webapp.server via:
  blender --background --python blender_project.py -- <spec.json>

spec.json:
  {
    "mesh": "/abs/path/shape.glb",
    "out":  "/abs/path/uid_textured.glb",
    "tex_size": 2048,
    "views": [ {"side":"front","image":"/abs/elev_front.png"}, ... ],   # sides: front/back/left/right/top/bottom
    "debug_dir": "/abs/dir"   # optional: write per-camera verification renders
  }

GLB convention (glTF Y-up; model-viewer default view = +Z front):
  front=+Z  back=-Z  right=+X  left=-X  top=+Y  bottom=-Y
After Blender's glTF import (Y-up -> Z-up), these map to Blender world directions:
  front=-Y back=+Y right=+X left=-X top=+Z bottom=-Z   (camera sits along that axis, looks at origin)
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Vector

# numpy>=1.24 removed aliases that Blender's bundled glTF io still references (see blender_convert.py).
for _n, _t in (("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)

# Side -> (camera position direction in Blender world, camera up vector).
SIDE_DIR = {
    "front": (Vector((0, -1, 0)), Vector((0, 0, 1))),
    "back":  (Vector((0, 1, 0)),  Vector((0, 0, 1))),
    "right": (Vector((1, 0, 0)),  Vector((0, 0, 1))),
    "left":  (Vector((-1, 0, 0)), Vector((0, 0, 1))),
    "top":   (Vector((0, 0, 1)),  Vector((0, 1, 0))),
    "bottom": (Vector((0, 0, -1)), Vector((0, 1, 0))),
}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in GLB")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    # bake transforms so matrix_world is identity (simplifies normals / projection)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def bbox(obj):
    cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
    mx = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
    return mn, mx, (mn + mx) / 2.0


def make_ortho_camera(name, direction, up, center, size):
    cam_data = bpy.data.cameras.new(name)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = size * 1.02            # frame the mesh tightly (uniform scale)
    cam_data.clip_start = 0.001
    cam_data.clip_end = size * 10 + 10
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    dist = size * 3 + 1
    cam.location = center + direction.normalized() * dist
    # orient: -Z of camera points toward the object; up as given
    fwd = (center - cam.location).normalized()
    rot = fwd.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot.to_euler()
    # set explicit up by aligning camera's Y to `up` as much as possible
    cam.rotation_euler = _aim(cam.location, center, up)
    return cam


def _aim(eye, center, up):
    fwd = (center - eye).normalized()
    up = up.normalized()
    right = fwd.cross(up)
    if right.length < 1e-6:
        up = Vector((0, 1, 0)) if abs(fwd.z) > 0.9 else Vector((0, 0, 1))
        right = fwd.cross(up)
    right.normalize()
    true_up = right.cross(fwd).normalized()
    # camera looks along -Z, up = +Y
    import mathutils
    mat = mathutils.Matrix((
        (right.x, true_up.x, -fwd.x),
        (right.y, true_up.y, -fwd.y),
        (right.z, true_up.z, -fwd.z),
    ))
    return mat.to_euler()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec = json.load(open(argv[0]))
    obj = import_glb(spec["mesh"])
    mn, mx, center = bbox(obj)
    size = max((mx - mn).x, (mx - mn).y, (mx - mn).z)
    tex_size = int(spec.get("tex_size", 2048))

    # 1) Smart-UV unwrap -> bake-target UV layer.
    me = obj.data
    while len(me.uv_layers) > 0:
        me.uv_layers.remove(me.uv_layers[0])
    bake_uv = me.uv_layers.new(name="bake_uv")
    me.uv_layers.active = bake_uv
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.005)
    bpy.ops.object.mode_set(mode="OBJECT")

    # 2) Build cameras for the provided sides.
    views = {v["side"]: v for v in spec["views"] if v["side"] in SIDE_DIR}
    cams = {}
    for side, v in views.items():
        d, up = SIDE_DIR[side]
        cams[side] = make_ortho_camera(f"cam_{side}", d, up, center, size)

    # 3) Per-face: assign to the camera whose view best faces the face (world normal . dir-to-cam).
    sides = list(views.keys())
    face_side = [None] * len(me.polygons)
    for fi, poly in enumerate(me.polygons):
        n = poly.normal.normalized()
        best, best_d = None, 0.15  # require a minimally head-on facing
        fc = poly.center
        for side in sides:
            to_cam = (cams[side].location - fc).normalized()
            dot = n.dot(to_cam)
            if dot > best_d:
                best, best_d = side, dot
        face_side[fi] = best

    # 4) proj_uv layer: each face's loop UVs = camera-projected coords of its assigned camera.
    # Use the camera MVP directly (world_to_camera_view is unreliable here without a fully
    # evaluated depsgraph, and silently returns out-of-frame coords).
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    mvps = {}
    for side, cam in cams.items():
        bpy.context.view_layer.update()
        proj = cam.calc_matrix_camera(deps, x=1, y=1)
        mvps[side] = proj @ cam.matrix_world.inverted()
    proj_uv = me.uv_layers.new(name="proj_uv")
    for fi, poly in enumerate(me.polygons):
        side = face_side[fi]
        if side is None:
            continue
        mvp = mvps[side]
        for li in poly.loop_indices:
            vidx = me.loops[li].vertex_index
            co = obj.matrix_world @ me.vertices[vidx].co
            clip = mvp @ co.to_4d()
            w = clip.w if abs(clip.w) > 1e-9 else 1.0
            proj_uv.data[li].uv = (clip.x / w * 0.5 + 0.5, clip.y / w * 0.5 + 0.5)

    # 5) Per-side emission materials sampling the elevation via proj_uv; assign per face.
    obj.data.materials.clear()
    side_mat_index = {}
    for i, side in enumerate(sides):
        img = bpy.data.images.load(views[side]["image"])
        mat = bpy.data.materials.new(f"mat_{side}")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.extension = "EXTEND"
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = "proj_uv"
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        obj.data.materials.append(mat)
        side_mat_index[side] = i
    # faces with no assigned side -> a neutral material (index = len)
    neutral = bpy.data.materials.new("mat_neutral")
    neutral.use_nodes = True
    nt = neutral.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.append(neutral)
    neutral_idx = len(sides)
    for fi, poly in enumerate(me.polygons):
        side = face_side[fi]
        poly.material_index = side_mat_index.get(side, neutral_idx)

    # 6) Target image + a target Image Texture node (on bake_uv) added to EVERY material, set active.
    target = bpy.data.images.new("baked", tex_size, tex_size, alpha=False)
    for mat in obj.data.materials:
        nt = mat.node_tree
        tnode = nt.nodes.new("ShaderNodeTexImage")
        tnode.image = target
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = "bake_uv"
        nt.links.new(uvn.outputs["UV"], tnode.inputs["Vector"])
        nt.nodes.active = tnode  # bake target

    # 7) Cycles EMIT bake -> target (each face emits its projected elevation colour).
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    except Exception:
        scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    me.uv_layers.active = me.uv_layers["bake_uv"]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", margin=max(2, tex_size // 256), use_clear=True)

    # 8) Replace materials with a single Principled material using the baked texture on bake_uv.
    obj.data.materials.clear()
    fmat = bpy.data.materials.new("baked_mat")
    fmat.use_nodes = True
    nt = fmat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    try:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0   # Blender 4.x
    except Exception:
        try:
            bsdf.inputs["Specular"].default_value = 0.0
        except Exception:
            pass
    bsdf.inputs["Roughness"].default_value = 1.0
    tnode = nt.nodes.new("ShaderNodeTexImage")
    tnode.image = target
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "bake_uv"
    nt.links.new(uvn.outputs["UV"], tnode.inputs["Vector"])
    nt.links.new(tnode.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(fmat)
    for layer in list(me.uv_layers):
        if layer.name != "bake_uv":
            me.uv_layers.remove(layer)

    # optional: per-camera verification renders so the side->face mapping can be checked
    debug_dir = spec.get("debug_dir")
    if debug_dir:
        import os
        scene.render.engine = "BLENDER_WORKBENCH"          # flat, unlit texture preview
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "TEXTURE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.resolution_x = scene.render.resolution_y = 512
        scene.render.film_transparent = True
        for side, cam in cams.items():
            scene.camera = cam
            scene.render.filepath = os.path.join(debug_dir, f"blenderproj_cam_{side}.png")
            bpy.ops.render.render(write_still=True)
        # 3/4 perspective camera to see the whole textured result at once
        cam34 = make_ortho_camera("cam_34", Vector((1, -1.4, 0.9)), Vector((0, 0, 1)), center, size)
        scene.camera = cam34
        scene.render.filepath = os.path.join(debug_dir, "blenderproj_cam_34.png")
        bpy.ops.render.render(write_still=True)

    # 9) Export GLB with the baked texture embedded.
    bpy.ops.export_scene.gltf(filepath=spec["out"], export_format="GLB",
                              export_image_format="AUTO", use_selection=False)
    print("BLENDER_PROJECT_DONE:", spec["out"])


if __name__ == "__main__":
    main()
