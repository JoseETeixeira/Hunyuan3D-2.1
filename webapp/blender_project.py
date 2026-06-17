"""Headless Blender camera-projection texture bake (3 modes).

Projects clean per-side ELEVATION images onto a mesh with standard orthographic cameras
(predictable conventions: uniform scale, correct top/bottom). Modes (chosen by spec):

  spec["mode"] == "geometry":
      Render a flat grey shaded view of the bare mesh from each side camera -> geom_<side>.png
      (used by the server to ask gpt-image to geometry-match each elevation before projection).

  spec["base_glb"] present  (OVERLAY / PBR mode):
      Import the Hunyuan-PBR-textured GLB (keeps its UVs + albedo + metallic/roughness), project
      the elevations OVER its base colour (elevations win on faces that face a camera; the Hunyuan
      albedo gap-fills the rest), keep the Hunyuan metallic/roughness, bake to the base UVs, export.

  otherwise  (standalone project):
      Smart-UV unwrap the bare mesh, project elevations, bake, export.

Run by webapp.server via:  blender --background --python blender_project.py -- <spec.json>
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

for _n, _t in (("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)

# Side -> (camera position direction in Blender world, camera up). GLB (glTF Y-up; model-viewer
# front=+Z) imports to Blender Z-up as: front=-Y back=+Y top=+Z bottom=-Z.
# LEFT/RIGHT alignment: to match the hyface path (PROJECTION_CAMS left=azim90, right=azim270 — i.e.
# left = glTF +X, right = glTF -X), the rig is mirrored across X vs the raw glTF axis labels. So here
# LEFT cameras sit at +X and RIGHT at -X (and the corners mirror with them). Keep this in sync with
# PROJECTION_CAMS so a user's "left"/"right" reference lands on the same physical side in both modes.
SIDE_DIR = {
    "front": (Vector((0, -1, 0)), Vector((0, 0, 1))),
    "back": (Vector((0, 1, 0)), Vector((0, 0, 1))),
    "right": (Vector((-1, 0, 0)), Vector((0, 0, 1))),  # glTF -X (mirrored to match hyface)
    "left": (Vector((1, 0, 0)), Vector((0, 0, 1))),    # glTF +X (mirrored to match hyface)
    "top": (Vector((0, 0, 1)), Vector((0, 1, 0))),
    "bottom": (Vector((0, 0, -1)), Vector((0, 1, 0))),
    # 3/4 corner views (azimuth 45deg, tilted down) — cover the diagonal faces + car tops/fronts
    # that the cardinal cameras only graze, reducing seams between adjacent cardinal faces.
    "fr": (Vector((-1, -1, 0.8)), Vector((0, 0, 1))),  # front-right (front + mirrored right -X)
    "fl": (Vector((1, -1, 0.8)), Vector((0, 0, 1))),   # front-left  (front + mirrored left  +X)
    "br": (Vector((-1, 1, 0.8)), Vector((0, 0, 1))),   # back-right
    "bl": (Vector((1, 1, 0.8)), Vector((0, 0, 1))),    # back-left
    # high 3/4 tilt tier (azimuth 45deg, steep ~55deg down) — roof edges + prop tops the flatter
    # mid corners and the absolute top-down only graze.
    "fr_hi": (Vector((-1, -1, 2.0)), Vector((0, 0, 1))),
    "fl_hi": (Vector((1, -1, 2.0)), Vector((0, 0, 1))),
    "br_hi": (Vector((-1, 1, 2.0)), Vector((0, 0, 1))),
    "bl_hi": (Vector((1, 1, 2.0)), Vector((0, 0, 1))),
    # BELOW-horizon fills (camera UNDER the prop, looking up-and-out, ~45deg below). Match the hyface
    # `_lo` views (PROJECTION_CAMS azimuth, elev -45). These reach the cars' DOWN-AND-OUT lower faces
    # (sills/rockers/wheel sides) that every level/down camera misses — used by the single-winner hybrid
    # bake to overlay just those lower faces on top of the cosine result. Same X-mirror as above.
    "front_lo": (Vector((0, -1, -1.0)), Vector((0, 0, 1))),
    "back_lo": (Vector((0, 1, -1.0)), Vector((0, 0, 1))),
    "left_lo": (Vector((1, 0, -1.0)), Vector((0, 0, 1))),    # mirrored: left = +X
    "right_lo": (Vector((-1, 0, -1.0)), Vector((0, 0, 1))),  # mirrored: right = -X
    "fl_lo": (Vector((1, -1, -1.41)), Vector((0, 0, 1))),
    "fr_lo": (Vector((-1, -1, -1.41)), Vector((0, 0, 1))),
    "bl_lo": (Vector((1, 1, -1.41)), Vector((0, 0, 1))),
    "br_lo": (Vector((-1, 1, -1.41)), Vector((0, 0, 1))),
}


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
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def bbox(obj):
    cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
    mx = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
    return mn, mx, (mn + mx) / 2.0


def _aim(eye, center, up):
    fwd = (center - eye).normalized()
    up = up.normalized()
    right = fwd.cross(up)
    if right.length < 1e-6:
        up = Vector((0, 1, 0)) if abs(fwd.z) > 0.9 else Vector((0, 0, 1))
        right = fwd.cross(up)
    right.normalize()
    true_up = right.cross(fwd).normalized()
    return Matrix(((right.x, true_up.x, -fwd.x), (right.y, true_up.y, -fwd.y), (right.z, true_up.z, -fwd.z))).to_euler()


def make_ortho_camera(name, direction, up, center, size):
    cd = bpy.data.cameras.new(name)
    cd.type = "ORTHO"
    cd.ortho_scale = size * 1.02
    cd.clip_start = 0.001
    cd.clip_end = size * 10 + 10
    cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = center + direction.normalized() * (size * 3 + 1)
    cam.rotation_euler = _aim(cam.location, center, up)
    return cam


def setup_cameras(views, center, size):
    return {s: make_ortho_camera(f"cam_{s}", SIDE_DIR[s][0], SIDE_DIR[s][1], center, size)
            for s in views if s in SIDE_DIR}


CARDINAL_SIDES = {"front", "back", "left", "right", "top", "bottom"}


def assign_faces(obj, me, cams, sides, min_dot=0.5, occlude=True, cardinal_bonus=0.2,
                 tilt_up_min=0.2):
    # Assign each face to the ONE camera that best paints it. Priority tiers (the "ordering"):
    #   1. CARDINALS (front/back/left/right/top/bottom): score = dot + cardinal_bonus. They own flat
    #      walls, the facade, the roof and any prop face they see near head-on — orthographic, no
    #      foreshortening, cleanest.
    #   2. MID 3/4 CORNERS (fr/fl/br/bl): score = dot. They win only the genuine 45deg diagonal/bevel
    #      faces no cardinal sees head-on (a corner at dot~0.95 still beats a cardinal grazing the same
    #      bevel at 0.71+0.2=0.91). bonus=0.2 is tuned to that crossover: high enough that a cardinal
    #      keeps the slightly-angled prop faces it depicts well (so a car doesn't ghost between front and
    #      a corner), low enough that true bevels still go to the corner.
    #   3. HIGH TILTS (*_hi): score = dot, but ELIGIBLE ONLY for upward-facing faces (world normal
    #      z > tilt_up_min). A steep-down tilt still has dot>0.5 with vertical walls and car sides, so
    #      without this guard it would steal and SMEAR them; restricting it to roof slopes + prop tops
    #      (its actual job) removes that artifact.
    # All gated by (a) raw dot > min_dot AND (b) VISIBLE (occlusion ray-cast — no bleed of trees onto
    # walls, cars onto the wall behind them). Faces no view wins fall back to the Hunyuan PBR base.
    mw = obj.matrix_world
    mwi = mw.inverted()
    nmat = mw.to_3x3().inverted().transposed()
    face_side = [None] * len(me.polygons)
    for fi, poly in enumerate(me.polygons):
        n = (nmat @ poly.normal).normalized()
        fc = mw @ poly.center
        best, best_score = None, min_dot
        for side in sides:
            camloc = cams[side].location
            d = camloc - fc
            if d.length < 1e-9:
                continue
            dot = n.dot(d.normalized())
            if dot <= min_dot:
                continue
            if side.endswith("_hi") and n.z < tilt_up_min:
                continue  # tilts only paint upward-facing faces (roof/tops), never vertical walls/car sides
            if side.endswith("_lo") and n.z > -tilt_up_min:
                continue  # below-horizon fills only paint DOWN-facing faces (car sills/rockers/underside),
                          # never vertical walls or roofs — those stay on the cosine base.
            score = dot + (cardinal_bonus if side in CARDINAL_SIDES else 0.0)
            if score <= best_score:
                continue
            if occlude:
                o = mwi @ camloc
                tgt = mwi @ fc
                hit, _loc, _nr, idx = obj.ray_cast(o, (tgt - o).normalized())
                if hit and idx != fi:
                    continue  # something between the camera and this face -> occluded
            best, best_score = side, score
        face_side[fi] = best
    return face_side


def _object_bbox(image_path):
    """Object bounding box (u0,u1,v0,v1 in [0,1], v bottom-up) within an elevation on a plain
    background. Used to align the elevation's object to the mesh silhouette during projection."""
    img = bpy.data.images.load(image_path)
    w, h = img.size
    px = np.array(img.pixels[:]).reshape(h, w, 4)  # row 0 = bottom (v=0)
    rgb = px[:, :, :3]
    bg = rgb[0, 0]
    mask = np.abs(rgb - bg).sum(-1) > 0.15
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 1.0, 0.0, 1.0)
    return (xs.min() / w, (xs.max() + 1) / w, ys.min() / h, (ys.max() + 1) / h)


def set_proj_uvs(obj, me, cams, face_side, views):
    deps = bpy.context.evaluated_depsgraph_get()
    mvps = {}
    for side, cam in cams.items():
        bpy.context.view_layer.update()
        mvps[side] = cam.calc_matrix_camera(deps, x=1, y=1) @ cam.matrix_world.inverted()

    # Per-camera FULL-MESH silhouette bbox in the camera frame (project ALL vertices). This is
    # the scale reference: the whole object's extent from that camera maps to the whole object in
    # the elevation. Using only the head-on (building) faces would mis-scale everything, because
    # the elevation's object bbox also includes the lot + foliage that extend beyond the building.
    verts_world = [obj.matrix_world @ v.co for v in me.vertices]
    full_sil = {}
    for side, mvp in mvps.items():
        us, vs = [], []
        for co in verts_world:
            clip = mvp @ co.to_4d()
            w = clip.w if abs(clip.w) > 1e-9 else 1.0
            us.append(clip.x / w * 0.5 + 0.5)
            vs.append(clip.y / w * 0.5 + 0.5)
        full_sil[side] = (min(us), max(us), min(vs), max(vs))
    obj_bbox = {s: _object_bbox(views[s]["image"]) for s in views}

    # Map each camera's full-mesh silhouette bbox onto its elevation's object bbox PER-AXIS
    # (u and v scaled independently), then project each assigned face's loops. Both are orthographic
    # views of the same object, so aligning the two outlines exactly places interior elements
    # (building, lot, cars, foliage) correctly even when the geometry-matched elevation was drawn at
    # a different aspect or margin than the mesh outline. A single uniform scale can't do this: when
    # the elevation's aspect differs (e.g. a narrow side drawn wide), it leaves the off-anchor axis
    # mismatched and pushes the mesh edges onto the wrong part of the image.
    proj_uv = me.uv_layers.new(name="proj_uv")
    for fi, poly in enumerate(me.polygons):
        side = face_side[fi]
        if side is None or side not in full_sil:
            continue
        su0, su1, sv0, sv1 = full_sil[side]
        sw, sh = su1 - su0, sv1 - sv0
        if sw < 1e-6 or sh < 1e-6:
            continue
        ou0, ou1, ov0, ov1 = obj_bbox.get(side, (0.0, 1.0, 0.0, 1.0))
        ku = (ou1 - ou0) / sw
        kv = (ov1 - ov0) / sh
        scu, scv = (su0 + su1) / 2, (sv0 + sv1) / 2
        ocu, ocv = (ou0 + ou1) / 2, (ov0 + ov1) / 2
        mvp = mvps[side]
        for li in poly.loop_indices:
            co = obj.matrix_world @ me.vertices[me.loops[li].vertex_index].co
            clip = mvp @ co.to_4d()
            w = clip.w if abs(clip.w) > 1e-9 else 1.0
            u, v = clip.x / w * 0.5 + 0.5, clip.y / w * 0.5 + 0.5
            proj_uv.data[li].uv = (ocu + (u - scu) * ku, ocv + (v - scv) * kv)
    return proj_uv


def _emit_image_mat(name, image, uv_map):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "EXTEND"
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = uv_map
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _emit_color_mat(name, rgba):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = rgba
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _image_into(principled, socket):
    """Walk back from a Principled input to the Image Texture feeding it (through 1 hop)."""
    inp = principled.inputs.get(socket)
    if not inp or not inp.links:
        return None
    src = inp.links[0].from_node
    if src.type == "TEX_IMAGE":
        return src.image
    for i in src.inputs:
        if i.links and i.links[0].from_node.type == "TEX_IMAGE":
            return i.links[0].from_node.image
    return None


def _cycles_gpu(scene):
    scene.render.engine = "CYCLES"
    # CPU by default: the container's Blender 3.0.1 cannot compile CUDA kernels for newer GPUs
    # ("Failed to execute compilation command"), and Workbench/EEVEE need a display headless.
    # CPU Cycles renders reliably everywhere. Opt into GPU with BLENDER_CYCLES_GPU=1.
    scene.cycles.device = "CPU"
    if os.environ.get("BLENDER_CYCLES_GPU", "0").lower() in ("1", "true", "yes"):
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"
    scene.cycles.samples = 1


def _grey_lit_scene(scene, samples=8):
    """Cycles render setup with a grey object material + bright ambient + THREE suns from different
    azimuths, so the geometry's relief reads clearly from ANY view (a single sun left the side/back
    views dark and flat, giving the colourise weak shape cues). Ambient lifts shadows so no face is
    black; the three suns keep directional relief. Cycles works headless (Workbench/EEVEE need a
    display)."""
    _cycles_gpu(scene)
    scene.cycles.samples = samples
    if scene.world is None:
        scene.world = bpy.data.worlds.new("w")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Strength"].default_value = 1.0  # bright ambient -> no dark faces on any view
    # 3 suns at different azimuths + elevations: every cardinal/corner view gets lit relief.
    for i, rot in enumerate([(0.5, 0.1, 0.4), (0.6, -0.2, 2.5), (0.5, 0.2, 4.2)]):
        sd = bpy.data.lights.new(f"sun{i}", "SUN")
        sd.energy = 2.2
        so = bpy.data.objects.new(f"sun{i}", sd)
        scene.collection.objects.link(so)
        so.rotation_euler = rot
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True


def _render_depth(scene, cams, spec, dims, size):
    """Render a normalized depth map per view (near = bright) via the Z pass + Map Range, for
    ControlNet-depth conditioning. Map Range is set PER VIEW to the object's actual depth extent
    along that camera's axis (not the global size), so the relief fills the 0..1 range with high
    contrast instead of being crushed into a flat grey band."""
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)
    rl = tree.nodes.new("CompositorNodeRLayers")
    mr = tree.nodes.new("CompositorNodeMapRange")
    mr.inputs["To Min"].default_value = 1.0   # near -> bright
    mr.inputs["To Max"].default_value = 0.0   # far  -> dark
    mr.use_clamp = True
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Depth"], mr.inputs["Value"])
    tree.links.new(mr.outputs["Value"], comp.inputs["Image"])
    scene.render.film_transparent = False  # solid bg -> far -> black
    scene.render.image_settings.color_mode = "RGB"
    dist = size * 3 + 1  # matches make_ortho_camera (distance uses the global size for all views)
    for side, cam in cams.items():
        d = SIDE_DIR[side][0]
        depth_ext = abs(d.x) * dims.x + abs(d.y) * dims.y + abs(d.z) * dims.z  # object extent along view axis
        half = depth_ext / 2 * 1.08 + 1e-3
        mr.inputs["From Min"].default_value = dist - half
        mr.inputs["From Max"].default_value = dist + half
        scene.camera = cam
        scene.render.filepath = os.path.join(spec["geom_dir"], f"{spec['uid']}_depth_{side}.png")
        bpy.ops.render.render(write_still=True)


def render_geometry(spec):
    """Render a grey shaded view (relief cue) + a depth map (ControlNet-depth) of the bare mesh
    from each side camera."""
    obj = import_glb(spec["mesh"])
    mn, mx, center = bbox(obj)
    size = max((mx - mn).x, (mx - mn).y, (mx - mn).z)
    cams = setup_cameras([v["side"] for v in spec["views"]], center, size)
    scene = bpy.context.scene
    obj.data.materials.clear()
    grey = bpy.data.materials.new("grey")
    grey.use_nodes = True
    bsdf = grey.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1)
        bsdf.inputs["Roughness"].default_value = 1.0
    obj.data.materials.append(grey)
    _grey_lit_scene(scene)
    scene.render.resolution_x = scene.render.resolution_y = int(spec.get("geo_size", 1024))
    for side, cam in cams.items():
        scene.camera = cam
        scene.render.filepath = os.path.join(spec["geom_dir"], f"{spec['uid']}_geom_{side}.png")
        bpy.ops.render.render(write_still=True)
    if spec.get("depth", False):  # depth pass is unused by the gemini-colorize pipeline; off by default
        _render_depth(scene, cams, spec, mx - mn, size)
    print("BLENDER_GEOMETRY_DONE")


def bake_project(spec):
    """Project elevations and bake. If spec['base_glb'], overlay onto that Hunyuan-PBR GLB
    (keep its UVs + metallic/roughness, gap-fill uncovered faces with its albedo). Else
    smart-UV the bare mesh and bake elevations only."""
    overlay = bool(spec.get("base_glb"))
    # matte: drop the Hunyuan base metallic/roughness in the final material (metallic 0, full
    # roughness). The flat cartoon colourise has no metal; keeping the base's metallic makes the
    # cars/props mirror a viewer's IBL and blow out white. Default ON for the stylised look.
    matte = spec.get("matte", True)
    obj = import_glb(spec["base_glb"] if overlay else spec["mesh"])
    me = obj.data
    mn, mx, center = bbox(obj)
    size = max((mx - mn).x, (mx - mn).y, (mx - mn).z)
    tex_size = int(spec.get("tex_size", 2048))
    scene = bpy.context.scene

    # base material's albedo + metallic/roughness (overlay mode only)
    base_albedo = base_mr = base_principled = base_mat = None
    if overlay:
        for m in me.materials:
            if m and m.use_nodes:
                for n in m.node_tree.nodes:
                    if n.type == "BSDF_PRINCIPLED":
                        base_mat, base_principled = m, n
                        break
            if base_principled:
                break
        if base_principled:
            base_albedo = _image_into(base_principled, "Base Color")
            base_mr = _image_into(base_principled, "Metallic") or _image_into(base_principled, "Roughness")

    # bake-target UV: reuse the base UV in overlay mode (so MR stays aligned); else smart-UV.
    if overlay and me.uv_layers:
        bake_uv_name = me.uv_layers.active.name
    else:
        while len(me.uv_layers) > 0:
            me.uv_layers.remove(me.uv_layers[0])
        bake_uv_name = me.uv_layers.new(name="bake_uv").name
        me.uv_layers.active = me.uv_layers[bake_uv_name]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.005)
        bpy.ops.object.mode_set(mode="OBJECT")

    views = {v["side"]: v for v in spec["views"] if v["side"] in SIDE_DIR}
    sides = list(views.keys())
    cams = setup_cameras(sides, center, size)
    face_side = assign_faces(obj, me, cams, sides, min_dot=float(spec.get("face_dot", 0.5)),
                             occlude=spec.get("occlude", True))
    set_proj_uvs(obj, me, cams, face_side, views)

    # temp materials for baking: assigned faces emit their elevation; unassigned faces emit the
    # Hunyuan albedo (overlay) or neutral grey (standalone).
    me.materials.clear()
    side_idx = {}
    for i, side in enumerate(sides):
        img = bpy.data.images.load(views[side]["image"])
        me.materials.append(_emit_image_mat(f"emit_{side}", img, "proj_uv"))
        side_idx[side] = i
    if overlay and base_albedo is not None:
        fill = _emit_image_mat("emit_basealbedo", base_albedo, bake_uv_name)
    else:
        fill = _emit_color_mat("emit_neutral", (0.5, 0.5, 0.5, 1))
    me.materials.append(fill)
    fill_idx = len(sides)
    for fi, poly in enumerate(me.polygons):
        poly.material_index = side_idx.get(face_side[fi], fill_idx)

    # bake target image (baseColor) on the bake UV
    target = bpy.data.images.new("baked_albedo", tex_size, tex_size, alpha=False)
    for mat in me.materials:
        nt = mat.node_tree
        tnode = nt.nodes.new("ShaderNodeTexImage")
        tnode.image = target
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = bake_uv_name
        nt.links.new(uvn.outputs["UV"], tnode.inputs["Vector"])
        nt.nodes.active = tnode

    _cycles_gpu(scene)
    me.uv_layers.active = me.uv_layers[bake_uv_name]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", margin=max(2, tex_size // 256), use_clear=True)

    # final material: new baked albedo + (overlay) the original metallic/roughness
    me.materials.clear()
    fmat = bpy.data.materials.new("final_mat")
    fmat.use_nodes = True
    nt = fmat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 1.0
    try:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
    except Exception:
        pass
    albedo_node = nt.nodes.new("ShaderNodeTexImage")
    albedo_node.image = target
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = bake_uv_name
    nt.links.new(uvn.outputs["UV"], albedo_node.inputs["Vector"])
    nt.links.new(albedo_node.outputs["Color"], bsdf.inputs["Base Color"])
    try:
        bsdf.inputs["Metallic"].default_value = 0.0
    except Exception:
        pass
    if overlay and base_mr is not None and not matte:
        mrn = nt.nodes.new("ShaderNodeTexImage")
        mrn.image = base_mr
        for _cs in ("Non-Color", "Linear", "sRGB"):  # 3.0.1's minimal OCIO lacks "Non-Color"
            try:
                mrn.image.colorspace_settings.name = _cs
                break
            except Exception:
                continue
        muvn = nt.nodes.new("ShaderNodeUVMap")
        muvn.uv_map = bake_uv_name
        nt.links.new(muvn.outputs["UV"], mrn.inputs["Vector"])
        try:
            sep = nt.nodes.new("ShaderNodeSeparateColor")              # Blender >= 3.3
            g_out, b_out = sep.outputs["Green"], sep.outputs["Blue"]
        except RuntimeError:
            sep = nt.nodes.new("ShaderNodeSeparateRGB")                # Blender 3.0.x
            g_out, b_out = sep.outputs["G"], sep.outputs["B"]
        nt.links.new(mrn.outputs["Color"], sep.inputs[0])
        nt.links.new(g_out, bsdf.inputs["Roughness"])                  # glTF: roughness=G
        nt.links.new(b_out, bsdf.inputs["Metallic"])                   # glTF: metallic=B
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    me.materials.append(fmat)
    for layer in list(me.uv_layers):
        if layer.name != bake_uv_name:
            me.uv_layers.remove(layer)

    if spec.get("debug_dir"):
        # Best-effort verification renders; MUST NOT crash the bake/export. Workbench/EEVEE fail
        # headless on Blender 3.0.1 ("Unable to open a display"), so use Cycles + ambient world,
        # wrapped in try/except.
        try:
            _cycles_gpu(scene)
            scene.cycles.samples = 4
            if scene.world is None:
                scene.world = bpy.data.worlds.new("w")
            scene.world.use_nodes = True
            bg = scene.world.node_tree.nodes.get("Background")
            if bg:
                bg.inputs["Strength"].default_value = 1.3
            scene.render.image_settings.file_format = "PNG"
            scene.render.resolution_x = scene.render.resolution_y = 512
            scene.render.film_transparent = True
            cam34 = make_ortho_camera("cam_34", Vector((1, -1.4, 0.9)), Vector((0, 0, 1)), center, size)
            for nm, cam in list(cams.items()) + [("34", cam34)]:
                scene.camera = cam
                scene.render.filepath = os.path.join(spec["debug_dir"], f"blenderproj_cam_{nm}.png")
                bpy.ops.render.render(write_still=True)
        except Exception as e:  # noqa: BLE001
            print("debug renders skipped:", e)

    bpy.ops.export_scene.gltf(filepath=spec["out"], export_format="GLB",
                              export_image_format="AUTO", use_selection=False)
    print("BLENDER_PROJECT_DONE:", spec["out"])


def main():
    spec = json.load(open(sys.argv[sys.argv.index("--") + 1]))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if spec.get("mode") == "geometry":
        render_geometry(spec)
    else:
        bake_project(spec)


if __name__ == "__main__":
    main()
