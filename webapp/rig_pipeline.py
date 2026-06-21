"""UniRig auto-rigging orchestration for the per-model studio (step 3).

Runs UniRig (https://github.com/VAST-AI-Research/UniRig) as a subprocess in its own env (via
server._unirig_run): skeleton -> skin -> merge, producing a rigged GLB. Then dumps the predicted
skeleton (headless Blender) and maps a curated subset of joints onto the 12 named markers the user
edits. Re-skin feeds an edited skeleton back through skin + merge.

server is imported lazily inside functions (server imports studio imports this — no import cycle).
NOTE: the joint->marker name table is heuristic and may need tuning against real UniRig bone names;
left/right is split by world-X sign (x>0 = left), which is a convention to confirm on a GPU host.
"""
import json
import os

# Fixed 12 markers (keys mirror the frontend RigState).
MARKERS = [
    "groin", "chin",
    "shoulder-l", "shoulder-r", "elbow-l", "elbow-r", "hand-l", "hand-r",
    "knee-l", "knee-r", "ankle-l", "ankle-r",
]

# Substring synonyms per logical joint (matched against lowercased bone names).
_SYN = {
    "groin": ["hips", "pelvis", "hip", "root", "spine"],
    "chin": ["head", "neck", "jaw", "chin"],
    "shoulder": ["shoulder", "clavicle", "upperarm", "upper_arm", "arm_upper", "uparm"],
    "elbow": ["elbow", "forearm", "lowerarm", "lower_arm", "arm_lower"],
    "hand": ["hand", "wrist"],
    "knee": ["knee", "shin", "calf", "lowerleg", "lower_leg", "leg_lower"],
    "ankle": ["ankle", "foot", "heel"],
}
_PAIRED = ("shoulder", "elbow", "hand", "knee", "ankle")


# --------------------------------------------------------------------------- UniRig + Blender stages
def _skeleton(server, mesh_glb, out_fbx):
    server._unirig_run("launch/inference/generate_skeleton.sh",
                       ["--input", mesh_glb, "--output", out_fbx], "skeleton")
    if not os.path.exists(out_fbx):
        raise RuntimeError("UniRig produced no skeleton FBX")


def _skin(server, skeleton_fbx, out_fbx):
    server._unirig_run("launch/inference/generate_skin.sh",
                       ["--input", skeleton_fbx, "--output", out_fbx], "skin")
    if not os.path.exists(out_fbx):
        raise RuntimeError("UniRig produced no skin FBX")


def _merge(server, source_fbx, target_mesh, out_glb):
    server._unirig_run("launch/inference/merge.sh",
                       ["--source", source_fbx, "--target", target_mesh, "--output", out_glb], "merge")
    if not os.path.exists(out_glb):
        raise RuntimeError("UniRig merge produced no rigged GLB")


def _dump_skeleton(server, rigged_path, out_json):
    server._blender_python(server.BLENDER_DUMP_SKELETON_SCRIPT, [rigged_path, out_json],
                           "skeleton-dump", "SKELETON_DUMP_DONE")
    return json.loads(open(out_json, encoding="utf-8").read())["bones"]


def _edit_skeleton(server, skeleton_fbx, out_fbx, spec_path, moves):
    spec_path_obj = spec_path
    with open(spec_path_obj, "w", encoding="utf-8") as f:
        json.dump({"in": skeleton_fbx, "out": out_fbx, "moves": moves}, f)
    server._blender_python(server.BLENDER_EDIT_SKELETON_SCRIPT, [spec_path_obj],
                           "skeleton-edit", "SKELETON_EDIT_DONE")
    if not os.path.exists(out_fbx):
        raise RuntimeError("Edited skeleton FBX was not produced")


# --------------------------------------------------------------------------- marker mapping
def _match(name, subs):
    return any(s in name for s in subs)


# All positions are glTF Y-up (what the dump emits + the viewer/trimesh use): index 0 = lateral X,
# index 1 = vertical Y, index 2 = depth Z. Left/right is split by X sign (x>=0 = left) — a convention
# to confirm on a GPU host.
def _name_pass(bones):
    """Map by anatomical name substrings, for rigs that use semantic bone names. UniRig usually emits
    generic `bone_{i}` names, so this often yields little and the geometry pass fills the rest."""
    markers, mb = {}, {}
    named = [(b["name"].lower(), b) for b in bones]

    def pick(subs):
        return [b for (n, b) in named if _match(n, subs)]

    g = pick(_SYN["groin"])
    if g:
        b = min(g, key=lambda b: b["head"][1]); markers["groin"] = b["head"]; mb["groin"] = b["name"]
    c = pick(_SYN["chin"])
    if c:
        b = max(c, key=lambda b: b["head"][1]); markers["chin"] = b["head"]; mb["chin"] = b["name"]
    for joint in _PAIRED:
        cands = pick(_SYN[joint])
        for side, grp in (("l", [b for b in cands if b["head"][0] >= 0]),
                          ("r", [b for b in cands if b["head"][0] < 0])):
            if grp:
                b = max(grp, key=lambda b: abs(b["head"][0]))
                markers[f"{joint}-{side}"] = b["head"]; mb[f"{joint}-{side}"] = b["name"]
    return markers, mb


def _geometry_pass(bones):
    """Infer the 12 markers from skeleton topology + positions — robust to generic bone names.
    Heuristic; tune against real UniRig output."""
    by_name = {b["name"]: b for b in bones}
    children = {}
    for b in bones:
        if b.get("parent"):
            children.setdefault(b["parent"], []).append(b["name"])

    def parent(b):
        p = b.get("parent")
        return by_name.get(p) if p else None

    def is_leaf(b):
        return not children.get(b["name"])

    def up(b):
        return b["head"][1]

    def lat(b):
        return b["head"][0]

    ys = [up(b) for b in bones]
    ymin, ymax = min(ys), max(ys)
    span = max(ymax - ymin, 1e-6)
    leaves = [b for b in bones if is_leaf(b)]
    markers, mb = {}, {}

    def put(k, b):
        if b is not None:
            markers[k] = b["head"]; mb[k] = b["name"]

    def ancestors(b):
        out, cur = [], b
        while cur is not None:
            out.append(cur); cur = parent(cur)
        return out

    def between(leaf, top):
        """Joint on the chain strictly between `leaf` and `top`, nearest mid height."""
        if leaf is None or top is None:
            return None
        chain, cur = [], parent(leaf)
        while cur is not None and cur["name"] != top["name"]:
            chain.append(cur); cur = parent(cur)
        if not chain:
            return None
        tgt = (up(leaf) + up(top)) / 2.0
        return min(chain, key=lambda b: abs(up(b) - tgt))

    put("chin", max(bones, key=up))  # highest joint

    # ankles = lowest leaf per side; hands = most-lateral non-foot leaf per side.
    feet = [b for b in leaves if up(b) < ymin + 0.45 * span]
    al = min([b for b in feet if lat(b) >= 0], key=up, default=None)
    ar = min([b for b in feet if lat(b) < 0], key=up, default=None)
    put("ankle-l", al); put("ankle-r", ar)
    pool = [b for b in leaves if b not in feet]
    hl = max([b for b in pool if lat(b) >= 0], key=lambda b: abs(lat(b)), default=None)
    hr = max([b for b in pool if lat(b) < 0], key=lambda b: abs(lat(b)), default=None)
    put("hand-l", hl); put("hand-r", hr)

    # groin = lowest common ancestor of the two ankles (pelvis/leg split); else central mid joint.
    groin = None
    if al is not None and ar is not None:
        aset = {a["name"] for a in ancestors(al)}
        groin = next((a for a in ancestors(ar) if a["name"] in aset), None)
    if groin is None:
        central = [b for b in bones if abs(lat(b)) < 0.15 * span]
        groin = min(central, key=lambda b: abs(up(b) - (ymin + 0.5 * span)), default=None)
    put("groin", groin)

    put("knee-l", between(al, groin)); put("knee-r", between(ar, groin))

    def shoulder_of(hand):
        if hand is None:
            return None
        anc = ancestors(hand)[1:]  # drop the hand
        lateral = [a for a in anc if abs(lat(a)) > 0.1 * span]
        return lateral[-1] if lateral else (anc[0] if anc else None)

    sl = shoulder_of(hl); sr = shoulder_of(hr)
    put("shoulder-l", sl); put("shoulder-r", sr)
    put("elbow-l", between(hl, sl)); put("elbow-r", between(hr, sr))
    return markers, mb


def map_markers(bones):
    """Map predicted bones -> {marker: [x,y,z]} + {marker: boneName}. Tries semantic names first
    (when the rig has them), then fills missing markers from skeleton geometry/topology. Unmapped
    markers are omitted. Never raises — a weird skeleton just yields fewer markers."""
    try:
        markers, marker_bones = _name_pass(bones)
    except Exception:  # noqa: BLE001
        markers, marker_bones = {}, {}
    if len(markers) < len(MARKERS):
        try:
            gm, gb = _geometry_pass(bones)
            for k, v in gm.items():
                markers.setdefault(k, v)
                if k not in marker_bones and k in gb:
                    marker_bones[k] = gb[k]
        except Exception:  # noqa: BLE001
            pass
    return markers, marker_bones


# --------------------------------------------------------------------------- public API
def run_full(mesh_glb, skeleton_fbx, skin_fbx, rigged_glb, skel_json):
    """Full auto-rig: skeleton -> skin -> merge -> dump -> map markers. Returns
    {markers, markerBones}. Writes rigged_glb + intermediates to the given paths."""
    from webapp import server
    _skeleton(server, mesh_glb, skeleton_fbx)
    _skin(server, skeleton_fbx, skin_fbx)
    _merge(server, skin_fbx, mesh_glb, rigged_glb)
    bones = _dump_skeleton(server, rigged_glb, skel_json)
    markers, marker_bones = map_markers(bones)
    return {"markers": markers, "markerBones": marker_bones}


def run_reskin(mesh_glb, skeleton_fbx, edited_fbx, skin_fbx, rigged_glb, spec_path, moves):
    """Re-skin after marker edits: edit the skeleton to the new joint positions, then skin + merge.
    `moves` is {boneName: [x,y,z]} in the rigged file's coordinate space."""
    from webapp import server
    _edit_skeleton(server, skeleton_fbx, edited_fbx, spec_path, moves)
    _skin(server, edited_fbx, skin_fbx)
    _merge(server, skin_fbx, mesh_glb, rigged_glb)


def recenter(mesh_glb, point, normal):
    """Place a joint at the limb's center: cast a ray from the clicked surface point inward along
    -normal through the mesh and return the midpoint of the entry/exit hits. CPU (trimesh)."""
    import numpy as np
    import trimesh
    m = trimesh.load(mesh_glb, force="mesh")
    p = np.asarray(point, dtype=float)
    n = np.asarray(normal, dtype=float)
    nlen = float(np.linalg.norm(n))
    if nlen < 1e-9:
        return [float(p[0]), float(p[1]), float(p[2])]
    n = n / nlen
    origin = p + n * 1e-3  # nudge just outside the surface so the clicked face counts as a hit
    locs, _, _ = m.ray.intersects_location([origin], [-n])
    if len(locs) == 0:
        return [float(p[0]), float(p[1]), float(p[2])]
    depth = (locs - origin) @ (-n)
    near = locs[int(np.argmin(depth))]
    far = locs[int(np.argmax(depth))]
    mid = (near + far) / 2.0
    return [float(mid[0]), float(mid[1]), float(mid[2])]
