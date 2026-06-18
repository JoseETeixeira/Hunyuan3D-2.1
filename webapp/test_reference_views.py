"""Pure unit tests for the staged reference-view prompts + dependency graph (no API calls).

Run:  python webapp/test_reference_views.py     (or via pytest)
"""
from webapp import reference_views as rv


def test_view_inputs_graph():
    assert rv.VIEW_INPUTS["front"] == []
    assert rv.VIEW_INPUTS["back"] == ["front"]
    assert rv.VIEW_INPUTS["top"] == ["front"]
    assert rv.VIEW_INPUTS["bottom"] == ["front"]
    # left/right need only front + top (the side doesn't show the back)
    assert rv.VIEW_INPUTS["left"] == ["front", "top"]
    assert rv.VIEW_INPUTS["right"] == ["front", "top"]
    assert rv.VIEW_INPUTS["front-left"] == ["front", "left", "top"]
    assert rv.VIEW_INPUTS["front-right"] == ["front", "right", "top"]
    assert rv.VIEW_INPUTS["back-left"] == ["back", "left", "top"]
    assert rv.VIEW_INPUTS["back-right"] == ["back", "right", "top"]
    assert set(rv.ALL_VIEWS) == set(rv.VIEW_INPUTS)
    assert len(rv.ALL_VIEWS) == 10


def test_view_tag_mapping():
    assert rv.VIEW_TO_TAG["front-left"] == "fl"
    assert rv.VIEW_TO_TAG["front-right"] == "fr"
    assert rv.VIEW_TO_TAG["back-left"] == "bl"
    assert rv.VIEW_TO_TAG["back-right"] == "br"
    assert rv.VIEW_TO_TAG["front"] == "front"


def test_prompt_required_clauses():
    # generic: no model-type-specific hardcoding (the dealership wording must be gone)
    forbidden = ("parked cars", "lamp post", "front lot", "facade", "dealership", "building")
    for v in rv.ALL_VIEWS:
        p = rv.build_prompt(v, 1).lower()
        assert "orthograph" in p, v
        assert "cartoon" in p, v
        # regression: never strip the model to a bare object
        assert "no ground plane" not in p and "no scenery" not in p and "single centered object" not in p, v
        for term in forbidden:
            assert term not in p, (v, term)


def test_per_view_visibility_rules():
    # front keeps any surrounding elements in their real positions
    front = rv.build_prompt("front", 1).lower()
    assert "in front of it" in front and "positions" in front
    # back and the back-corners exclude front-only elements
    for v in ("back", "back-left", "back-right"):
        assert "not visible from behind" in rv.build_prompt(v, 1).lower(), v
    # bottom = underside only
    assert "underside" in rv.build_prompt("bottom", 1).lower()
    # top = strict top-down plan with no vertical faces
    top = rv.build_prompt("top", 1).lower()
    assert "top-down" in top and "do not show any vertical" in top
    # side views: the front must not face the camera
    for v in ("left", "right"):
        assert "must not face the camera" in rv.build_prompt(v, 1).lower(), v


def test_corner_handedness():
    # every 3/4 corner anchors handedness (TOP map + anti-mirror) and names the correct visible side
    for v in ("front-left", "front-right", "back-left", "back-right"):
        p = rv.build_prompt(v, 5).lower()
        assert "do not mirror" in p, v
        assert "handedness" in p, v
    assert "left side" in rv.build_prompt("front-left", 5).lower()
    assert "right side" in rv.build_prompt("front-right", 5).lower()
    assert "left side" in rv.build_prompt("back-left", 5).lower()
    assert "right side" in rv.build_prompt("back-right", 5).lower()


def test_corner_rotation_directions():
    assert "front rotated 45 degrees counter-clockwise" in rv.build_prompt("front-left", 4).lower()
    assert "front rotated 45 degrees clockwise" in rv.build_prompt("front-right", 4).lower()
    assert "back rotated 45 degrees clockwise" in rv.build_prompt("back-left", 4).lower()
    assert "back rotated 45 degrees counter-clockwise" in rv.build_prompt("back-right", 4).lower()


def test_ground_aligned_side_cameras():
    # the four side views use a ground-aligned, untilted camera; floor shows only in the top view
    for v in ("front", "back", "left", "right"):
        p = rv.build_prompt(v, 1).lower()
        assert "perfectly horizontal" in p and "ground level" in p, v
        assert "only" in p and "top view" in p, v
    # top/bottom and the 3/4 corners do NOT get the ground-aligned side-camera clause
    for v in ("top", "bottom", "front-left", "back-right"):
        assert "perfectly horizontal" not in rv.build_prompt(v, 1).lower(), v


def test_rotation_consistency_and_tweak():
    back2 = rv.build_prompt("back", 2).lower()
    left2 = rv.build_prompt("left", 2).lower()
    right2 = rv.build_prompt("right", 2).lower()
    assert "rotated 180" in back2
    # left/right state the rotation direction AND the reference frame (seen from directly above)
    assert "rotated 90 degrees clockwise" in right2 and "from directly above" in right2
    assert "rotated 90 degrees counter-clockwise" in left2 and "from directly above" in left2
    # seed passed AND used as the layout authority when generating from front
    assert "already-approved" not in rv.build_prompt("front", 1).lower()
    assert "already-approved" in left2 and "authority" in left2 and "seed" in left2
    assert "make the roof blue" in rv.build_prompt("front", 1, "make the roof blue")


def test_cardinal_exclusivity():
    # the six cardinal views each show ONLY their one face — no other view, no 3/4 angle
    for v in ("front", "back", "left", "right", "top", "bottom"):
        p = rv.build_prompt(v, 1).lower()
        assert "render exactly one canonical" in p, v
        assert "no 3/4" in p, v
    # the 3/4 corners are NOT single-face, so the exclusivity clause must NOT apply
    for v in ("front-left", "front-right", "back-left", "back-right"):
        assert "render exactly one canonical" not in rv.build_prompt(v, 1).lower(), v


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all reference_views tests passed")
