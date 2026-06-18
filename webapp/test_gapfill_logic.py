"""Pure unit tests for the gap-fill camera-selection core (no torch / no GPU).

Run directly:  python webapp/test_gapfill_logic.py
Or via pytest:  pytest webapp/test_gapfill_logic.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from webapp.gapfill_logic import best_candidate, count_covered  # noqa: E402

COS75 = math.cos(math.radians(75.0))


def test_count_covered_alignment():
    # +X normals are covered by a +X lookat, not by a +Y lookat.
    normals = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
    assert count_covered(normals, [1, 0, 0], COS75) == 3
    assert count_covered(normals, [0, 1, 0], COS75) == 0
    assert count_covered(np.empty((0, 3), np.float32), [1, 0, 0], COS75) == 0


def test_count_covered_threshold():
    # A normal 70deg off lookat is covered (>cos75); 80deg off is not.
    n70 = np.array([[math.cos(math.radians(70)), math.sin(math.radians(70)), 0]], np.float32)
    n80 = np.array([[math.cos(math.radians(80)), math.sin(math.radians(80)), 0]], np.float32)
    assert count_covered(n70, [1, 0, 0], COS75) == 1
    assert count_covered(n80, [1, 0, 0], COS75) == 0


def test_best_candidate_picks_max():
    # 5 normals face +X, 1 faces +Y -> the +X camera wins.
    normals = np.array([[1, 0, 0]] * 5 + [[0, 1, 0]], dtype=np.float32)
    cands = [("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])]
    key, score = best_candidate(normals, cands, COS75)
    assert key == "x" and score == 5


def test_best_candidate_none_when_uncovered():
    # Normals face -X; no candidate (which all face away) covers them.
    normals = np.array([[-1, 0, 0]] * 4, dtype=np.float32)
    cands = [("x", [1, 0, 0]), ("y", [0, 1, 0])]
    key, score = best_candidate(normals, cands, COS75)
    assert key is None and score == 0


def test_greedy_cap_and_stop():
    # Simulate the pipeline's selection policy against synthetic clusters: greedy pick by coverage,
    # stop at the cap or when the best candidate covers < min_texels.
    clusters = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    normals = np.array([clusters["x"]] * 10 + [clusters["y"]] * 8 + [clusters["z"]] * 2, dtype=np.float32)
    cands = [(k, v) for k, v in clusters.items()]
    max_cams, min_texels = 6, 5

    remaining = normals.copy()
    tried, chosen = set(), []
    while len(chosen) < max_cams:
        key, score = best_candidate(remaining, [(k, L) for k, L in cands if k not in tried], COS75)
        if key is None or score < min_texels:
            break
        tried.add(key)
        L = np.array(clusters[key], np.float32)
        keep = remaining @ L <= COS75            # drop the texels this camera covered
        remaining = remaining[keep]
        chosen.append(key)

    # x (10) and y (8) clear the min; z has only 2 (< min_texels) -> excluded by early-stop.
    assert chosen == ["x", "y"]
    assert len(remaining) == 2  # the z normals remain uncovered


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"[test_gapfill_logic] {fn.__name__} OK")
    print(f"[test_gapfill_logic] {len(fns)} passed")


if __name__ == "__main__":
    _run()
