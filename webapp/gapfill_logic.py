"""Pure (numpy-only) selection logic for auto gap-fill camera placement.

Separated from the GPU bake in pipeline.py so the camera-ranking core can be unit-tested
without torch/CUDA or a loaded paint model.
"""
import numpy as np


def count_covered(normals, lookat, cos_thres):
    """How many surface normals (N,3) a camera with view direction `lookat` (3,) covers, using the
    SAME gate the renderer's bake applies: cos(lookat, normal) > cos_thres."""
    normals = np.asarray(normals, dtype=np.float32)
    if normals.size == 0:
        return 0
    return int((normals @ np.asarray(lookat, dtype=np.float32) > cos_thres).sum())


def best_candidate(normals, candidate_lookats, cos_thres):
    """Pick the candidate camera covering the most of `normals`.

    candidate_lookats: list of (key, lookat(3,)). Returns (key, score); (None, 0) when no candidate
    covers any normal. A candidate that covers 0 normals is never chosen, so the caller can stop the
    greedy loop once `best_candidate` returns a score below its min-texel threshold.
    """
    best_key, best_score = None, 0
    for key, lookat in candidate_lookats:
        s = count_covered(normals, lookat, cos_thres)
        if s > best_score:
            best_key, best_score = key, s
    return best_key, best_score
