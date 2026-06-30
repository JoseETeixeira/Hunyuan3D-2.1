"""CPU regression tests for the large-hole flat-fill that prevents cv2.INPAINT_NS from drawing a
wavy iso-contour "shadow gradient" fan over big empty UV islands (e.g. an unpainted wall).

Run:  pytest hy3dpaint/DifferentiableRenderer/test_uv_inpaint_fill.py

Imports the pure helper from uv_inpaint_fill (numpy + cv2 only) so the algorithm is verified without
the GPU renderer.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

try:
    from hy3dpaint.DifferentiableRenderer.uv_inpaint_fill import flat_fill_large_holes
except Exception:  # when run from inside the package dir
    from uv_inpaint_fill import flat_fill_large_holes


def _scene(cream=(212, 196, 160)):
    """A big central hole (a plain wall) ringed by uniform `cream` known texels, padded with a known
    grey frame so the only large hole is the wall. mask = 255 where known (NS's keep mask)."""
    H = W = 256
    src = np.zeros((H, W, 3), np.uint8)
    keep = np.zeros((H, W), np.uint8)
    cream = np.array(cream, np.uint8)
    keep[36:40, 36:204] = keep[200:204, 36:204] = 255
    keep[36:204, 36:40] = keep[36:204, 200:204] = 255
    src[keep > 0] = cream
    keep[:36, :] = keep[204:, :] = keep[:, :36] = keep[:, 204:] = 255
    src[(keep > 0) & (src.sum(2) == 0)] = np.array([90, 90, 90], np.uint8)
    hole = (255 - keep).astype(np.uint8)
    return src, keep, hole, cream


def test_large_hole_flat_filled_no_fan():
    src, keep, hole, cream = _scene()
    filled, residual = flat_fill_large_holes(src, keep, hole, min_px=4096)
    center = filled[60:180, 60:180].reshape(-1, 3)
    # The wall comes out uniform cream — no gradient.
    assert np.allclose(center.mean(0), cream, atol=2)
    assert center.std(0).max() < 1.0
    # The interior is removed from the NS hole (only a thin seam ring + small holes remain).
    assert (residual > 0).sum() < (hole > 0).sum() // 4
    # A full NS pass over the residual must not reintroduce a fan in the interior.
    out = cv2.inpaint(filled, residual, 3, cv2.INPAINT_NS)
    assert out[60:180, 60:180].reshape(-1, 3).std(0).max() < 1.5


def test_small_hole_untouched():
    # A hole below min_px must be left entirely to NS (helper returns the inputs unchanged).
    src, keep, _, _ = _scene()
    keep2 = np.full_like(keep, 255)
    keep2[120:124, 120:124] = 0  # 16-px scratch
    src2 = src.copy()
    hole2 = (255 - keep2).astype(np.uint8)
    filled, residual = flat_fill_large_holes(src2, keep2, hole2, min_px=4096)
    assert np.array_equal(filled, src2)
    assert np.array_equal(residual, hole2)


def test_no_known_texels_is_noop():
    src, _, hole, _ = _scene()
    keep0 = np.zeros(hole.shape, np.uint8)
    filled, residual = flat_fill_large_holes(src, keep0, hole, min_px=4096)
    assert np.array_equal(filled, src)
    assert np.array_equal(residual, hole)
