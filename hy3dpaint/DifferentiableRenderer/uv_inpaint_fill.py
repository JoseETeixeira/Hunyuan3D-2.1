"""Large-hole flat-fill for UV-texture inpainting (numpy + cv2 only, no torch/nvdiffrast).

cv2.INPAINT_NS solves a harmonic (Laplace) fill: over a thin gutter it is invisible, but over a LARGE
empty UV island (a plain wall no view painted) it propagates the boundary colours inward as smooth
iso-contours -> a wavy "shadow gradient" fan. `flat_fill_large_holes` flat-fills big holes from their
nearest known texel first (uniform, no fan) so NS only blends the thin residual seam afterwards.

Kept in its own dep-light module so the behaviour is unit-testable on CPU without importing the GPU
renderer.
"""
import numpy as np
import cv2


def flat_fill_large_holes(src, keep_mask, hole, min_px):
    """Flat-fill large empty UV islands from their nearest KNOWN texel (Voronoi).

    src        uint8 HxWx3 texture
    keep_mask  uint8 HxW, 255 where the texel is trusted/known (NS's `mask`)
    hole       uint8 HxW, 255 where the texel must be filled (255 - keep_mask)
    min_px     a connected hole component is flat-filled only if it has more than this many texels;
               smaller holes (gutters, scratches) are left in `hole` for the normal NS pass.

    Returns (filled_src, residual_hole): residual_hole drops the flat-filled interiors and keeps a
    1px border ring around them so the subsequent NS pass only blends the seam, plus every small hole.
    """
    known = (keep_mask > 0).astype(np.uint8)
    if cv2.countNonZero(known) == 0:  # nothing to copy from -> leave it to NS
        return src, hole

    # Per-component gate: only big holes get the flat fill; small holes stay with NS.
    n, comp = cv2.connectedComponents((hole > 0).astype(np.uint8), connectivity=8)
    big = np.zeros(hole.shape, np.uint8)
    for lbl in range(1, n):
        m = comp == lbl
        if int(m.sum()) > min_px:
            big[m] = 255
    if cv2.countNonZero(big) == 0:
        return src, hole

    # Nearest known texel for every pixel via the distance-transform label trick. distanceTransform
    # measures distance to the nearest ZERO pixel, so feed it (known -> 0); DIST_LABEL_PIXEL tags each
    # pixel with the label of its nearest known texel. Map label -> that texel's colour.
    inv = np.where(known > 0, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ky, kx = np.where(known > 0)
    known_labels = labels[ky, kx]                       # label assigned to each known texel
    table = np.zeros((int(labels.max()) + 1, 2), np.int64)
    table[known_labels] = np.stack([ky, kx], axis=1)    # label -> source (y, x)
    nearest = table[labels]                             # HxWx2 source coords
    filled = src.copy()
    sel = big > 0
    filled[sel] = src[nearest[..., 0][sel], nearest[..., 1][sel]]

    # Keep a thin border ring of the flat-filled region in the residual hole so NS feathers the seam
    # between filled island and its real neighbours; clear the rest so NS does not re-fan the interior.
    ring = cv2.dilate(big, np.ones((3, 3), np.uint8)) & cv2.bitwise_not(big)
    residual = hole.copy()
    residual[big > 0] = 0
    residual[ring > 0] = 255
    return filled, residual
