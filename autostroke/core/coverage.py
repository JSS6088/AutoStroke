"""Predict coverage in a second, instead of finding it out after a full bake.

The trick is not a formula. It runs the REAL stamp test -- same footprint, same brush
mask, same crease gate, same candidate order -- against a few thousand random points on
the surface rather than every texel. Cost is proportional to points x strokes, so a few
thousand points is ~100x cheaper than 2.9M texels while landing within a point of the
answer (measured: 82.2 vs 81.4, 93.6 vs 93.2, 50.0 vs 48.9).

It samples the mesh, not the position map, so it works before anything has been baked.
"""

import numpy as np


def sample_surface(P, Q, R, n, seed=0):
    """`n` area-weighted random points on a triangle soup, and their normals.

    Area-weighted so the estimate matches what a texel grid sees: every unit of surface
    is equally likely to be probed, regardless of how the mesh is cut up. The normals come
    back because the crease guard needs them -- without them the preview would skip the
    guard and over-report.
    """
    P = np.asarray(P, np.float64)
    Q = np.asarray(Q, np.float64)
    R = np.asarray(R, np.float64)
    area = 0.5 * np.linalg.norm(np.cross(Q - P, R - P), axis=1)
    tot = area.sum()
    if tot <= 0 or len(area) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    rng = np.random.default_rng(seed)
    t = rng.choice(len(area), size=n, p=area / tot)
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    fold = (u + v) > 1.0            # reflect into the triangle: uniform by area
    u[fold] = 1.0 - u[fold]
    v[fold] = 1.0 - v[fold]
    pts = P[t] + u * (Q[t] - P[t]) + v * (R[t] - P[t])
    nrm = np.cross(Q - P, R - P)[t]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pts.astype(np.float32), nrm.astype(np.float32)


def estimate_coverage(points, seeds, mask, cfg, seed_tan=None, stamp_rot=0.0,
                      pt_nrm=None):
    """Fraction of `points` a stamp would claim. Returns (coverage, n_points)."""
    from . import baker
    n = len(points)
    if n == 0 or len(seeds["position"]) == 0:
        return 0.0, 0
    res = baker.resolve_uv_blocking(
        points, np.zeros((n, 2), np.float32),
        seeds["position"], seeds["id"], seeds["UVMap"], seeds["radius"],
        seeds["normal"], None, mask, cfg, seed_tan, stamp_rot, pt_nrm=pt_nrm)
    return float(res[2].mean()), n
