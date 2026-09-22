"""Surface geometry: tangent frames and the curvature direction field.

Moved verbatim from bake_indirection.py; the only change is that module-level config
globals became explicit `cfg` parameters. Pure numpy -- no bpy.
"""

import numpy as np

def quat_to_TB(q, cfg):
    """(N,4) quaternions -> per-seed tangent T and bitangent B (rot-matrix cols 0,1)."""
    if cfg.quat_wxyz:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    T = np.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y)], -1)
    B = np.stack([2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x)], -1)
    return T.astype(np.float64), B.astype(np.float64)


def frame_from_tangent(N, Tin):
    """Build an orthonormal in-plane frame from a normal and a desired tangent.

    The tangent is projected into the surface plane and normalised, so a GN tangent
    that is only approximately perpendicular to the normal still yields a valid frame.
    Degenerate inputs (zero length, or parallel to the normal) fall back to an
    arbitrary perpendicular rather than producing NaNs.
    """
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    T = Tin - (Tin * N).sum(1)[:, None] * N          # project into the tangent plane
    L = np.linalg.norm(T, axis=1)
    bad = L < 1e-9
    if bad.any():                                    # arbitrary perpendicular fallback
        alt = np.where(np.abs(N[bad, 0:1]) < 0.9,
                       np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        Tb = np.cross(N[bad], alt)
        T[bad] = Tb
        L[bad] = np.linalg.norm(Tb, axis=1)
    T = T / np.maximum(L, 1e-12)[:, None]
    B = np.cross(N, T)                               # right-handed, unit by construction
    return T.astype(np.float64), B.astype(np.float64)


def local_frame(N):
    """Per-row orthonormal (N, e1, e2); e1/e2 span the plane perpendicular to N."""
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    alt = np.where(np.abs(N[:, 0:1]) < 0.9,
                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    e1 = np.cross(N, alt)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    return N, e1, np.cross(N, e1)


def knn_indices(P, K, max_bytes):
    """K nearest SEEDS for every seed (neighbours are seeds, never texels).

    Chunked the same way as the DEBUG_VORONOI path: the distance matrix is
    (rows x seeds), so bounding rows alone is not a memory bound -- it grows with seed
    count too. Rows per batch are derived from the actual seed count.
    """
    n = len(P)
    K = min(K, n - 1)
    rows = max(1, min(n, int(max_bytes // (n * 8))))
    if not np.isfinite(P).all():
        raise ValueError(
            "seed positions contain NaN or inf -- the direction field cannot be fitted. "
            "This usually means a degenerate face produced a bad stroke position.")
    p2 = (P * P).sum(1)
    nb = np.empty((n, K), np.int64)
    for a in range(0, n, rows):
        b = min(a + rows, n)
        # numpy reports spurious invalid/overflow warnings from BLAS matmul on some
        # builds -- the check above rules out a real NaN, so the warning is noise and
        # was appearing in the console next to unrelated errors.
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            d = P[a:b] @ P.T                # in place, as in resolve_uv
        d *= -2.0
        d += p2[None, :]
        d[np.arange(b - a), np.arange(a, b)] = np.inf      # exclude self
        nb[a:b] = np.argpartition(d, K, axis=1)[:, :K]
    return nb


def principal_min_direction(P, N, nb, w):
    """Least-curved direction per seed, from the shape operator (Weingarten map).

    For each seed: express neighbour offsets `u` and neighbour normal deltas `v` in the
    local tangent frame, then fit the 2x2 matrix S with v ~ u.S -- "step this way, the
    normal tilts that way". Its eigenvectors are the principal directions and its
    eigenvalues the principal curvatures; the eigenvector with the smaller |eigenvalue|
    is the direction along which the normal changes least.

    `w` is a 0/1 weight per neighbour. A zeroed row contributes nothing to either u^T u
    or u^T v, so masking is exactly "drop that neighbour" with no special-casing.

    Fully vectorised: the 2x2 solve is done in closed form rather than per-seed lstsq.
    """
    n = len(P)
    Nn, e1, e2 = local_frame(N)
    dp = P[nb] - P[:, None, :]                       # (n,K,3) tangential offsets
    dn = N[nb] - Nn[:, None, :]                      # (n,K,3) normal variation
    ww = w[..., None]
    u = np.stack([(dp * e1[:, None, :]).sum(-1), (dp * e2[:, None, :]).sum(-1)], -1) * ww
    v = np.stack([(dn * e1[:, None, :]).sum(-1), (dn * e2[:, None, :]).sum(-1)], -1) * ww

    A = np.einsum('nki,nkj->nij', u, u)              # u^T u   (n,2,2)
    Bm = np.einsum('nki,nkj->nij', u, v)             # u^T v   (n,2,2)
    det = A[:, 0, 0]*A[:, 1, 1] - A[:, 0, 1]*A[:, 1, 0]
    tr = A[:, 0, 0] + A[:, 1, 1]
    bad = np.abs(det) <= 1e-12 * np.maximum(tr*tr, 1e-30)   # collinear/degenerate patch
    safe = np.where(bad, 1.0, det)
    Ai = np.empty_like(A)
    Ai[:, 0, 0] =  A[:, 1, 1] / safe
    Ai[:, 1, 1] =  A[:, 0, 0] / safe
    Ai[:, 0, 1] = -A[:, 0, 1] / safe
    Ai[:, 1, 0] = -A[:, 1, 0] / safe
    S = Ai @ Bm
    S = 0.5 * (S + S.transpose(0, 2, 1))             # true shape operator is symmetric
    ev, evec = np.linalg.eigh(S)
    k = np.argmin(np.abs(ev), axis=1)                # smaller |curvature| = least bending
    sel = evec[np.arange(n), :, k]                   # (n,2) in local-frame coords
    D = sel[:, 0:1] * e1 + sel[:, 1:2] * e2
    D[bad] = e1[bad]                                 # degenerate -> arbitrary in-plane
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    return D, int(bad.sum())


def smooth_line_field(T, N, nb, w, iters, mix):
    """Diffuse the per-seed directions into a coherent field.

    Two traps this exists to avoid:
      * Direction is modulo 180 deg -- T and -T are the same stroke -- so averaging raw
        vectors lets agreeing neighbours cancel to zero. Fixed by the double-angle
        representation (cos 2t, sin 2t), in which T and -T are the SAME point.
      * Every seed has its own tangent frame, so bare angles are not comparable. Each
        neighbour's direction is projected into seed i's plane and measured against i's
        e1 first -- a discrete stand-in for parallel transport.

    Iterating many small passes (rather than using one huge K) propagates confident
    directions outward while leaving real feature structure intact. Umbilic regions, which
    have no direction of their own, simply inherit from their neighbours.
    """
    Nn, e1, e2 = local_frame(N)
    ok_w = w > 0
    for _ in range(iters):
        Tj = T[nb]                                            # (n,K,3)
        Tp = Tj - (Tj * Nn[:, None, :]).sum(-1)[..., None] * Nn[:, None, :]
        c = (Tp * e1[:, None, :]).sum(-1)                     # into i's frame
        sN = (Tp * e2[:, None, :]).sum(-1)
        L = np.hypot(c, sN)
        ok = ok_w & (L > 1e-12)
        inv = np.where(ok, 1.0 / np.maximum(L, 1e-12), 0.0)
        c = c * inv
        sN = sN * inv
        c2 = c*c - sN*sN                                      # double angle -> T == -T
        s2 = 2.0 * c * sN
        cnt = ok.sum(1)
        oc = (T * e1).sum(1)
        os_ = (T * e2).sum(1)
        ol = np.maximum(np.hypot(oc, os_), 1e-12)
        oc /= ol
        os_ /= ol
        oc2 = oc*oc - os_*os_
        os2 = 2.0 * oc * os_
        has = cnt > 0
        avc = np.where(has, c2.sum(1) / np.maximum(cnt, 1), oc2)
        avs = np.where(has, s2.sum(1) / np.maximum(cnt, 1), os2)
        th = 0.5 * np.arctan2((1-mix)*os2 + mix*avs, (1-mix)*oc2 + mix*avc)
        T = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2   # in-plane and unit
    return T


def compute_direction_field(P, N, cfg):
    """Per-seed stroke direction along the surface's least-curved direction."""
    P = P.astype(np.float64)
    N = N.astype(np.float64)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    nb = knn_indices(P, cfg.curv_k, cfg.curv_max_bytes)
    dots = (N[nb] * N[:, None, :]).sum(-1)
    w = (dots >= cfg.curv_normal_min).astype(np.float64)
    thin = w.sum(1) < cfg.curv_min_nb          # too few left: keep them all rather than fail
    w[thin] = 1.0
    T, n_bad = principal_min_direction(P, N, nb, w)
    T = smooth_line_field(T, N, nb, w, cfg.curv_smooth_iters, cfg.curv_smooth_mix)
    stats = dict(dropped=int((w == 0).sum()), relaxed=int(thin.sum()), degenerate=n_bad,
                 K=min(cfg.curv_k, len(P) - 1))
    return T, stats
