"""Reachable surface: which triangles each stroke may paint.

The stroke search's own tests -- sphere, painted box, brush mask, crease -- are all
measured in 3D or in the stroke's own plane. None of them knows which SURFACE a stroke was
placed on, so a stroke reached any nearby surface facing the same way: a second layer a few
millimetres away, a panel floating over a hull, an eyeball in its socket, the far side of a
fold. The normal map mostly hid it (the stolen normal is nearly right); the indirection
map did not, because colour read through the pointer came from the wrong part.

The rule added here: a stroke may only paint triangles it can reach ACROSS THE SURFACE --
triangles inside its sphere that connect back to its own triangle through triangles that
are also inside the sphere. Two stacked layers are both inside the sphere but share no
path; the two sides of a fold connect only around the fold, which is outside it.

Connectivity joins triangles that share a vertex POSITION, not a vertex index. Imported
meshes routinely split vertices along UV seams and hard edges; joining by index would cut
the surface into its UV islands and stop strokes at every seam -- the exact artifact this
tool exists to remove.

Everything here is pure numpy and runs once per (mesh, stroke set), cached by the caller.
The result is handed to both resolvers in the form each one wants:
  - per TRIANGLE, the strokes that can reach it  -> the GPU search's candidate lists;
  - per STROKE, the triangles it can reach       -> the CPU resolver's extra test.
Both views are cut from the same (stroke, triangle) pairs, so the two resolvers cannot
disagree about reach.
"""

import numpy as np

MAX_PAIRS = 60_000_000
"""Refuse beyond this many (stroke, triangle) pairs rather than exhaust memory. At 12 bytes a
pair in flight that is under 1 GB; real meshes measured at a few million."""


class ReachError(RuntimeError):
    """The mesh/stroke combination is too large to build reach sets for."""


# ---------------------------------------------------------------------------
# The surface graph
# ---------------------------------------------------------------------------

def weld(P, Q, R, tol=1e-6):
    """(T, 3) welded vertex id per triangle corner: corners at the same POSITION get the
    same id, whatever their vertex index. `tol` is relative to the bounding-box diagonal.

    Keyed as one int64 per corner (21 bits an axis), which unique()s far faster than a
    row-wise unique on a (3T, 3) array."""
    C = np.concatenate([P, Q, R]).astype(np.float64)
    lo = C.min(0)
    diag = float(np.linalg.norm(C.max(0) - lo))
    step = max(diag * tol, 1e-30)
    q = np.round((C - lo) / step).astype(np.int64)
    q = np.minimum(q, (1 << 21) - 1)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    _, vid = np.unique(key, return_inverse=True)
    return vid.reshape(3, -1).T


def tri_adjacency(tri_vid):
    """CSR (start, nbr): triangles are neighbours if they share a welded VERTEX.

    Vertex rather than edge sharing tolerates imperfect topology -- fans, T-junctions,
    triangles meeting at a single corner -- that should still read as one surface."""
    T = len(tri_vid)
    v = tri_vid.ravel()
    t = np.repeat(np.arange(T, dtype=np.int64), 3)
    o = np.argsort(v, kind="stable")
    v, t = v[o], t[o]
    # every (t_a, t_b) pair within each vertex's group, by the repeat/offset trick
    gstart = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    gsize = np.diff(np.r_[gstart, len(v)])
    size_of = np.repeat(gsize, gsize)
    first_of = np.repeat(gstart, gsize)
    a = np.repeat(t, size_of)
    off = np.arange(int(size_of.sum())) - np.repeat(np.cumsum(size_of) - size_of, size_of)
    b = t[np.repeat(first_of, size_of) + off]
    keep = a != b
    key = np.unique(a[keep] * T + b[keep])
    a, b = key // T, key % T
    start = np.searchsorted(a, np.arange(T + 1))
    return start, b


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def point_tri_dist2(p, A, B, C):
    """Squared distance from each point p[i] to triangle (A[i], B[i], C[i]), exactly.

    Ericson's closest-point-on-triangle, region by region, vectorised. Exact rather than a
    centroid/circumradius bound: on a low-poly mesh a big triangle's slack could reach
    across a fold and join two surfaces that should stay apart."""
    p, A, B, C = (np.asarray(x, np.float64) for x in (p, A, B, C))
    ab, ac, ap = B - A, C - A, p - A
    bp, cp = p - B, p - C
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        e_ab = d1 / (d1 - d3)
        e_ac = d2 / (d2 - d6)
        e_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        den = va + vb + vc
        iv, iw = vb / den, vc / den
    conds = [
        (d1 <= 0) & (d2 <= 0),                                  # vertex A
        (d3 >= 0) & (d4 <= d3),                                 # vertex B
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),                      # edge AB
        (d6 >= 0) & (d5 <= d6),                                 # vertex C
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),                      # edge AC
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),        # edge BC
    ]
    v = np.select(conds, [0.0, 1.0, e_ab, 0.0, 0.0, 1.0 - e_bc], default=iv)
    w = np.select(conds, [0.0, 0.0, 0.0, 1.0, e_ac, e_bc], default=iw)
    v, w = np.nan_to_num(v), np.nan_to_num(w)       # degenerate triangles: nearest corner
    q = A + ab * v[:, None] + ac * w[:, None]
    return ((p - q) ** 2).sum(1)


# ---------------------------------------------------------------------------
# Reach
# ---------------------------------------------------------------------------

def reachable(seed_tri, seed_pos, r, P, Q, R, adj):
    """Sorted int64 keys `stroke * T + tri`: every triangle each stroke can reach.

    A flood fill from each stroke's own triangle, all strokes at once: each ring expands
    the frontier to its neighbours, keeps those whose closest point is inside the stroke's
    sphere (distance < r), and drops pairs already visited. Stops when no frontier is left.

    The stroke's own triangle is always included, whatever its size -- the sphere may sit
    entirely inside a large triangle and still paint it."""
    start, nbr = adj
    T = len(P)
    r2 = np.asarray(r, np.float64) ** 2
    S = np.arange(len(seed_tri), dtype=np.int64)
    Fc = np.asarray(seed_tri, np.int64)
    visited = np.unique(S * T + Fc)
    S, Fc = visited // T, visited % T
    while len(S):
        cnt = start[Fc + 1] - start[Fc]
        total = int(cnt.sum())
        if total == 0:
            break
        if len(visited) + total > MAX_PAIRS:
            raise ReachError(
                "Too many stroke-surface pairs to track (over %s). Lower Stroke Count or "
                "Stroke Size, or bake a lighter mesh." % format(MAX_PAIRS, ","))
        s2 = np.repeat(S, cnt)
        off = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        f2 = nbr[np.repeat(start[Fc], cnt) + off]
        k = np.unique(s2 * T + f2)
        k = k[~np.isin(k, visited, assume_unique=True)]
        if not len(k):
            break
        s2, f2 = k // T, k % T
        ok = point_tri_dist2(seed_pos[s2], P[f2], Q[f2], R[f2]) < r2[s2]
        k = k[ok]
        visited = np.union1d(visited, k)
        S, Fc = k // T, k % T
    return visited


def tri_lists(keys, n_tris):
    """Per triangle, the strokes that can reach it: (start, count, strokes), with triangle
    t's strokes at strokes[start[t] : start[t] + count[t]]. The GPU search's candidates."""
    t = keys % n_tris
    s = keys // n_tris
    o = np.argsort(t, kind="stable")
    counts = np.bincount(t, minlength=n_tris)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return starts, counts, s[o]


def stroke_tris(keys, n_tris, n_strokes):
    """Per stroke, the SORTED triangles it can reach: (start, tris), stroke i's at
    tris[start[i] : start[i + 1]]. Keys are already sorted by (stroke, tri)."""
    s = keys // n_tris
    return np.searchsorted(s, np.arange(n_strokes + 1)), keys % n_tris


def in_sorted(values, sorted_arr):
    """Boolean membership of each value in a sorted array (searchsorted, no hashing)."""
    if not len(sorted_arr):
        return np.zeros(len(values), bool)
    i = np.searchsorted(sorted_arr, values).clip(0, len(sorted_arr) - 1)
    return sorted_arr[i] == values


def build(P, Q, R, seed_tri, seed_pos, r):
    """Everything both resolvers need, from one set of (stroke, triangle) pairs."""
    P, Q, R = (np.asarray(x, np.float64) for x in (P, Q, R))
    T = len(P)
    adj = tri_adjacency(weld(P, Q, R))
    keys = reachable(np.asarray(seed_tri, np.int64), np.asarray(seed_pos, np.float64),
                     r, P, Q, R, adj)
    t_start, t_count, t_strokes = tri_lists(keys, T)
    s_start, s_tris = stroke_tris(keys, T, len(seed_tri))
    return dict(n_tris=T, pairs=len(keys),
                tri_start=t_start, tri_count=t_count, tri_strokes=t_strokes,
                stroke_start=s_start, stroke_tris=s_tris)


def build_for_seeds(seeds, cfg):
    """build() for a seed set from bridge/seeds.build_seeds, which keeps each stroke's
    triangle (`_tri`) and the triangle corners (`_mesh`). Each stroke is sized with
    baker.effective_radius -- the radius the search itself paints with."""
    from .baker import effective_radius             # baker imports this module
    m = seeds["_mesh"]
    r = effective_radius(seeds["radius"], np.asarray(seeds["id"], np.int64), cfg)
    return build(m["P"], m["Q"], m["R"], seeds["_tri"], seeds["position"], r)


# ---------------------------------------------------------------------------
# Which triangle each texel belongs to
# ---------------------------------------------------------------------------

def _pieces(x0, x1, y0, y1, budget):
    """Split each triangle's pixel box into row bands of at most `budget` pixels, so one
    huge triangle (a two-triangle plane covering the whole map) cannot blow memory."""
    w = np.maximum(x1 - x0 + 1, 0)
    h = np.maximum(y1 - y0 + 1, 0)
    rows = np.maximum(budget // np.maximum(w, 1), 1)
    nb = np.where((w > 0) & (h > 0), -(-h // rows), 0)
    tri = np.repeat(np.arange(len(w)), nb)
    band = np.arange(int(nb.sum())) - np.repeat(np.cumsum(nb) - nb, nb)
    by0 = y0[tri] + band * rows[tri]
    by1 = np.minimum(by0 + rows[tri] - 1, y1[tri])
    return tri, by0, by1


def raster_tri_ids(uvP, uvQ, uvR, P, Q, R, H, W, pos_map=None, valid=None, margin=0,
                   budget=1 << 22):
    """(H*W,) int32: the triangle each texel belongs to, or -1.

    Texel (x, y) of the flat map is UV ((x + 0.5)/W, 1 - (y + 0.5)/H), the same convention
    ops/bake.py builds uv_self with. A texel belongs to the triangle whose UV triangle
    contains its centre. Where UVs overlap, several triangles contain it; the one whose
    interpolated 3D position is closest to the baked position there wins, so the answer
    agrees with what Cycles baked. Cycles' bake margin also marks texels OUTSIDE every
    triangle as valid; those take their nearest drawn neighbour's triangle, grown the same
    4-neighbour way baker.dilate_fill grows the maps, far enough to reach `margin` pixels
    diagonally as well as straight out."""
    from .baker import dilate_fill
    uv = [np.asarray(x, np.float64) for x in (uvP, uvQ, uvR)]
    XY = [np.stack([u[:, 0] * W - 0.5, (1.0 - u[:, 1]) * H - 0.5], 1) for u in uv]
    lo = np.minimum(np.minimum(XY[0], XY[1]), XY[2])
    hi = np.maximum(np.maximum(XY[0], XY[1]), XY[2])
    x0 = np.clip(np.ceil(lo[:, 0] - 1e-9), 0, W).astype(np.int64)
    x1 = np.clip(np.floor(hi[:, 0] + 1e-9), -1, W - 1).astype(np.int64)
    y0 = np.clip(np.ceil(lo[:, 1] - 1e-9), 0, H).astype(np.int64)
    y1 = np.clip(np.floor(hi[:, 1] + 1e-9), -1, H - 1).astype(np.int64)

    best = np.full(H * W, -1, np.int64)
    best_d = np.full(H * W, np.inf)
    posf = None if pos_map is None else np.asarray(pos_map, np.float64).reshape(-1, 3)
    P3 = [np.asarray(x, np.float64) for x in (P, Q, R)]

    tri, by0, by1 = _pieces(x0, x1, y0, y1, budget)
    npx = (x1[tri] - x0[tri] + 1) * (by1 - by0 + 1)
    cum = np.cumsum(npx)
    a = 0
    while a < len(tri):
        b = max(int(np.searchsorted(cum, (cum[a - 1] if a else 0) + budget, "right")), a + 1)
        t, yb0 = tri[a:b], by0[a:b]
        w = x1[t] - x0[t] + 1
        n = npx[a:b]
        tt = np.repeat(t, n)
        off = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        px = x0[tt] + off % np.repeat(w, n)
        py = np.repeat(yb0, n) + off // np.repeat(w, n)
        # barycentrics of the texel centre in UV-pixel space
        A, B, C = XY[0][tt], XY[1][tt], XY[2][tt]
        v0, v1 = B - A, C - A
        v2 = np.stack([px - A[:, 0], py - A[:, 1]], 1)
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        eps = 1e-7
        inside = (l1 >= -eps) & (l2 >= -eps) & (l1 + l2 <= 1 + eps) & (den != 0)
        tt, px, py, l1, l2 = tt[inside], px[inside], py[inside], l1[inside], l2[inside]
        pix = py * W + px
        if posf is not None:
            q = (P3[0][tt] * (1 - l1 - l2)[:, None] + P3[1][tt] * l1[:, None]
                 + P3[2][tt] * l2[:, None])
            d = ((q - posf[pix]) ** 2).sum(1)
        else:
            d = np.zeros(len(pix))
        # the nearest candidate per texel in this chunk, then against the running best
        o = np.lexsort((tt, d, pix))
        pix, tt, d = pix[o], tt[o], d[o]
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, tt, d = pix[first], tt[first], d[first]
        better = d < best_d[pix]
        best[pix[better]] = tt[better]
        best_d[pix[better]] = d[better]
        a = b

    if margin:
        # dilate_fill grows one 4-neighbour step per pass, so n passes reach n pixels
        # straight out but only n/2 diagonally; Cycles' margin reaches `margin` pixels in
        # every direction, so it takes 2 * margin passes to cover its corners. Texels that
        # were not in the margin are masked back out by `valid` below.
        grid = best.reshape(H, W)
        grid = dilate_fill(grid, grid >= 0, 2 * int(margin))
        best = grid.ravel()
    if valid is not None:
        best = np.where(np.asarray(valid).ravel(), best, -1)
    return best.astype(np.int32)
