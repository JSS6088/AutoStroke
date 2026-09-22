"""Where strokes go on a mesh, and how big each one is.

Pure numpy, no bpy, so the whole scheme stays testable outside Blender.

Three decisions, each settled by measurement rather than taste (see tests/test_sampling.py):

  counts   N = clamp(floor(A*p + u_face), min, max), dithered per face
  place    longest-edge bisection, bit l of the point index picks the half at level l
  size     r = sqrt(leaf area) * sqrt(leaf aspect / 2)
"""

import numpy as np

def hash01_u64(keys):
    """Deterministic [0,1) per integer key (splitmix64), for dithering and brush choice."""
    x = np.asarray(keys, np.int64).astype(np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x = x ^ (x >> np.uint64(31))
    return (x.astype(np.float64) / 2.0 ** 64)


def stroke_counts(areas, density, min_strokes=1, max_strokes=16, keys=None):
    """Strokes per face: clamp(floor(A*density + u_face), min, max).

    DITHERED, not rounded. With plain rounding every face of a near-uniform mesh crosses
    A*p = 0.5 at the same density, so the whole mesh jumps from 1 stroke to 2 at once and
    every stroke shrinks by sqrt(2) together. The per-face offset staggers those crossings,
    so the average count follows density smoothly while each face stays deterministic.
    """
    areas = np.asarray(areas, np.float64)
    if keys is None:
        keys = np.arange(len(areas))
    u = hash01_u64(keys)
    n = np.floor(areas * float(density) + u)
    return np.clip(n, min_strokes, max_strokes).astype(np.int64)


def solve_density(areas, target, min_strokes=1, max_strokes=128, iters=40):
    """Density that yields ~`target` strokes in total, respecting the per-face clamps.

    Total strokes rise monotonically with density, so a bisection lands on it in a few
    dozen cheap steps. Solving rather than dividing target by area matters because the
    clamps bend the relationship: on a low-poly mesh a few large faces hit the cap and
    stop contributing, so the naive target/area under-delivers.

    Returns (density, reachable_total). If the clamps make the target unreachable,
    reachable_total is the most the mesh can produce and the caller should say so.
    """
    areas = np.asarray(areas, np.float64)
    if len(areas) == 0 or areas.sum() <= 0:
        return 1.0, 0
    ceiling = int(len(areas) * max_strokes)
    floor_ = int(len(areas) * min_strokes)
    target = int(np.clip(target, floor_, ceiling))

    def total_at(p):
        return int(stroke_counts(areas, p, min_strokes, max_strokes).sum())

    lo, hi = 1e-6, max(1.0, target / areas.sum())
    while total_at(hi) < target and hi < 1e12:
        hi *= 4.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if total_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi, total_at(hi)


def _longest_edge_first(P, Q, R):
    """Rotate each triangle (vectorised) so its longest edge is P->Q. Winding is kept."""
    e0 = ((Q - P) ** 2).sum(-1)
    e1 = ((R - Q) ** 2).sum(-1)
    e2 = ((P - R) ** 2).sum(-1)
    k = np.argmax(np.stack([e0, e1, e2], -1), axis=-1)
    P2 = np.where(k[:, None] == 1, Q, np.where(k[:, None] == 2, R, P))
    Q2 = np.where(k[:, None] == 1, R, np.where(k[:, None] == 2, P, Q))
    R2 = np.where(k[:, None] == 1, P, np.where(k[:, None] == 2, Q, R))
    return P2, Q2, R2


def radical_inverse2(i):
    """Van der Corput base 2: bit-reversed index, in [0,1).

    Value depends only on i, so any prefix of the sequence is both evenly spread and
    unchanged when the sequence grows -- which is what keeps the triangle each stroke
    belongs to independent of how many strokes the face was given.
    """
    i = np.asarray(i, np.uint64)
    r = np.zeros_like(i)
    for _ in range(32):
        r = (r << np.uint64(1)) | (i & np.uint64(1))
        i >>= np.uint64(1)
    return r.astype(np.float64) / 2.0 ** 32


def _path_bits(idx):
    """Index bits, with the unused high bits filled from a hash of the index.

    For i < N only the low bits of i carry information; left as zeros they make every
    deep level take the same side, so the point drifts into one corner of its own
    responsibility region instead of sitting inside it. Filling them from hash(i) keeps
    the path deterministic and dependent on i alone -- so nesting still holds -- while
    placing the point somewhere sensible within its region.
    """
    idx = np.asarray(idx, np.int64)
    bitlen = np.zeros(len(idx), np.int64)
    v = idx.copy()
    while np.any(v > 0):
        bitlen += (v > 0)
        v >>= np.int64(1)
    h = (hash01_u64(idx ^ np.int64(0x5bf03635)) * 2.0 ** 62).astype(np.int64)
    return (idx | (h << bitlen)).astype(np.int64)


def leb_leaves(P, Q, R, idx, depth=32):
    """Longest-edge bisection: the leaf triangle each point index lands in.

    At every level the current longest edge is cut at its midpoint. Both halves have
    exactly equal area (same apex, equal bases), so sampling stays area-uniform, and bit
    `l` of the index picks the half at level `l`.

    Two properties this buys:
      * NESTED -- a point's leaf depends only on its index, never on how many points the
        face was asked for, so raising density never moves an existing stroke.
      * SHAPE-AWARE -- the cut always crosses the longest side, so a sliver is divided
        along its length until the pieces are compact. The four-way midpoint split divides
        both directions every level instead, wasting half its subdivisions on a width with
        room for one row of points (measured: 16 positions along a strip vs 90).

    All arrays are (n,2) or (n,3); `idx` is (n,) integer.
    """
    P = np.array(P, np.float64, copy=True)
    Q = np.array(Q, np.float64, copy=True)
    R = np.array(R, np.float64, copy=True)
    idx = np.asarray(idx, np.int64)
    for level in range(depth):
        P, Q, R = _longest_edge_first(P, Q, R)
        M = 0.5 * (P + Q)
        take_q = ((idx >> np.int64(level)) & 1).astype(bool)[:, None]
        P, Q = np.where(take_q, M, P), np.where(take_q, Q, M)
    return P, Q, R


def tri_area(P, Q, R):
    """Triangle area in 2D or 3D."""
    u, v = Q - P, R - P
    if P.shape[-1] == 2:
        return 0.5 * np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    return 0.5 * np.linalg.norm(np.cross(u, v), axis=-1)


def tri_aspect(P, Q, R):
    """Longest edge over the height above it, i.e. L^2 / 2A.

    NOT longest-over-shortest: an isosceles sliver has two long edges and a long base, so
    that measure reports 2.0 on a shape that is really 100:1 and would skip the correction
    on exactly the faces that need it.
    """
    L = np.sqrt(np.maximum.reduce([((Q - P) ** 2).sum(-1),
                                   ((R - Q) ** 2).sum(-1),
                                   ((P - R) ** 2).sum(-1)]))
    A = tri_area(P, Q, R)
    return np.where(A > 1e-30, L * L / (2.0 * np.maximum(A, 1e-30)), 1.0)


def stroke_radius(leaf_area, leaf_aspect, alpha=0.5, ideal_aspect=2.0):
    """r = sqrt(a) * (aspect / ideal)^alpha.

    sqrt(a) is the side of a SQUARE of that area, so it matches a face's real extent only
    when the face is compact. A sliver of length L and width w needs r ~ L/2 while
    sqrt(a) = sqrt(Lw/2); their ratio is sqrt(aspect/2), which is where alpha = 1/2 and
    the divide-by-2 come from -- both derived, not tuned. alpha = 1 covers no better and
    triples the spill onto neighbouring faces.

    Dividing by the ideal keeps compact faces near their old size: an equilateral's leaves
    sit at aspect 2.31, so an uncorrected sqrt(aspect) would enlarge every stroke by 1.5x.
    """
    return np.sqrt(leaf_area) * np.power(np.maximum(leaf_aspect, 1e-12) / ideal_aspect, alpha)


def sample_faces(tri_P, tri_Q, tri_R, face_of_tri, counts, depth=32,
                 alpha=0.5, ideal_aspect=2.0):
    """Place every face's strokes and size them.

    A face's N points are spread over ITS triangles by area, so a quad behaves like one
    quad rather than two independent triangles. Returns positions plus the per-stroke leaf
    area/aspect the radius is built from, and the index of each stroke within its face
    (which is what makes the sequence nested).

    tri_* are (T,3) triangle corners, face_of_tri is (T,) face index per triangle,
    counts is (F,) strokes per face.

    Returns (positions, leaf_area, leaf_aspect, face_id, index_within_face, triangle_id).
    """
    tri_P = np.asarray(tri_P, np.float64)
    tri_Q = np.asarray(tri_Q, np.float64)
    tri_R = np.asarray(tri_R, np.float64)
    face_of_tri = np.asarray(face_of_tri, np.int64)
    counts = np.asarray(counts, np.int64)
    n_faces = len(counts)

    areas = tri_area(tri_P, tri_Q, tri_R)
    order = np.argsort(face_of_tri, kind="stable")
    total = int(counts.sum())
    if total == 0:
        e = np.zeros((0, tri_P.shape[1]))
        z = np.zeros(0, np.int64)
        return e, np.zeros(0), np.zeros(0), z, z, z

    face_id = np.repeat(np.arange(n_faces), counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)

    # pick each stroke's triangle by cumulative area within its own face
    tri_pick = np.empty(total, np.int64)
    sub_idx = np.empty(total, np.int64)
    start = 0
    for f in range(n_faces):
        n = int(counts[f])
        if n == 0:
            continue
        tris = order[face_of_tri[order] == f]
        a = areas[tris]
        cdf = np.cumsum(a) / max(a.sum(), 1e-30)
        # Which triangle each stroke belongs to, by area. The fraction comes from the
        # radical inverse of the stroke's own index, NOT from (k+0.5)/n: the latter
        # depends on n, so raising density would reshuffle a quad's strokes between its
        # two triangles and move strokes that should have stayed put.
        frac = radical_inverse2(np.arange(n))
        which = np.searchsorted(cdf, frac, side="left").clip(0, len(tris) - 1)
        tri_pick[start:start + n] = tris[which]
        # each triangle gets its own 0..k-1 run, so bisection stays nested per triangle
        sub = np.zeros(n, np.int64)
        for j in range(len(tris)):
            m = which == j
            sub[m] = np.arange(int(m.sum()))
        sub_idx[start:start + n] = sub
        start += n

    # Two depths, on purpose:
    #   shallow = enough levels to separate this triangle's points. That leaf IS the
    #             point's responsibility region, so its shape sets the stroke size.
    #   deep    = the position, which must not depend on how many points were asked for.
    n_in_tri = np.bincount(tri_pick, minlength=len(tri_P))[tri_pick]
    shallow = np.maximum(np.ceil(np.log2(np.maximum(n_in_tri, 1))).astype(np.int64), 1)

    P0, Q0, R0 = tri_P[tri_pick], tri_Q[tri_pick], tri_R[tri_pick]
    lasp = np.empty(total)
    for k in np.unique(shallow):                       # a handful of distinct depths
        m = shallow == k
        sp, sq, sr = leb_leaves(P0[m], Q0[m], R0[m], sub_idx[m], int(k))
        lasp[m] = tri_aspect(sp, sq, sr)
    la = tri_area(P0, Q0, R0) / np.maximum(n_in_tri, 1)   # exact share of the triangle

    lp, lq, lr = leb_leaves(P0, Q0, R0, _path_bits(sub_idx), depth)
    pos = (lp + lq + lr) / 3.0
    # tri_pick is returned rather than re-derived by the caller: guessing a stroke's
    # triangle from the nearest centroid picks the wrong one near a quad's diagonal,
    # which would give that stroke the wrong UV and normal.
    return pos, la, lasp, face_id, within, tri_pick
