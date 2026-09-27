"""Reachable surface: a stroke paints only surface it can reach ACROSS the mesh.

The bug this guards: the stroke search measured distance in 3D and angle against the
stroke's normal, so a stroke painted any nearby surface facing the same way -- a second
layer 2 mm away, a floating panel, the far side of a fold. core/reach.py restricts each
stroke to triangles inside its sphere that connect back to its own triangle through
triangles also inside it. These checks prove, against brute force where one exists:

  - the geometry primitives (exact point-triangle distance, welding by position);
  - the reach rule on the cases that motivated it (stacked layers, a fold) and on the
    case it must NOT break (a surface split along a UV seam);
  - the vectorised flood fill equals a plain per-stroke BFS;
  - which texel belongs to which triangle, including overlapping UVs and bake margin;
  - end to end through resolve_uv: the bug reproduces without reach and is gone with it,
    and on a single connected surface reach changes nothing.

Run directly: python3 autostroke/tests/test_reach.py
"""

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import baker, reach                                  # noqa: E402
from core.config import Config                                 # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("   %-60s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def grid(n, x0=0.0, x1=1.0, y0=0.0, y1=1.0, z=lambda x, y: 0.0 * x):
    """An n x n vertex grid as triangles: (P, Q, R) corners and per-corner UVs = (x, y)
    normalised. Vertices are SHARED by index here; positions are what weld() sees."""
    xs, ys = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    V = np.stack([xs, ys, z(xs, ys)], -1).reshape(-1, 3)
    UV = np.stack([(xs - x0) / (x1 - x0), (ys - y0) / (y1 - y0)], -1).reshape(-1, 2)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    tri = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]], UV[tri[:, 0]], UV[tri[:, 1]], UV[tri[:, 2]]


def cat(*parts):
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(len(parts[0])))


def components(adj, T):
    start, nbr = adj
    lab = -np.ones(T, int)
    c = 0
    for s in range(T):
        if lab[s] >= 0:
            continue
        lab[s] = c
        q = deque([s])
        while q:
            t = q.popleft()
            for u in nbr[start[t]:start[t + 1]]:
                if lab[u] < 0:
                    lab[u] = c
                    q.append(u)
        c += 1
    return c, lab


def brute_reach(s_tri, s_pos, r, P, Q, R, adj):
    """Plain per-stroke BFS with the exact distance: the definition, written the slow way."""
    start, nbr = adj
    out = set()
    for i in range(len(s_tri)):
        seen = {int(s_tri[i])}
        q = deque([int(s_tri[i])])
        while q:
            t = q.popleft()
            for u in nbr[start[t]:start[t + 1]]:
                u = int(u)
                if u in seen:
                    continue
                d = reach.point_tri_dist2(s_pos[i:i + 1], P[u:u + 1], Q[u:u + 1], R[u:u + 1])[0]
                if d < r[i] ** 2:
                    seen.add(u)
                    q.append(u)
        out |= {i * len(P) + t for t in seen}
    return np.array(sorted(out), np.int64)


def centroid_seeds(P, Q, R, tris, r):
    tris = np.asarray(tris)
    return tris, (P[tris] + Q[tris] + R[tris]) / 3.0, np.full(len(tris), float(r))


# ---------------------------------------------------------------------------

def main():
    primitives()
    reach_rule()
    flood_fill_is_exact()
    views_agree()
    texel_triangles()
    end_to_end()
    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


def primitives():
    print("\nGEOMETRY PRIMITIVES")
    rng = np.random.default_rng(1)
    n = 300
    A, B, C = (rng.normal(size=(n, 3)) for _ in range(3))
    B[:10] = A[:10] + 1e-9 * rng.normal(size=(10, 3))          # near-degenerate slivers
    p = rng.normal(size=(n, 3)) * 1.5
    exact = reach.point_tri_dist2(p, A, B, C)
    k = 160
    u, v = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k))
    m = (u + v) <= 1
    u, v = u[m], v[m]
    brute = np.empty(n)
    for i in range(n):
        q = A[i] + np.outer(u, B[i] - A[i]) + np.outer(v, C[i] - A[i])
        brute[i] = ((q - p[i]) ** 2).sum(1).min()
    check("point_tri_dist2 is never above dense brute force",
          (exact <= brute + 1e-9).all(), "worst %.2e" % (exact - brute).max())
    gap = np.sqrt(brute) - np.sqrt(exact)
    check("...and within the sampling resolution of it", gap.max() < 0.05,
          "max gap %.4f" % gap.max())
    on = reach.point_tri_dist2((A + B + C) / 3, A, B, C)
    check("a point on a triangle is at distance 0", on[10:].max() < 1e-24,
          "worst %.1e" % on[10:].max())
    check("...near-degenerate slivers stay finite and tiny", np.isfinite(on).all()
          and on[:10].max() < 1e-12, "worst %.1e" % on[:10].max())

    # welding: a surface split along a UV seam, with the seam's vertices DUPLICATED
    Pa, Qa, Ra, *_ = grid(8, 0.0, 0.5, 0.0, 1.0)
    Pb, Qb, Rb, *_ = grid(8, 0.5, 1.0, 0.0, 1.0)
    P, Q, R = np.concatenate([Pa, Pb]), np.concatenate([Qa, Qb]), np.concatenate([Ra, Rb])
    T = len(P)
    welded = reach.tri_adjacency(reach.weld(P, Q, R))
    check("a seam with duplicated vertices welds into ONE surface",
          components(welded, T)[0] == 1)
    # what joining by vertex INDEX sees: each half's vertices numbered on their own, so
    # the seam's duplicates never match -- the failure welding exists to prevent
    va = reach.weld(Pa, Qa, Ra)
    vb = reach.weld(Pb, Qb, Rb) + va.max() + 1
    by_index = reach.tri_adjacency(np.concatenate([va, vb]))
    check("  (joined by index instead, it splits at the seam)",
          components(by_index, T)[0] == 2, "%d pieces" % components(by_index, T)[0])


def reach_rule():
    print("\nTHE RULE, ON THE CASES THAT MOTIVATED IT")
    # two stacked layers 2 mm apart, strokes on both with radius far bigger than the gap
    lo = grid(12, z=lambda x, y: 0.0 * x)
    hi = grid(12, z=lambda x, y: 0.0 * x + 0.002)
    P, Q, R = cat(lo, hi)[:3]
    T, Tlo = len(P), len(lo[0])
    rng = np.random.default_rng(3)
    s_tri, s_pos, r = centroid_seeds(P, Q, R, rng.integers(0, T, 60), 0.2)
    rs = reach.build(P, Q, R, s_tri, s_pos, r)
    keys = np.repeat(np.arange(len(s_tri)), np.diff(rs["stroke_start"])) * T + rs["stroke_tris"]
    s, t = keys // T, keys % T
    cross = int(((t < Tlo) != (s_tri[s] < Tlo)).sum())
    check("stacked layers 2 mm apart: no stroke reaches the other layer", cross == 0,
          "%d crossing pairs of %d" % (cross, len(keys)))
    # the old rule (sphere only) would have: count triangles of the other layer in range
    near_other = 0
    for i in range(len(s_tri)):
        other = np.arange(Tlo, T) if s_tri[i] < Tlo else np.arange(Tlo)
        d = reach.point_tri_dist2(np.repeat(s_pos[i:i + 1], len(other), 0),
                                  P[other], Q[other], R[other])
        near_other += int((d < r[i] ** 2).sum())
    check("  (distance alone puts the other layer in range)", near_other > 0,
          "%d other-layer triangles inside spheres" % near_other)
    same = 0
    for i in range(len(s_tri)):
        mine = np.arange(Tlo) if s_tri[i] < Tlo else np.arange(Tlo, T)
        d = reach.point_tri_dist2(np.repeat(s_pos[i:i + 1], len(mine), 0),
                                  P[mine], Q[mine], R[mine])
        want = set(mine[d < r[i] ** 2].tolist())
        got = set(rs["stroke_tris"][rs["stroke_start"][i]:rs["stroke_start"][i + 1]].tolist())
        same += want == got
    check("...while each stroke still reaches ALL of its own layer in range",
          same == len(s_tri), "%d of %d strokes" % (same, len(s_tri)))

    # one connected strip folded back on itself: bottom at z=0, a turn at x=1, top at z=g
    g, n = 0.01, 40
    xs = np.linspace(0, 1, n)
    turn = np.linspace(-np.pi / 2, np.pi / 2, 9)[1:-1]
    path = np.concatenate([np.stack([xs, np.zeros(n)], 1),
                           np.stack([1 + (g / 2) * np.cos(turn), g / 2 + (g / 2) * np.sin(turn)], 1),
                           np.stack([xs[::-1], np.full(n, g)], 1)])
    m = len(path)
    V = np.concatenate([np.c_[path[:, 0], np.zeros(m), path[:, 1]],
                        np.c_[path[:, 0], np.full(m, 0.2), path[:, 1]]])
    a = np.arange(m - 1)
    tri = np.concatenate([np.stack([a, a + 1, a + 1 + m], 1), np.stack([a, a + 1 + m, a + m], 1)])
    P, Q, R = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
    T = len(P)
    top = ((P[:, 2] + Q[:, 2] + R[:, 2]) / 3) > g * 0.99
    start_tri = int(np.argmin(np.abs((P[:, 0] + Q[:, 0] + R[:, 0]) / 3 - 0.5)
                              + 10 * ((P[:, 2] + Q[:, 2] + R[:, 2]) / 3)))
    for rad, expect_cross, label in ((0.2, False, "fold beyond the sphere"),
                                     (0.7, True, "fold inside the sphere")):
        s_tri, s_pos, r = centroid_seeds(P, Q, R, [start_tri], rad)
        rs = reach.build(P, Q, R, s_tri, s_pos, r)
        hit_top = bool(top[rs["stroke_tris"]].any())
        # directly above the stroke, 1 cm away, is in range either way
        above = top & (np.abs((P[:, 0] + Q[:, 0] + R[:, 0]) / 3 - 0.5) < 0.05)
        check("folded strip, %s: reaches the top layer = %s" % (label, expect_cross),
              hit_top == expect_cross and above.any(),
              "stroke r=%.1f, top triangles reached: %d" % (rad, int(top[rs["stroke_tris"]].sum())))


def flood_fill_is_exact():
    print("\nTHE VECTORISED FLOOD FILL EQUALS A PLAIN BFS")
    rng = np.random.default_rng(5)
    P, Q, R, *_ = grid(18, z=lambda x, y: 0.08 * np.sin(9 * x) * np.cos(7 * y))
    P2, Q2, R2, *_ = grid(10, 0.3, 0.7, 0.3, 0.7, z=lambda x, y: 0.0 * x + 0.02)
    P, Q, R = np.concatenate([P, P2]), np.concatenate([Q, Q2]), np.concatenate([R, R2])
    T = len(P)
    s_tri = rng.integers(0, T, 50)
    s_pos = (P[s_tri] + Q[s_tri] + R[s_tri]) / 3
    r = rng.uniform(0.03, 0.25, 50)
    adj = reach.tri_adjacency(reach.weld(P, Q, R))
    fast = reach.reachable(s_tri, s_pos, r, P, Q, R, adj)
    slow = brute_reach(s_tri, s_pos, r, P, Q, R, adj)
    check("identical (stroke, triangle) pairs", np.array_equal(fast, slow),
          "%d vs %d pairs" % (len(fast), len(slow)))
    check("  (the fixture is not trivial: strokes reach many triangles)", len(fast) > 20 * 50,
          "%.1f per stroke" % (len(fast) / 50.0))


def views_agree():
    print("\nTHE GPU VIEW AND THE CPU VIEW ARE THE SAME PAIRS")
    rng = np.random.default_rng(8)
    P, Q, R, *_ = grid(14, z=lambda x, y: 0.05 * np.sin(6 * x))
    T = len(P)
    s_tri, s_pos, r = centroid_seeds(P, Q, R, rng.integers(0, T, 40), 0.15)
    rs = reach.build(P, Q, R, s_tri, s_pos, r)
    by_tri = {(int(s), t) for t in range(T)
              for s in rs["tri_strokes"][rs["tri_start"][t]:rs["tri_start"][t] + rs["tri_count"][t]]}
    by_stroke = {(i, int(t)) for i in range(len(s_tri))
                 for t in rs["stroke_tris"][rs["stroke_start"][i]:rs["stroke_start"][i + 1]]}
    check("per-triangle lists == per-stroke lists", by_tri == by_stroke and len(by_tri) == rs["pairs"],
          "%d pairs" % len(by_tri))
    sorted_ok = all(np.all(np.diff(rs["stroke_tris"][rs["stroke_start"][i]:rs["stroke_start"][i + 1]]) > 0)
                    for i in range(len(s_tri)))
    check("per-stroke triangles are sorted (in_sorted relies on it)", sorted_ok)
    check("in_sorted matches np.isin",
          np.array_equal(reach.in_sorted(np.arange(-3, 40), np.array([0, 5, 9, 33])),
                         np.isin(np.arange(-3, 40), [0, 5, 9, 33])))


def texel_triangles():
    print("\nWHICH TRIANGLE EACH TEXEL BELONGS TO")
    P, Q, R, uP, uQ, uR = grid(9, z=lambda x, y: 0.1 * x)
    H = W = 48
    ids = reach.raster_tri_ids(uP, uQ, uR, P, Q, R, H, W)
    ys, xs = np.divmod(np.arange(H * W), W)
    u, v = (xs + 0.5) / W, 1.0 - (ys + 0.5) / H
    bad = 0
    for k in range(H * W):
        t = ids[k]
        a, b, c = uP[t], uQ[t], uR[t]
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        l1, l2 = np.linalg.solve(m, [u[k] - a[0], v[k] - a[1]])
        if t < 0 or min(l1, l2) < -1e-6 or l1 + l2 > 1 + 1e-6:
            bad += 1
    check("every texel lands in a triangle that contains its centre", bad == 0,
          "%d of %d wrong" % (bad, H * W))

    # two triangles sharing the SAME UVs (mirrored halves) at different 3D places
    P2, Q2, R2 = P + [0, 0, 5.0], Q + [0, 0, 5.0], R + [0, 0, 5.0]
    PP, QQ, RR = np.concatenate([P, P2]), np.concatenate([Q, Q2]), np.concatenate([R, R2])
    uu = [np.concatenate([x, x]) for x in (uP, uQ, uR)]
    base = reach.raster_tri_ids(*uu, PP, QQ, RR, H, W)
    # a baked position map that says the upper copy is what Cycles wrote
    pos = np.zeros((H * W, 3))
    ok_mask = base >= 0
    t0 = base % len(P)
    l = np.stack([u, v], 1)
    pos[:, :2] = l
    pos[:, 2] = 0.1 * l[:, 0] + 5.0
    pick = reach.raster_tri_ids(*uu, PP, QQ, RR, H, W, pos_map=pos)
    check("overlapping UVs: the triangle matching the baked position wins",
          (pick[ok_mask] >= len(P)).all() and np.array_equal(pick[ok_mask] % len(P), t0[ok_mask]))

    # bake margin: texels outside every triangle but marked valid take a neighbour's id
    half = grid(9, 0.0, 0.5, 0.0, 1.0)
    hu = [np.c_[x[:, 0] * 0.5, x[:, 1]] for x in half[3:]]   # UVs cover the left half only
    plain = reach.raster_tri_ids(*hu, *half[:3], H, W)
    grown = reach.raster_tri_ids(*hu, *half[:3], H, W, margin=4,
                                 valid=np.ones(H * W, bool))
    col = np.arange(H * W) % W
    edge = (col >= W // 2) & (col < W // 2 + 4)
    check("margin texels take their nearest drawn neighbour's triangle",
          (plain[edge] < 0).all() and (grown[edge] >= 0).all()
          and np.array_equal(grown[plain >= 0], plain[plain >= 0]))
    # a diagonal corner of the margin: `margin` pixels right AND `margin` pixels past the
    # island's end -- Cycles fills it; 4-neighbour growth needs 2 * margin passes to get there
    tri_half = grid(9, 0.0, 0.5, 0.0, 0.5)
    qu = [np.c_[x[:, 0] * 0.5, x[:, 1] * 0.5] for x in tri_half[3:]]   # bottom-left quarter
    got = reach.raster_tri_ids(*qu, *tri_half[:3], H, W, margin=4, valid=np.ones(H * W, bool))
    corner = (H // 2 - 4) * W + (W // 2 + 3)                          # 4 up, 4 right, diagonal
    check("margin reaches its diagonal corners too", got[corner] >= 0, "id %d" % got[corner])
    check("...and invalid texels stay -1",
          (reach.raster_tri_ids(*hu, *half[:3], H, W, valid=np.zeros(H * W, bool)) == -1).all())
    big = reach.raster_tri_ids(uP, uQ, uR, P, Q, R, H, W, budget=64)
    check("a tiny memory budget (row bands) gives the same answer", np.array_equal(big, ids))


def end_to_end():
    print("\nEND TO END THROUGH resolve_uv")
    cfg = Config(mask_scale=1.0, crease_angle_deg=60.0)
    mask = np.ones((32, 32), np.float32)
    rng = np.random.default_rng(11)

    def scene(layers):
        parts = [grid(16, z=lambda x, y, zz=zz: 0.0 * x + zz) for zz in layers]
        P, Q, R, uP, uQ, uR = cat(*parts)
        T = len(P)
        # texels: random points on random triangles, tagged with their triangle
        tt = rng.integers(0, T, 6000)
        a, b = rng.random(6000), rng.random(6000)
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        pts = P[tt] + (Q[tt] - P[tt]) * a[:, None] + (R[tt] - P[tt]) * b[:, None]
        s_tri = rng.integers(0, T, 150)
        s_pos = (P[s_tri] + Q[s_tri] + R[s_tri]) / 3
        return P, Q, R, tt, pts.astype(np.float32), s_tri, s_pos

    def run(P, Q, R, tt, pts, s_tri, s_pos, with_reach):
        n = len(s_tri)
        seed_r = np.full(n, 0.12, np.float32)
        nrm = np.tile([[0, 0, 1.0]], (n, 1)).astype(np.float32)
        uv = np.stack([np.arange(n) / n + 1e-4, np.zeros(n)], 1).astype(np.float32)
        kw = {}
        if with_reach:
            rs = reach.build(P, Q, R, s_tri, s_pos,
                             baker.effective_radius(seed_r, np.arange(n), cfg))
            kw = dict(pt_tri=tt, stroke_reach=(rs["stroke_start"], rs["stroke_tris"]))
        out = baker.resolve_uv_blocking(
            pts, np.zeros((len(pts), 2), np.float32), s_pos.astype(np.float32),
            np.arange(n, dtype=np.int64), uv, seed_r, nrm, None, mask, cfg,
            np.tile([[1.0, 0, 0]], (n, 1)), 0.0,
            pt_nrm=np.tile([[0, 0, 1.0]], (len(pts), 1)).astype(np.float32), **kw)
        win = np.where(out[2], np.round((out[0][:, 0] - 1e-4) * n).astype(int), -1)
        return out, win

    P, Q, R, tt, pts, s_tri, s_pos = scene([0.0, 0.002])
    Tlo = len(P) // 2
    for with_reach, want_bug in ((False, True), (True, False)):
        out, win = run(P, Q, R, tt, pts, s_tri, s_pos, with_reach)
        wrong = int(((win >= 0) & ((s_tri[np.maximum(win, 0)] < Tlo) != (tt < Tlo))).sum())
        check("stacked layers, reach %s: texels won by the OTHER layer's stroke"
              % ("ON " if with_reach else "OFF"),
              (wrong > 0) if want_bug else (wrong == 0),
              "%d of %d covered texels" % (wrong, int(out[2].sum())))

    P, Q, R, tt, pts, s_tri, s_pos = scene([0.0])
    a, _ = run(P, Q, R, tt, pts, s_tri, s_pos, False)
    b, _ = run(P, Q, R, tt, pts, s_tri, s_pos, True)
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    check("one connected surface: reach changes nothing, byte for byte", same,
          "%d texels covered" % int(a[2].sum()))


if __name__ == "__main__":
    sys.exit(main())
