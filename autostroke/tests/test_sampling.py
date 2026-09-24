"""Every claim the sampler rests on, with the measured number attached.

Run:  python3 autostroke/tests/test_sampling.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import sampling as S   # noqa: E402

fails = []


def check(name, ok, detail):
    print("   %-52s %-5s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        fails.append(name)


def strip_points(sampler, n_total, L=1.0, w=0.01):
    """1 m x 1 cm bevel strip as two triangles; returns points + per-stroke radius."""
    P = np.array([[0, 0], [0, 0]], float)
    Q = np.array([[L, 0], [L, w]], float)
    R = np.array([[L, w], [0, w]], float)
    counts = np.array([n_total // 2, n_total - n_total // 2])
    return sampler(P, Q, R, np.array([0, 1]), counts)


def covered_length(x, r, L=1.0):
    g = np.diff(np.concatenate([[0.0], np.sort(x), [L]]))
    rr = np.atleast_1d(r)
    reach = 2 * float(np.median(rr))
    return min(1.0, sum(min(gi, reach) for gi in g) / L)


def main():
    print("SAMPLER")

    # --- 1. bevel strip: the four-way split's failure case ------------------
    pos, la, lasp, fid, _, _ = strip_points(S.sample_faces, 128)
    r = S.stroke_radius(la, lasp)
    cov = covered_length(pos[:, 0], r)
    check("bevel strip 1m x 1cm, 128 strokes: length covered", cov >= 0.95,
          "%.0f%%  (four-way split reference: 30%%)" % (100 * cov))

    # The correction earns its keep at LOW counts -- a bevel with a couple of strokes.
    # By 128 strokes the placement alone nearly covers the strip (97%), so asserting
    # the difference there would prove nothing.
    pos8, la8, lasp8, _, _, _ = strip_points(S.sample_faces, 8)
    cov8 = covered_length(pos8[:, 0], S.stroke_radius(la8, lasp8))
    cov8_flat = covered_length(pos8[:, 0], np.sqrt(la8))
    check("bevel strip, only 8 strokes: covered", cov8 >= 0.95, "%.0f%%" % (100 * cov8))
    check("  without the aspect correction it is not", cov8_flat < 0.75,
          "%.0f%%  (correction is worth +%.0f points here)"
          % (100 * cov8_flat, 100 * (cov8 - cov8_flat)))

    # --- 2. aspect measure must see an isosceles sliver ---------------------
    iso = (np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0.5, 0.01]]))
    a_iso = float(S.tri_aspect(*iso))
    check("isosceles sliver reports its true aspect", a_iso > 90,
          "%.0f  (longest/shortest edge would say 2.0)" % a_iso)

    # --- 3. compact faces must not be inflated ------------------------------
    eq = (np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0.5, np.sqrt(3) / 2]]))
    lp, lq, lr = S.leb_leaves(np.repeat(eq[0], 8, 0), np.repeat(eq[1], 8, 0),
                              np.repeat(eq[2], 8, 0), np.arange(8), depth=3)
    ratio = float(np.mean(S.stroke_radius(S.tri_area(lp, lq, lr),
                                          S.tri_aspect(lp, lq, lr)) / np.sqrt(S.tri_area(lp, lq, lr))))
    check("equilateral leaves keep ~their uncorrected size", 0.9 <= ratio <= 1.15,
          "radius x%.2f" % ratio)
    for n_eq in (1, 16, 64):
        pe, lae, laspe, _, _, _ = S.sample_faces(eq[0], eq[1], eq[2], np.array([0]), np.array([n_eq]))
        rr = float(np.median(S.stroke_radius(lae, laspe) / np.sqrt(lae)))
        check("  compact face unchanged at N=%d" % n_eq, 0.95 <= rr <= 1.15, "radius x%.2f" % rr)

    # --- 4. nesting: raising density must not move existing strokes ---------
    tri = (np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0.4, 0.3]]))
    a = S.leb_leaves(np.repeat(tri[0], 16, 0), np.repeat(tri[1], 16, 0),
                     np.repeat(tri[2], 16, 0), np.arange(16))
    b = S.leb_leaves(np.repeat(tri[0], 17, 0), np.repeat(tri[1], 17, 0),
                     np.repeat(tri[2], 17, 0), np.arange(17))
    ca = (a[0] + a[1] + a[2]) / 3
    cb = ((b[0] + b[1] + b[2]) / 3)[:16]
    moved = int((np.linalg.norm(ca - cb, axis=1) > 1e-12).sum())
    check("raising density moves no existing stroke", moved == 0, "%d of 16 moved" % moved)

    # --- 4b. nesting across a DENSITY change, on a quad ---------------------
    # The stricter version of 4: a quad's strokes are shared between its two triangles,
    # and that choice must not depend on the count either. Using (k+0.5)/n for it left
    # only 70% of strokes in place on a real mesh.
    qP = np.array([[0, 0], [0, 0]], float)
    qQ = np.array([[1, 0], [1, 1]], float)
    qR = np.array([[1, 1], [0, 1]], float)
    keep = []
    for n in (9, 13):
        pos_n, _, _, fid_n, wi_n, _ = S.sample_faces(qP, qQ, qR, np.array([0, 0]), np.array([n]))
        keep.append((fid_n.astype(np.int64) * 1000 + wi_n, pos_n))
    common, ia, ib = np.intersect1d(keep[0][0], keep[1][0], return_indices=True)
    drift = np.linalg.norm(keep[0][1][ia] - keep[1][1][ib], axis=1)
    check("quad: raising the count moves no surviving stroke",
          len(common) == 9 and drift.max() == 0.0,
          "%d shared, max drift %.1e" % (len(common), drift.max()))

    # --- 5. dithering: density must act smoothly on a uniform mesh ----------
    areas = 0.01 * (1 + 0.02 * np.random.default_rng(0).standard_normal(2000))
    tot = [S.stroke_counts(areas, p).sum() for p in (60, 90, 120, 150, 180)]
    steps = np.diff(tot)
    biggest = int(steps.max())
    check("uniform mesh: stroke count rises smoothly with density",
          biggest < 0.5 * len(areas), "biggest single step %d strokes over %d faces"
          % (biggest, len(areas)))
    plain = [np.clip(np.round(areas * p), 1, 16).sum() for p in (60, 90, 120, 150, 180)]
    check("  (plain rounding would jump the whole mesh at once)",
          int(np.diff(plain).max()) >= 0.5 * len(areas),
          "plain rounding step: %d" % int(np.diff(plain).max()))

    # --- 7. area-uniformity: bisection must not bias ------------------------
    pos2, la2, _, _, _, _ = S.sample_faces(
        np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]),
        np.array([0]), np.array([4096]))
    err = abs(float(pos2.mean(0)[0]) - 1 / 3) + abs(float(pos2.mean(0)[1]) - 1 / 3)
    spread = float(la2.max() / la2.min())
    check("sampling stays area-uniform", err < 0.01, "centroid error %.4f" % err)
    check("  every leaf has equal area", spread < 1.001, "max/min leaf area %.6f" % spread)

    # --- 8. a quad is one unit, not two triangles ---------------------------
    _, _, _, fid2, _, _ = S.sample_faces(
        np.array([[0., 0.], [0., 0.]]), np.array([[1., 0.], [1., 1.]]),
        np.array([[1., 1.], [0., 1.]]), np.array([0, 0]), np.array([10]))
    check("a quad's strokes are shared across its triangles", (fid2 == 0).all() and len(fid2) == 10,
          "%d strokes, all on face 0" % len(fid2))

    # --- 9. fan cap: the topology that used to leave a ring-shaped hole -----
    rng = np.random.default_rng(0)
    k = 48
    ang = np.linspace(0, 2 * np.pi, k + 1)
    rim = np.stack([np.cos(ang), np.sin(ang)], 1)
    fP, fQ, fR = np.zeros((k, 2)), rim[:-1], rim[1:]
    counts = S.stroke_counts(S.tri_area(fP, fQ, fR), 15)
    pos3, la3, lasp3, _, _, _ = S.sample_faces(fP, fQ, fR, np.arange(k), counts)
    r3 = S.stroke_radius(la3, lasp3)
    g = rng.random((20000, 2)) * 2 - 1
    probe = g[np.linalg.norm(g, axis=1) < 1.0]
    hub = probe[np.linalg.norm(probe, axis=1) < 0.3]
    cov_all = float((np.sqrt(((probe[:, None] - pos3[None]) ** 2).sum(-1)) <= r3[None]).any(1).mean())
    cov_hub = float((np.sqrt(((hub[:, None] - pos3[None]) ** 2).sum(-1)) <= r3[None]).any(1).mean())
    check("fan cap, 48 wedges: whole disc covered", cov_all >= 0.99, "%.0f%%" % (100 * cov_all))
    check("  and the hub, which used to be a ring-shaped hole", cov_hub >= 0.95,
          "%.0f%%" % (100 * cov_hub))

    # --- 10. no regression on ordinary faces --------------------------------
    sq_probe = rng.random((4000, 2)) * 0.96 + 0.02
    qP = np.array([[0, 0], [0, 0]], float)
    qQ = np.array([[1, 0], [1, 1]], float)
    qR = np.array([[1, 1], [0, 1]], float)
    pos4, _, _, _, _, _ = S.sample_faces(qP, qQ, qR, np.array([0, 1]), np.array([16, 16]))
    D = np.sqrt(((pos4[:, None] - pos4[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    even = np.sqrt(1.0 / len(pos4))
    closest = float(D.min() / even)
    holeq = float(np.sqrt(((sq_probe[:, None] - pos4[None]) ** 2).sum(-1)).min(1).max() / even)
    check("square panel: spacing no worse than the four-way split",
          closest >= 0.6 and holeq <= 1.0,
          "closest %.2fx  hole %.2fx  (four-way: 0.67x / 0.87x)" % (closest, holeq))

    # --- 6. the size cap: a count-capped face must not keep growing ---------
    # This is the actual reported bug: Max Strokes per Face bounds COUNT, not size, so
    # leaf_area = face_area / max_strokes grows exactly as fast as the face once capped,
    # and radius (uncorrected) grows without bound right along with it.
    print("\nSTROKE SIZE STOPS GROWING ONCE A FACE IS COUNT-CAPPED")
    areas = np.concatenate([np.full(50, 1.0), [16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0]])
    big = np.array([16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0])

    density, _ = S.solve_density(areas, target=2000, min_strokes=1, max_strokes=128)
    counts = S.stroke_counts(areas, density, min_strokes=1, max_strokes=128)
    leaf_at = {a: areas[np.where(areas == a)[0][-1]] / max(counts[np.where(areas == a)[0][-1]], 1)
              for a in big}

    uncapped = np.array([S.stroke_radius(np.array([leaf_at[a]]), np.array([2.0]))[0]
                         for a in big])
    check("without density: confirms the bug -- radius keeps growing, ~32x end to end",
          uncapped[-1] / uncapped[0] > 20,
          "%.2f -> %.2f (x%.1f)" % (uncapped[0], uncapped[-1], uncapped[-1] / uncapped[0]))

    capped = np.array([S.stroke_radius(np.array([leaf_at[a]]), np.array([2.0]),
                                       density=density)[0] for a in big])
    R = 1.0 / np.sqrt(density)
    check("with density: every capped face stays under the natural ceiling",
          bool((capped <= R * 1.001).all()),
          "R=%.3f, max seen %.3f" % (R, capped.max()))
    check("...and the growth from smallest to largest capped face is dramatically tamed",
          capped[-1] / capped[0] < 3.0,
          "%.2f -> %.2f (x%.1f, vs x%.1f uncapped)"
          % (capped[0], capped[-1], capped[-1] / capped[0], uncapped[-1] / uncapped[0]))

    # small, NOT count-capped strokes must be practically unaffected -- this is the
    # backward-compatibility half of the same claim, not just "does it cap". density=100
    # gives R=0.1; leaf_area=0.0001 gives a raw radius of 0.01, a tenth of R -- well
    # inside the near-identity region, not sitting AT the ceiling itself.
    small = S.stroke_radius(np.array([0.0001]), np.array([2.0]), density=100.0)[0]
    small_raw = S.stroke_radius(np.array([0.0001]), np.array([2.0]))[0]
    check("a normal, well-under-the-ceiling stroke is barely touched",
          abs(small - small_raw) / small_raw < 0.02,
          "%.6f vs %.6f (%.2f%% difference)"
          % (small, small_raw, 100 * abs(small - small_raw) / small_raw))

    # density=None (the default, every OTHER call in this file) must reproduce the
    # exact uncapped formula -- no caller anywhere else in the codebase opts in by
    # accident.
    exact = float(S.stroke_radius(np.array([5.0]), np.array([3.0]))[0])
    expect = float(np.sqrt(5.0) * (3.0 / 2.0) ** 0.5)
    check("omitting density changes nothing (exact match, not just 'close')",
          exact == expect, "%.10f vs %.10f" % (exact, expect))

    print("\n%s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
