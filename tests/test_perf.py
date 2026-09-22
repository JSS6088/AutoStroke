"""Spatial acceleration: the rejection must be an exclusion, never an approximation.

resolve_uv sorts texels into Morton order, cuts them into tiles, and skips whole strokes
per tile with a sphere/box test. The danger is a test that is too TIGHT: it would drop
strokes at tile boundaries, which shows up as faint seams on a grid -- easy to miss by eye
and invisible to a coverage number. The defence is that output must not depend on the tile
size at all, which is what most of this file checks.

Pure numpy. Run directly: python3 autostroke/tests/test_perf.py
"""

import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import baker                                       # noqa: E402
from core.config import Config                               # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("   %-52s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# A corrugated surface: the smooth part gives curvature, the triangle ridges give creases
# where the normal flips ~60 deg. On a gentle surface the crease guard never fires and half
# of what follows would silently test nothing.
def _surface(u, v, period=1.5, ridge=0.45):
    t = u / period
    tri = 2.0 * np.abs(t - np.floor(t) - 0.5)
    dzdu = 0.6 * np.cos(u) * np.cos(v * 0.7) + ridge * 2.0 / period * np.sign(t - np.floor(t) - 0.5)
    dzdv = -0.42 * np.sin(u) * np.sin(v * 0.7)
    n = np.cross(np.stack([np.ones_like(u), np.zeros_like(u), dzdu], 1),
                 np.stack([np.zeros_like(u), np.ones_like(u), dzdv], 1))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    z = 0.6 * np.sin(u) * np.cos(v * 0.7) + ridge * tri
    return np.stack([u, v, z], 1).astype(np.float32), n.astype(np.float32)


def scene(n_pts=60_000, n_seeds=900, seed=7):
    rng = np.random.default_rng(seed)
    pts, pn = _surface(rng.random(n_pts) * 6.0, rng.random(n_pts) * 6.0)
    pos, nrm = _surface(rng.random(n_seeds) * 6.0, rng.random(n_seeds) * 6.0)
    return dict(pts=pts, pn=pn, pos=pos, nrm=nrm,
                uv=np.stack([rng.random(n_seeds), rng.random(n_seeds)], 1).astype(np.float32),
                rad=(0.05 + 0.25 * rng.random(n_seeds)).astype(np.float32),
                ids=np.arange(n_seeds, dtype=np.int64))


def bars(k):
    out = []
    for i in range(k):
        y, x = (np.mgrid[0:96, 0:96] / 95.0) - 0.5
        t = np.radians(15.0 * i)
        xr, yr = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
        out.append(((np.abs(xr) < 0.30 - 0.04 * i)
                    & (np.abs(yr) < 0.10 + 0.02 * i)).astype(np.float32))
    return out


def run(s, mask, cfg, k=slice(None)):
    return baker.resolve_uv_blocking(
        s["pts"], np.zeros((len(s["pts"]), 2), np.float32), s["pos"][k], s["ids"][k],
        s["uv"][k], s["rad"][k], s["nrm"][k], None, mask, cfg,
        s["nrm"][k].astype(np.float64), 0.0, pt_nrm=s["pn"])


# every branch of resolve_uv, so tile-independence is checked on all of them
CASES = [
    ("baseline",            dict(),                                       1, False),
    ("guard off",           dict(crease_guard=False),                     1, False),
    ("crease 10 deg",       dict(crease_angle_deg=10.0),                  1, False),
    ("four brushes",        dict(),                                       4, False),
    ("size variation",      dict(size_random=1.0),                        1, False),
    ("jitter 45",           dict(rot_jitter_deg=45.0),                    1, False),
    ("auto-align off",      dict(mask_auto_align=False),                  4, False),
    ("everything at once",  dict(size_random=1.0, rot_jitter_deg=45.0,
                                 crease_angle_deg=30.0),                  4, False),
    ("single stroke",       dict(),                                       1, True),
]


def main():
    print("\nSPATIAL ACCELERATION\n")
    s = scene()
    masks = {1: bars(1)[0], 4: bars(4)}

    # ---- the load-bearing test -------------------------------------------
    # A too-tight box test drops strokes only near tile boundaries, so it shows up as a
    # DIFFERENCE BETWEEN TILE SIZES long before anyone notices seams in a render.
    worst = None
    for name, kw, nb, solo in CASES:
        k = slice(0, 1) if solo else slice(None)
        ref = None
        for chunk in (0, 1024, 4096, 16384, 200_000):
            r = run(s, masks[nb], Config(chunk=chunk, **kw), k)
            sig = [digest(r[i]) for i in range(4)]
            if ref is None:
                ref, cov = sig, r[2].mean()
            elif sig != ref:
                worst = (name, chunk)
        check("tile-independent: %s" % name, worst is None,
              "uv %s  cov %.4f" % (ref[0], cov))
        if worst:
            break

    # ---- the helpers -----------------------------------------------------
    print()
    P = s["pts"].astype(np.float64)
    o = baker.morton_order(P)
    check("morton_order is a permutation",
          np.array_equal(np.sort(o), np.arange(len(P))))
    # the point of it: consecutive texels must be physically close, or the box test
    # cannot exclude anything
    step_sorted = np.linalg.norm(np.diff(P[o], axis=0), axis=1).mean()
    step_raw = np.linalg.norm(np.diff(P, axis=0), axis=1).mean()
    check("and it actually clusters in 3D", step_sorted < step_raw / 10.0,
          "mean neighbour step %.4f sorted vs %.4f raw (%.0fx tighter)"
          % (step_sorted, step_raw, step_raw / max(step_sorted, 1e-12)))
    check("morton_order survives a degenerate bbox",
          len(baker.morton_order(np.zeros((16, 3)))) == 16)

    c = [baker.auto_chunk(m, n) for m, n in
         ((800_000, 2_000), (800_000, 20_000), (13_400_000, 20_000), (1_000, 1))]
    check("auto_chunk stays in range and grows with the workload",
          c[0] <= c[1] <= c[2] and c[3] == 1024 and c[2] <= 16384,
          "800K/2K=%d  800K/20K=%d  13.4M/20K=%d  tiny=%d" % tuple(c))

    # ---- cost model ------------------------------------------------------
    # Not a stopwatch on absolute speed -- an assertion that the rejection still exists.
    # Without it this is 26.7 s; with it, ~1.1 s. 8 s leaves room for a slow machine while
    # still failing loudly if the O(texels x strokes) behaviour ever comes back.
    print()
    big = scene(800_000, 8_000)
    t0 = time.time()
    run(big, masks[1], Config())
    secs = time.time() - t0
    check("cost is no longer texels x strokes", secs < 8.0,
          "%.2f s (26.7 s without rejection)" % secs)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
