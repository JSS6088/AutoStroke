"""Who wins a contested texel: the smallest stroke, by final rendered radius.

The resolver awards each texel to the FIRST candidate in its sort order that covers it, so
the sort order IS the tie-break rule. It sorts on `seed_radius * Stroke Size * size
variation` -- the mark that actually lands, not the patch of surface it came from -- so a
fine mark is never swallowed by a coarse one.

This replaced a two-tier split at the median radius, which only guaranteed that SOME stroke
from the smaller half beat SOME stroke from the larger; inside a tier the winner fell to
id, which is arbitrary. Measured on a corrugated fixture, the straight sort moved 53% of
contested texels and improved per-texel normal error from median 1.9 to 1.5 deg (p90 4.3 to
3.6), because a smaller stamp fits the surface under it better.

Pure numpy. Run directly: python3 autostroke/tests/test_ordering.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import baker                                       # noqa: E402
from core.config import Config                               # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def scene(n_pts=6000, n_seeds=60, seed=3):
    """Few strokes, wide radius spread, heavy overlap -- so texels are actually contested."""
    rng = np.random.default_rng(seed)
    u, v = rng.random(n_pts) * 4.0, rng.random(n_pts) * 4.0
    pts = np.stack([u, v, 0.3 * np.sin(u) * np.cos(v)], 1).astype(np.float32)
    du = np.stack([np.ones_like(u), np.zeros_like(u), 0.3 * np.cos(u) * np.cos(v)], 1)
    dv = np.stack([np.zeros_like(u), np.ones_like(u), -0.3 * np.sin(u) * np.sin(v)], 1)
    n = np.cross(du, dv); n /= np.linalg.norm(n, axis=1, keepdims=True)
    su, sv = rng.random(n_seeds) * 4.0, rng.random(n_seeds) * 4.0
    pos = np.stack([su, sv, 0.3 * np.sin(su) * np.cos(sv)], 1).astype(np.float32)
    sdu = np.stack([np.ones_like(su), np.zeros_like(su), 0.3 * np.cos(su) * np.cos(sv)], 1)
    sdv = np.stack([np.zeros_like(su), np.ones_like(su), -0.3 * np.sin(su) * np.sin(sv)], 1)
    sn = np.cross(sdu, sdv); sn /= np.linalg.norm(sn, axis=1, keepdims=True)
    return dict(pts=pts, pn=n.astype(np.float32), pos=pos, nrm=sn.astype(np.float32),
                uv=np.stack([np.arange(n_seeds) / n_seeds + 0.001,
                             np.full(n_seeds, 0.5)], 1).astype(np.float32),
                rad=(0.04 + 0.30 * rng.random(n_seeds)).astype(np.float32),
                ids=np.arange(n_seeds, dtype=np.int64))


def run(s, cfg, k=slice(None)):
    mask = np.ones((64, 64), np.float32)
    return baker.resolve_uv_blocking(
        s["pts"], np.zeros((len(s["pts"]), 2), np.float32), s["pos"][k], s["ids"][k],
        s["uv"][k], s["rad"][k], s["nrm"][k], None, mask, cfg,
        s["nrm"][k].astype(np.float64), 0.0, pt_nrm=s["pn"])


def winners(s, cfg):
    """Index of the stroke that won each texel, or -1."""
    r = run(s, cfg)
    key = {round(float(u[0]), 6): i for i, u in enumerate(s["uv"])}
    w = np.array([key.get(round(float(u[0]), 6), -1) for u in r[0]])
    return np.where(r[2], w, -1), r


def main():
    print("\nCANDIDATE ORDER\n")
    s = scene()

    for label, cfg in (("no size variation", Config()),
                       ("size variation 1.0", Config(size_random=1.0))):
        won, _ = winners(s, cfg)
        # Brute force: run each stroke ALONE to learn exactly which texels it would take,
        # then the winner among those candidates must be the one with the smallest radius.
        # Solving it independently of the resolver is the point -- asserting against the
        # resolver's own ordering would pass even if the rule were wrong.
        n = len(s["ids"])
        claims = np.zeros((n, len(s["pts"])), bool)
        for j in range(n):
            claims[j] = run(s, cfg, slice(j, j + 1))[2]
        var = float(cfg.size_random)
        r_eff = s["rad"].astype(np.float64) * cfg.mask_scale
        if var:
            r_eff = r_eff * 2.0 ** (
                (baker.hash01(s["ids"] ^ np.int64(baker.SALT_SIZE)) * 2.0 - 1.0) * var)

        painted = np.flatnonzero(won >= 0)
        contested = 0
        wrong = 0
        for t in painted:
            cand = np.flatnonzero(claims[:, t])
            if len(cand) < 2:
                continue
            contested += 1
            best = cand[np.lexsort((-s["ids"][cand], r_eff[cand]))[0]]
            wrong += int(won[t] != best)
        check("smallest stroke wins every contested texel (%s)" % label, wrong == 0,
              "%d contested, %d wrong" % (contested, wrong))

    # The sort is on the FINAL radius, so changing only size variation must move winners.
    a, _ = winners(s, Config())
    b, _ = winners(s, Config(size_random=1.0))
    both = (a >= 0) & (b >= 0)
    check("size variation changes who wins, not just how big",
          (a[both] != b[both]).mean() > 0.05,
          "%.0f%% of painted texels change hands" % (100 * (a[both] != b[both]).mean()))

    check("deterministic across runs",
          np.array_equal(winners(s, Config(size_random=1.0))[0], b))

    # Exact ties fall to id descending -- the only tie-break left.
    t = scene()
    t["rad"] = np.full(len(t["ids"]), 0.15, np.float32)
    wt, _ = winners(t, Config())
    dup = np.flatnonzero(wt >= 0)
    ok = True
    for x in dup[:400]:
        cand = np.flatnonzero(np.array([run(t, Config(), slice(j, j + 1))[2][x]
                                        for j in range(len(t["ids"]))]))
        if len(cand) > 1 and wt[x] != cand.max():
            ok = False
            break
    check("on equal radius the higher id wins", ok)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
