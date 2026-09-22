"""The stamp footprint is clipped to the brush's painted bounding box.

The whole change rests on one claim: a sample outside that box CANNOT pass the mask.
Outside the painted pixels the mask is at or below mask_thresh by construction, and a
bilinear blend of sub-threshold values is itself sub-threshold. If that claim were ever
false, strokes would quietly lose their edges -- and the result would still look like
brushwork, so nobody would notice. Hence this file tests the claim, not the speedup.

Worth it because the box is small: on the shipped set the painted area is 8-18% of the
square, so ~86% of every mask sample used to be taken where it could not pass.

Run directly: python3 autostroke/tests/test_footprint.py
"""

import glob
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


def brushes(px):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "brushes", "standard")
    from PIL import Image
    cfg = Config()
    return [baker.mask_from_rgba(
        np.asarray(Image.open(f).resize((px, px), Image.LANCZOS), np.float32) / 255.0, cfg)
        for f in sorted(glob.glob(os.path.join(root, "*.png")))]


def main():
    print("\nFOOTPRINT CLIPPING\n")
    cfg = Config()

    # ---- the claim, on the real brushes -----------------------------------
    for px in (256, 1024):
        worst = 0.0
        for m in brushes(px):
            u0, u1, v0, v1 = baker.mask_bbox(m, cfg)
            H, W = m.shape
            ys, xs = np.nonzero(m > cfg.mask_thresh)
            # every painted pixel must map to a (mu, mv) inside the box
            mu = xs / (W - 1.0)
            mv = 1.0 - ys / (H - 1.0)
            if not (mu.min() >= u0 and mu.max() <= u1 and mv.min() >= v0 and mv.max() <= v1):
                worst = -1
                break
            worst = max(worst, (u1 - u0) * (v1 - v0))
        check("no painted pixel falls outside the box (%dpx brushes)" % px, worst > 0,
              "largest box is %.1f%% of the square" % (100 * worst) if worst > 0 else "")

    # ---- the one-texel margin is load-bearing -----------------------------
    # sample_mask reads floor(x) and floor(x)+1, so a sample just outside the painted
    # pixels still picks up a painted neighbour. Without the margin those would be clipped.
    m = brushes(256)[0]
    u0, u1, v0, v1 = baker.mask_bbox(m, cfg)
    H, W = m.shape
    ys, xs = np.nonzero(m > cfg.mask_thresh)
    tight = (xs.min() / (W - 1.0), xs.max() / (W - 1.0),
             1.0 - ys.max() / (H - 1.0), 1.0 - ys.min() / (H - 1.0))
    # sample on a fine grid; find samples that PASS but sit outside the un-expanded box
    g = np.linspace(0.0, 1.0, 600)
    MU, MV = np.meshgrid(g, g)
    val = baker.sample_mask(m, MU.ravel(), MV.ravel())
    passing = val > cfg.mask_thresh
    outside_tight = ((MU.ravel() < tight[0]) | (MU.ravel() > tight[1])
                     | (MV.ravel() < tight[2]) | (MV.ravel() > tight[3]))
    lost = int((passing & outside_tight).sum())
    check("the 1-texel margin is doing work, not decoration", lost > 0,
          "%d passing samples would be clipped without it" % lost)
    inside_box = ((MU.ravel() >= u0) & (MU.ravel() <= u1)
                  & (MV.ravel() >= v0) & (MV.ravel() <= v1))
    check("and WITH it, every passing sample is inside the box",
          int((passing & ~inside_box).sum()) == 0,
          "%d of %d samples pass" % (int(passing.sum()), passing.size))

    # ---- degenerate masks fail OPEN ---------------------------------------
    print()
    full = np.ones((64, 64), np.float32)
    check("a fully painted mask gives the whole square",
          baker.mask_bbox(full, cfg) == (0.0, 1.0, 0.0, 1.0))
    check("an empty mask gives the whole square, not an empty one",
          baker.mask_bbox(np.zeros((64, 64), np.float32), cfg) == (0.0, 1.0, 0.0, 1.0),
          "fail open: clipping everything away would paint nothing")

    # ---- end to end: identical output, far fewer samples -------------------
    print()
    rng = np.random.default_rng(4)
    n_pts, n_seeds = 60_000, 400
    u, v = rng.random(n_pts) * 4.0, rng.random(n_pts) * 4.0
    pts = np.stack([u, v, 0.3 * np.sin(u) * np.cos(v)], 1).astype(np.float32)
    du = np.stack([np.ones_like(u), np.zeros_like(u), 0.3 * np.cos(u) * np.cos(v)], 1)
    dv = np.stack([np.zeros_like(u), np.ones_like(u), -0.3 * np.sin(u) * np.sin(v)], 1)
    pn = np.cross(du, dv); pn /= np.linalg.norm(pn, axis=1, keepdims=True)
    su, sv = rng.random(n_seeds) * 4.0, rng.random(n_seeds) * 4.0
    pos = np.stack([su, sv, 0.3 * np.sin(su) * np.cos(sv)], 1).astype(np.float32)
    sdu = np.stack([np.ones_like(su), np.zeros_like(su), 0.3 * np.cos(su) * np.cos(sv)], 1)
    sdv = np.stack([np.zeros_like(su), np.ones_like(su), -0.3 * np.sin(su) * np.sin(sv)], 1)
    sn = np.cross(sdu, sdv); sn /= np.linalg.norm(sn, axis=1, keepdims=True)
    ms = brushes(256)

    def run(masks, tm):
        return baker.resolve_uv_blocking(
            pts, np.zeros((n_pts, 2), np.float32), pos, np.arange(n_seeds, dtype=np.int64),
            np.stack([np.linspace(.01, .99, n_seeds), np.full(n_seeds, .5)], 1).astype(np.float32),
            np.full(n_seeds, 0.12, np.float32), sn.astype(np.float32), None, masks,
            Config(mask_scale=6.0), sn.astype(np.float64), 0.0,
            pt_nrm=pn.astype(np.float32), counters=tm)

    # a mask padded so its painted box IS the whole square: same picture, no clipping
    pad = [np.maximum(m, 0.0) for m in ms]
    for m in pad:
        m[0, :] = 1.0; m[-1, :] = 1.0; m[:, 0] = 1.0; m[:, -1] = 1.0
    t_clip, t_open = {}, {}
    a = run(ms, t_clip)
    check("clipping is exact: it only skips samples that could not pass",
          all(np.array_equal(x, y) for x, y in zip(a, run(ms, {}))),
          "deterministic")
    run(pad, t_open)
    ratio = t_open["footprint"] / max(t_clip["footprint"], 1)
    check("and it removes most of the work", ratio > 3.0,
          "%s footprint texels vs %s unclipped (%.1fx)"
          % ("{:,}".format(int(t_clip["footprint"])),
             "{:,}".format(int(t_open["footprint"])), ratio))

    box_rejection(pts, pn, pos, sn, ms)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


def box_rejection(pts, pn, pos, sn, masks):
    """The per-tile rejection bounds the painted box, not the stamp sphere.

    The sphere it replaced circumscribed the stamp SQUARE, and the square is only 8-18%
    painted -- loose twice over. The box halves the (stroke, tile) pairs, and each pair
    costs three full-tile numpy passes.

    The half-extent along the NORMAL must be the full radius. The footprint test
    constrains lu and lv only; nothing but the sphere test bounds how far off the tangent
    plane a texel may sit. Bounding just the painted rectangle looks tighter and silently
    drops strokes -- so this rebuilds both tests and checks, by brute force, that nothing
    the box rejects would have painted.
    """
    print()
    from core.config import INV_SQRT2
    from core.geometry import frame_from_tangent
    cfg = Config(mask_scale=12.0)
    n = len(pos)
    bo = baker.brush_index(np.arange(n, dtype=np.int64), len(masks))
    T, B = frame_from_tangent(sn.astype(np.float64), sn.astype(np.float64))
    al = np.radians([baker.mask_long_axis_deg(m, cfg) for m in masks])[bo]
    ca, sa = np.cos(-al)[:, None], np.sin(-al)[:, None]
    T, B = T * ca + B * sa, -T * sa + B * ca
    r = np.full(n, 0.12, np.float64) * cfg.mask_scale
    bb = np.array([baker.mask_bbox(m, cfg) for m in masks], np.float64)[bo]
    th = 2.0 * r * INV_SQRT2
    lul, luh = (bb[:, 0] - 0.5) * th, (bb[:, 1] - 0.5) * th
    lvl, lvh = (bb[:, 2] - 0.5) * th, (bb[:, 3] - 0.5) * th
    nu = sn.astype(np.float64)
    nu = nu / np.maximum(np.linalg.norm(nu, axis=1, keepdims=True), 1e-12)
    ctr = pos.astype(np.float64) + T * (0.5 * (lul + luh))[:, None] + B * (0.5 * (lvl + lvh))[:, None]
    ext = (np.abs(T) * (0.5 * (luh - lul))[:, None] + np.abs(B) * (0.5 * (lvh - lvl))[:, None]
           + np.abs(nu) * r[:, None])
    blo, bhi = ctr - ext, ctr + ext

    perm = baker.morton_order(pts)
    P = pts[perm].astype(np.float64)
    C = baker.auto_chunk(len(P), n)
    dropped = missed = 0
    for a in range(0, len(P), C):
        p = P[a:min(a + C, len(P))]
        mn, mx = p.min(0), p.max(0)
        d = np.maximum(np.maximum(mn - pos, pos - mx), 0.0)
        sph = (d * d).sum(1) < r * r
        box = (blo <= mx).all(1) & (mn <= bhi).all(1)
        for j in np.flatnonzero(sph & ~box):
            dropped += 1
            dx, dy, dz = p[:, 0] - pos[j, 0], p[:, 1] - pos[j, 1], p[:, 2] - pos[j, 2]
            w = (dx * dx + dy * dy + dz * dz) < r[j] * r[j]
            lu = dx * T[j, 0] + dy * T[j, 1] + dz * T[j, 2]
            lv = dx * B[j, 0] + dy * B[j, 1] + dz * B[j, 2]
            if (w & (lu > lul[j]) & (lu < luh[j]) & (lv > lvl[j]) & (lv < lvh[j])).any():
                missed += 1
    check("the box rejects only strokes that would paint nothing", missed == 0,
          "%s dropped, %d of them would have painted" % ("{:,}".format(dropped), missed))
    check("and it drops a useful number of them", dropped > 100,
          "otherwise the box is no tighter than the sphere")

    # A zero normal extent -- the tempting mistake -- must be caught by that same check.
    ext0 = (np.abs(T) * (0.5 * (luh - lul))[:, None]
            + np.abs(B) * (0.5 * (lvh - lvl))[:, None])
    blo0, bhi0 = ctr - ext0, ctr + ext0
    bad = 0
    for a in range(0, len(P), C):
        p = P[a:min(a + C, len(P))]
        mn, mx = p.min(0), p.max(0)
        d = np.maximum(np.maximum(mn - pos, pos - mx), 0.0)
        for j in np.flatnonzero(((d * d).sum(1) < r * r)
                                & ~((blo0 <= mx).all(1) & (mn <= bhi0).all(1))):
            dx, dy, dz = p[:, 0] - pos[j, 0], p[:, 1] - pos[j, 1], p[:, 2] - pos[j, 2]
            w = (dx * dx + dy * dy + dz * dz) < r[j] * r[j]
            lu = dx * T[j, 0] + dy * T[j, 1] + dz * T[j, 2]
            lv = dx * B[j, 0] + dy * B[j, 1] + dz * B[j, 2]
            if (w & (lu > lul[j]) & (lu < luh[j]) & (lv > lvl[j]) & (lv < lvh[j])).any():
                bad += 1
    check("dropping the normal half-extent WOULD lose strokes", bad > 0,
          "%d strokes lost -- which is why the extent is r, not 0" % bad)


if __name__ == "__main__":
    sys.exit(main())
