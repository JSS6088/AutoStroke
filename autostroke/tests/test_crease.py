"""The crease guard: does it stop at the fold, and does it stop in the SAME place?

The old guard tested distance off the stroke's tangent plane against k*radius. It kept
strokes off the far side of a fold, but the cut moved with the stroke's own radius, so
along one fold every stroke stopped somewhere different and the strokes looked chopped
mid-shape. These tests pin the two properties that fixed it: the cut is a property of the
surface (an angle), and the guard can never leave bare surface.

Pure numpy -- no Blender. Run directly: python3 autostroke/tests/test_crease.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import baker                                       # noqa: E402
from core.config import Config                               # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("   %-52s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def folded_plane(n=120, half=1.0):
    """A 90 degree fold: y=0 for x<0, and x=0 for y<0. Points plus true normals."""
    t = np.linspace(0.02, half, n)
    z = np.linspace(-half, half, n)
    T, Z = np.meshgrid(t, z, indexing="ij")
    a = np.stack([-T.ravel(), np.zeros(T.size), Z.ravel()], 1)   # floor, normal +y
    b = np.stack([np.zeros(T.size), -T.ravel(), Z.ravel()], 1)   # wall,  normal +x
    pts = np.concatenate([a, b]).astype(np.float32)
    nrm = np.concatenate([np.tile([0, 1., 0], (len(a), 1)),
                          np.tile([1., 0, 0], (len(b), 1))]).astype(np.float32)
    side = np.concatenate([np.zeros(len(a), np.int8), np.ones(len(b), np.int8)])
    return pts, nrm, side


def run(pts, nrm, pos, snrm, radius, cfg):
    n = len(pos)
    mask = np.ones((64, 64), np.float32)           # a square stamp: shape is not the subject
    return baker.resolve_uv_blocking(
        pts, np.zeros((len(pts), 2), np.float32),
        pos.astype(np.float32), np.arange(n, dtype=np.int64),
        np.stack([np.linspace(0.1, 0.9, n), np.full(n, 0.5)], 1).astype(np.float32),
        np.asarray(radius, np.float32), snrm.astype(np.float32), None, mask, cfg,
        snrm.astype(np.float64), 0.0, pt_nrm=nrm)


def main():
    print("\nCREASE GUARD\n")
    pts, nrm, side = folded_plane()
    # one stroke per side, both big enough to reach well around the fold
    pos = np.array([[-0.4, 0.0, 0.0], [0.0, -0.4, 0.0]])
    snrm = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rad = np.array([0.9, 0.9])

    off = run(pts, nrm, pos, snrm, rad, Config(crease_guard=False))
    on = run(pts, nrm, pos, snrm, rad, Config(crease_guard=True, crease_angle_deg=45.0))

    def bleed(res):
        """Fraction of painted texels whose stroke faces >45 deg away from the surface."""
        d = res[1][res[2]] * 2.0 - 1.0
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        ang = np.degrees(np.arccos(np.clip((d * nrm[res[2]]).sum(1), -1, 1)))
        return float((ang > 45.0).mean())

    check("guard off lets a stamp paint across a 90 deg fold", bleed(off) > 0.2,
          "%.0f%% of the surface is painted by the wrong side" % (100 * bleed(off)))
    check("guard on keeps each side to its own stroke", bleed(on) < 0.01,
          "%.1f%% wrong-side" % (100 * bleed(on)))

    # Rejecting COSTS coverage, and that is the intended trade. An earlier version gave
    # the texel the least-wrong stroke instead so that coverage never fell; that painted
    # normals facing into the model, which render black. Uncovered is not a hole -- the
    # uncovered path writes the true surface normal, so the texel shades like the
    # untouched model.
    check("the guard never invents coverage", on[2].mean() <= off[2].mean() + 1e-9,
          "off %.1f%%  on %.1f%%" % (100 * off[2].mean(), 100 * on[2].mean()))

    # One stroke, both sides of the fold: the far side has no candidate that agrees, so
    # it must be left alone entirely -- coverage genuinely drops, and that is the trade.
    solo = run(pts, nrm, pos[:1], snrm[:1], rad[:1],
               Config(crease_guard=True, crease_angle_deg=45.0))
    solo_off = run(pts, nrm, pos[:1], snrm[:1], rad[:1], Config(crease_guard=False))
    far = side == 1
    check("a texel no stroke agrees with is left to the original normal",
          not solo[2][far].any(),
          "%d of %d far-side texels painted" % (int(solo[2][far].sum()), int(far.sum())))
    check("and that costs coverage, rather than painting a bad normal",
          solo[2].mean() < solo_off[2].mean() - 0.1,
          "guard off %.0f%% -> on %.0f%%" % (100 * solo_off[2].mean(), 100 * solo[2].mean()))

    # A smoothly curving surface is where the two criteria part company: on a quarter
    # cylinder the normal turns steadily, so "how far does a stroke wrap" is exactly the
    # question the guard answers. Two strokes, so a rejected texel has somewhere to go and
    # the fallback is not what is being measured.
    n = 400
    th = np.linspace(0.0, np.pi / 2, n)
    cyl = np.stack([np.cos(th), np.sin(th), np.zeros(n)], 1).astype(np.float32)
    ends = cyl[[0, n - 1]]                       # one stroke at each end of the arc

    # Radius independence -- the property the old k*radius test did not have. The far
    # stroke is held fixed; only the near stroke's radius changes.
    # Candidate order puts the higher id first, so the stroke at 90 deg claims everything
    # the guard lets it have and the other takes the rest -- the boundary between them IS
    # the cut, and it should sit `crease_angle` away from the 90 deg stroke.
    def boundary(radii, angle):
        res = run(cyl, cyl.copy(), ends, ends.copy(), np.asarray(radii),
                  Config(crease_guard=True, crease_angle_deg=angle))
        far = res[2] & (np.abs(res[0][:, 0] - 0.9) < 1e-6)
        return float(np.degrees(th[far].min())) if far.any() else 90.0

    stops = [boundary([r, 1.6], 40.0) for r in (0.6, 1.6)]
    check("cut position does not move with stroke radius",
          abs(stops[0] - stops[1]) < 0.5,
          "other stroke r=0.6 -> cut at %.1f deg, r=1.6 -> %.1f deg" % tuple(stops))

    reach = [boundary([1.6, 1.6], a) for a in (75.0, 60.0)]
    check("Stroke Cutoff sets how far a stroke wraps a curve",
          abs(reach[0] - 15.0) < 1.5 and abs(reach[1] - 30.0) < 1.5,
          "75 deg -> cut at %.0f deg, 60 deg -> cut at %.0f deg (want 15 / 30)" % tuple(reach))

    # Tighten the angle past the point where the two strokes can still meet and the band
    # in the middle belongs to neither. It is then left BARE -- each stroke stops at its
    # own 20 deg, and the 50 deg between them keeps the original surface normal. An
    # earlier version split that band between the two least-wrong strokes instead, which
    # is precisely how normals facing the wrong way got painted.
    res = run(cyl, cyl.copy(), ends, ends.copy(), np.array([1.6, 1.6]),
              Config(crease_guard=True, crease_angle_deg=20.0))
    painted = np.degrees(th[res[2]])
    gap = float(((painted > 25) & (painted < 65)).sum())
    check("too tight to reach: the band goes bare, it is not split",
          gap == 0 and res[2].mean() < 0.55,
          "nothing painted between 25 and 65 deg; %.0f%% of the arc covered"
          % (100 * res[2].mean()))

    # The invariant the whole change exists for: nothing painted is ever further from its
    # own surface than the dial allows. This is the test that would have caught the black
    # speckles, and it fails the moment a least-wrong fallback comes back.
    worst = []
    for a in (20.0, 45.0, 70.0):
        r = run(cyl, cyl.copy(), ends, ends.copy(), np.array([1.6, 1.6]),
                Config(crease_guard=True, crease_angle_deg=a))
        d = r[1][r[2]] * 2.0 - 1.0
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        err = np.degrees(np.arccos(np.clip((d * cyl[r[2]]).sum(1), -1, 1)))
        worst.append(err.max() - a)
    check("no painted texel ever exceeds the Stroke Cutoff", max(worst) < 0.5,
          "worst overshoot across 20/45/70 deg: %.2f deg" % max(worst))

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
