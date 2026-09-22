"""A stroke carries the SHADING normal at its own point, not its triangle's facet normal.

The facet normal paints a flat patch into a curved surface, which on a smooth-shaded mesh
reads as polygonal blotches. It also put the two ends of the crease guard in different
spaces -- the texel side has always been Cycles' shading normal, so a flat stroke normal
was being compared against an interpolated one.

`bridge.mesh` is imported without bpy here: only the pure-numpy half is under test, which
is the half that can be wrong arithmetically.

Run directly: python3 autostroke/tests/test_normals.py
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("bpy", types.SimpleNamespace(
    types=types.SimpleNamespace(Operator=object), data=None,
    path=types.SimpleNamespace(abspath=lambda p: p)))

from bridge import mesh as M                                  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def main():
    print("\nSTROKE NORMALS\n")
    rng = np.random.default_rng(5)
    n = 4000
    P = np.tile([0., 0., 0.], (n, 1))
    Q = np.tile([1., 0., 0.], (n, 1))
    R = np.tile([0., 1., 0.], (n, 1))
    a, b = rng.random(n), rng.random(n)
    fold = a + b > 1
    a[fold], b[fold] = 1 - a[fold], 1 - b[fold]
    pos = P + a[:, None] * (Q - P) + b[:, None] * (R - P)

    u, v, w = M.barycentric_weights(pos, P, Q, R)
    check("weights sum to 1", np.abs(u + v + w - 1).max() < 1e-9,
          "max drift %.2e" % np.abs(u + v + w - 1).max())
    check("weights reconstruct the point",
          np.abs(u[:, None] * P + v[:, None] * Q + w[:, None] * R - pos).max() < 1e-9)
    for name, p, want in (("corner P", P[0], (1, 0, 0)), ("corner Q", Q[0], (0, 1, 0)),
                          ("corner R", R[0], (0, 0, 1))):
        g = np.array(M.barycentric_weights(p[None], P[:1], Q[:1], R[:1])).ravel()
        check("%s gets weight %s" % (name, want), np.allclose(g, want, atol=1e-9))

    # Flat shading: all three corners carry the facet normal, so interpolation must be a
    # no-op. This is what makes the change safe on hard-surface meshes.
    print()
    flat = np.tile([0., 0., 1.], (n, 1))
    got = M.barycentric_normal(pos, P, Q, R, flat, flat, flat)
    check("flat-shaded mesh: interpolation changes nothing",
          np.abs(got - flat).max() < 1e-6, "max drift %.2e" % np.abs(got - flat).max())

    # Smooth shading: corners disagree, so the stroke normal must track its position.
    nP = np.tile([0., 0., 1.], (n, 1))
    nQ = np.tile([0.8, 0., 0.6], (n, 1))
    nR = np.tile([0., 0.8, 0.6], (n, 1))
    got = M.barycentric_normal(pos, P, Q, R, nP, nQ, nR)
    check("results are unit length",
          np.abs(np.linalg.norm(got, axis=1) - 1).max() < 1e-6)
    spread = np.degrees(np.arccos(np.clip((got * nP[0]).sum(1), -1, 1)))
    check("smooth-shaded mesh: the normal varies across the triangle",
          spread.max() > 20.0, "0 to %.1f deg across one face" % spread.max())
    # Continuity is the whole point, so test it where it means something: nudge each
    # point a little and the normal must move only a little. (Sorting by one barycentric
    # coordinate does NOT order points by proximity -- two points adjacent in v can sit at
    # opposite ends of the triangle in w.)
    eps = 1e-3
    near = pos.copy()
    near[:, 0] += eps * (0.5 - a)        # stays inside the triangle, moves in both axes
    near[:, 1] += eps * (0.5 - b)
    got2 = M.barycentric_normal(near, P, Q, R, nP, nQ, nR)
    step = np.degrees(np.arccos(np.clip((got * got2).sum(1), -1, 1)))
    check("and varies smoothly, not in steps", step.max() < 1.0,
          "a %.0e nudge moves the normal at most %.4f deg" % (eps, step.max()))

    # A corner must return exactly that corner's normal.
    got1 = M.barycentric_normal(Q[:1], P[:1], Q[:1], R[:1], nP[:1], nQ[:1], nR[:1])
    check("at a corner it is that corner's normal",
          np.allclose(got1[0], nQ[0] / np.linalg.norm(nQ[0]), atol=1e-6))

    # Degenerate triangle: must not produce NaN, since a mesh can always contain one.
    z = np.zeros((1, 3))
    d = M.barycentric_normal(z, z, z, z, nP[:1], nQ[:1], nR[:1])
    check("a degenerate triangle yields no NaN", np.isfinite(d).all(), str(d[0].round(3)))

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
