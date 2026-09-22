"""Per-stroke variation: which brush a stroke draws, and how big it comes out.

What has to hold: the refactor is a refactor (one brush gives byte-identical output), the
draw is deterministic and even, it is independent of the other per-stroke randoms, and
each brush's own long axis -- not some other brush's -- is what gets lined up with the
stroke direction.

Pure numpy plus a tiny bpy stub for the folder scan. Run directly:
    python3 autostroke/tests/test_brushes.py
"""

import hashlib
import os
import shutil
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import baker                                       # noqa: E402
from core.config import Config                               # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---- a fixed scene, so byte-identity is a meaningful claim -----------------
def scene(n_pts=30000, n_seeds=500, seed=7):
    rng = np.random.default_rng(seed)
    u = rng.random(n_pts) * 6.0
    v = rng.random(n_pts) * 6.0
    pts = np.stack([u, v, 0.6 * np.sin(u) * np.cos(v * 0.7)], 1).astype(np.float32)
    du = np.stack([np.ones_like(u), np.zeros_like(u), 0.6 * np.cos(u) * np.cos(v * 0.7)], 1)
    dv = np.stack([np.zeros_like(u), np.ones_like(u), -0.42 * np.sin(u) * np.sin(v * 0.7)], 1)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    su, sv = rng.random(n_seeds) * 6.0, rng.random(n_seeds) * 6.0
    spos = np.stack([su, sv, 0.6 * np.sin(su) * np.cos(sv * 0.7)], 1).astype(np.float32)
    sdu = np.stack([np.ones_like(su), np.zeros_like(su), 0.6 * np.cos(su) * np.cos(sv * 0.7)], 1)
    sdv = np.stack([np.zeros_like(su), np.ones_like(su), -0.42 * np.sin(su) * np.sin(sv * 0.7)], 1)
    sn = np.cross(sdu, sdv)
    sn /= np.linalg.norm(sn, axis=1, keepdims=True)
    return dict(pts=pts, pt_nrm=n.astype(np.float32), pos=spos, nrm=sn.astype(np.float32),
                uv=np.stack([np.linspace(0.05, 0.95, n_seeds), rng.random(n_seeds)], 1
                            ).astype(np.float32),
                rad=(0.10 + 0.10 * rng.random(n_seeds)).astype(np.float32),
                ids=np.arange(n_seeds, dtype=np.int64))


def bar(px=128, w=0.30, h=0.08, angle=0.0):
    """A rectangle at `angle` degrees -- a brush whose long axis is known exactly."""
    y, x = (np.mgrid[0:px, 0:px] / (px - 1.0)) - 0.5
    t = np.radians(angle)
    xr = x * np.cos(t) + y * np.sin(t)
    yr = -x * np.sin(t) + y * np.cos(t)
    return ((np.abs(xr) < w) & (np.abs(yr) < h)).astype(np.float32)


def run(s, mask, cfg, rot=0.0):
    return baker.resolve_uv_blocking(
        s["pts"], np.zeros((len(s["pts"]), 2), np.float32), s["pos"], s["ids"],
        s["uv"], s["rad"], s["nrm"], None, mask, cfg,
        s["nrm"].astype(np.float64), rot, pt_nrm=s["pt_nrm"])


def main():
    print("\nBRUSH SETS\n")
    s = scene()
    cfg = Config()
    one = bar(angle=0.0)

    # A set of one must behave exactly like the single brush it replaces, or every number
    # measured before this change stops being comparable.
    a = run(s, one, cfg)
    b = run(s, [one], cfg)
    check("a set of one is byte-identical to a bare mask",
          all(digest(a[i]) == digest(b[i]) for i in (0, 1, 2, 3)),
          "uv %s" % digest(a[0]))

    # Determinism and spread.
    ids = np.arange(20000, dtype=np.int64)
    idx = baker.brush_index(ids, 4)
    check("the draw is deterministic",
          np.array_equal(idx, baker.brush_index(ids, 4)))
    counts = np.bincount(idx, minlength=4) / len(ids)
    check("the draw is even across the set", np.abs(counts - 0.25).max() < 0.01,
          "shares " + " ".join("%.3f" % c for c in counts))
    check("order does not matter",
          np.array_equal(baker.brush_index(ids[::-1], 4), idx[::-1]))
    check("a set of one always draws brush 0",
          (baker.brush_index(ids, 1) == 0).all())

    # Independent of the other two per-stroke draws, or shape correlates with tone and the
    # set stops breaking up the pattern.
    lum = baker.hash01(ids)
    jit = baker.hash01(ids ^ np.int64(0x9e3779b9))
    cl = abs(float(np.corrcoef(idx, lum)[0, 1]))
    cj = abs(float(np.corrcoef(idx, jit)[0, 1]))
    check("shape is independent of tone and of rotation jitter",
          cl < 0.02 and cj < 0.02, "corr vs tone %.4f, vs jitter %.4f" % (cl, cj))

    # Each brush must be aligned by ITS OWN long axis. Two bars 90 deg apart: if one
    # brush's offset were applied to both, one of them comes out across the flow.
    check("mask_long_axis_deg reads each bar correctly",
          abs(baker.mask_long_axis_deg(bar(angle=0.0), cfg)) < 1.0
          and abs(abs(baker.mask_long_axis_deg(bar(angle=90.0), cfg)) - 90.0) < 1.0,
          "%.1f and %.1f deg" % (baker.mask_long_axis_deg(bar(angle=0.0), cfg),
                                 baker.mask_long_axis_deg(bar(angle=90.0), cfg)))

    # With auto-align on, a set of {bar, bar rotated 90} must paint the SAME footprint as
    # a set of {bar, bar}: aligning each brush by its own axis cancels the 90 deg out.
    aligned = Config(mask_auto_align=True)
    same = run(s, [bar(angle=0.0), bar(angle=0.0)], aligned)
    mixed = run(s, [bar(angle=0.0), bar(angle=90.0)], aligned)
    check("auto-align uses each brush's own axis, not one for all",
          digest(same[0]) == digest(mixed[0]),
          "identical footprint from two differently-drawn sets")

    # And with auto-align OFF the same pair must NOT match -- otherwise the test above
    # would pass for the boring reason that the brush index is being ignored.
    raw = Config(mask_auto_align=False)
    check("brush choice actually reaches the stamp",
          digest(run(s, [bar(angle=0.0), bar(angle=0.0)], raw)[0])
          != digest(run(s, [bar(angle=0.0), bar(angle=90.0)], raw)[0]))

    # Coverage tracks the set: heavier brushes paint more. No normalisation, by design.
    thin = run(s, [bar(w=0.30, h=0.04)], cfg)[2].mean()
    fat = run(s, [bar(w=0.30, h=0.16)], cfg)[2].mean()
    check("a heavier set paints more (nothing is normalised away)", fat > thin * 1.5,
          "thin %.1f%% vs fat %.1f%%" % (100 * thin, 100 * fat))

    check("an empty mask list is refused", refused(lambda: run(s, [], cfg)))

    # ---- size variation ---------------------------------------------------
    print()
    ids = np.arange(40000, dtype=np.int64)

    def factor(r):
        return 2.0 ** ((baker.hash01(ids ^ np.int64(baker.SALT_SIZE)) * 2.0 - 1.0) * r)

    f1 = factor(1.0)
    check("variation 0 leaves every stroke at the dial value",
          np.array_equal(factor(0.0), np.ones(len(ids))))
    check("variation 1 spans half to double",
          abs(f1.min() - 0.5) < 1e-3 and abs(f1.max() - 2.0) < 1e-3,
          "%.4f .. %.4f" % (f1.min(), f1.max()))
    # Exponential, not linear: halving has to be as likely as doubling, or the whole
    # mesh drifts bigger as the dial goes up.
    check("halving is as likely as doubling",
          abs((f1 < 1.0).mean() - 0.5) < 0.01 and abs(np.median(f1) - 1.0) < 0.01,
          "%.1f%% below 1.0, median %.4f" % (100 * (f1 < 1.0).mean(), np.median(f1)))
    check("Stroke Size 5 at variation 1 gives 2.5 to 10",
          abs((5 * f1).min() - 2.5) < 1e-2 and abs((5 * f1).max() - 10.0) < 1e-2,
          "%.3f .. %.3f" % ((5 * f1).min(), (5 * f1).max()))
    check("the draw is deterministic", np.array_equal(factor(1.0), f1))
    # Independent of the other three per-stroke draws, or size correlates with shade and
    # the variations stop reading as variation.
    cs = [abs(float(np.corrcoef(f1, d)[0, 1])) for d in
          (baker.hash01(ids), baker.hash01(ids ^ np.int64(0x9e3779b9)),
           baker.brush_index(ids, 4).astype(np.float64))]
    check("size is independent of tone, jitter and brush choice", max(cs) < 0.02,
          "corr %.4f / %.4f / %.4f" % tuple(cs))

    # And it reaches the stamp: bigger strokes must paint more.
    plain = run(s, one, Config(size_random=0.0))[2].mean()
    varied = run(s, one, Config(size_random=1.0))[2].mean()
    check("variation changes what gets painted", abs(varied - plain) > 0.01,
          "coverage %.1f%% -> %.1f%%" % (100 * plain, 100 * varied))

    # ---- the folder scan, with bpy stubbed --------------------------------
    print()
    root = tempfile.mkdtemp()
    try:
        for name, files in (("standard", 3), ("rough", 2), ("notes", 0)):
            os.makedirs(os.path.join(root, name))
            for i in range(files):
                open(os.path.join(root, name, "b%d.png" % i), "wb").close()
        open(os.path.join(root, "loose.png"), "wb").close()
        sys.modules["bpy"] = types.SimpleNamespace(
            path=types.SimpleNamespace(abspath=lambda p: p), data=None, types=None)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from bridge import brushes as B

        st = types.SimpleNamespace(brush_dir=root, brush_set="rough")
        check("only subfolders holding images count as sets",
              set(B.sets(root)) == {"rough", "standard"}, str(B.sets(root)))
        # First in the list IS the default -- a dynamic enum cannot take default=.
        check("standard is listed first, so a fresh scene gets it",
              B.sets(root)[0] == "standard" and B.set_items(st, None)[0][0] == "standard",
              str(B.sets(root)))
        check("the chosen set's files come back sorted",
              [os.path.basename(p) for p in B.files(st)[2]] == ["b0.png", "b1.png"])
        st.brush_set = 'NONE'
        check("an unset enum falls back to the first set -- i.e. standard",
              B.files(st)[1] == B.sets(root)[0] == "standard", B.files(st)[1])

        items = B.set_items(st, None)
        check("enum items are cached, not rescanned per redraw",
              B.set_items(st, None) is B._SET_ITEMS and len(items) == 2)
        os.makedirs(os.path.join(root, "extra"))
        open(os.path.join(root, "extra", "b.png"), "wb").close()
        check("a folder added mid-session needs Refresh", len(B.set_items(st, None)) == 2)
        B.refresh()
        check("Refresh picks it up", len(B.set_items(st, None)) == 3)

        st.brush_dir = os.path.join(root, "notes")
        check("an empty folder reports no sets rather than crashing",
              B.set_items(st, None)[0][0] == 'NONE')

        # A folder with no 'standard' keeps plain alphabetical order.
        B.refresh()
        st.brush_dir = root
        os.rename(os.path.join(root, "standard"), os.path.join(root, "zzz"))
        check("without a standard set the order is just alphabetical",
              B.sets(root) == ["extra", "rough", "zzz"], str(B.sets(root)))
    finally:
        shutil.rmtree(root, ignore_errors=True)
        sys.modules.pop("bpy", None)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


def refused(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
