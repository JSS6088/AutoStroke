"""The stroke budget, and the Min/Max per Face clamp that keeps you away from it.

Min Strokes per Face is a floor applied to EVERY face, so it multiplies by the face count
and ignores Stroke Count: 64 on a 20,000-face mesh is 1,280,000 strokes. The direction
field's kNN is O(N^2) and runs on Blender's main thread, so that combination froze Blender
hard enough to need killing.

Two things are tested, because two things have to hold:
  * the arithmetic the guard rests on -- that the floor really does multiply
  * that the clamp lands on the LARGEST value that fits, not merely a safe one. A clamp
    that snapped to some round number would pass a "no crash" test while quietly taking
    away settings that were fine.

Run directly: python3 autostroke/tests/test_budget.py
"""

import ast
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import sampling as S                                # noqa: E402


def _load_pure():
    """The budget half of bridge/seeds.py, lifted out of the shipping source.

    Importing the module drags in bpy through bridge.mesh. Retyping these here would
    pass while the real ones were broken, so they are exec'd from the file instead --
    the same trick test_live.py uses.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "bridge", "seeds.py")
    tree = ast.parse(open(path).read())
    want = {"_too_many", "predicted_total"}
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    consts = [n for n in tree.body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("STROKE_BUDGET", "PREVIEW_BUDGET")]
    if len(keep) != len(want):
        raise SystemExit("bridge/seeds.py no longer defines %s"
                         % sorted(want - {n.name for n in keep}))
    ns = {"np": np, "S": S}
    exec(compile(ast.Module(body=consts + keep, type_ignores=[]), "<seeds>", "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in
                                    list(want) + ["STROKE_BUDGET", "PREVIEW_BUDGET"]})


SB = _load_pure()

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


class FakeObj:
    """Just enough object for predicted_total: face areas and a world matrix."""

    class _Polys:
        def __init__(self, a):
            self._a = np.asarray(a, np.float32)

        def __len__(self):
            return len(self._a)

        def foreach_get(self, attr, out):
            out[:] = self._a

    class _Data:
        def __init__(self, a):
            self.polygons = FakeObj._Polys(a)

    class _M:
        def to_3x3(self):
            return [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]

    def __init__(self, areas):
        self.data = FakeObj._Data(areas)
        self.matrix_world = FakeObj._M()
        self.type = 'MESH'


def main():
    print("\nTHE FLOOR MULTIPLIES")
    # the claim the whole guard rests on, against the real sampler
    for n_faces, min_s in ((1000, 8), (20000, 64), (137, 3)):
        areas = np.full(n_faces, 0.001)          # tiny faces: density wants ~0 strokes
        density, _ = S.solve_density(areas, 10, min_s, 128)
        counts = S.stroke_counts(areas, density, min_strokes=min_s, max_strokes=128)
        total = int(counts.sum())
        check("%s faces x min %d -> at least %s strokes"
              % ("{:,}".format(n_faces), min_s, "{:,}".format(n_faces * min_s)),
              total >= n_faces * min_s, "got %s" % "{:,}".format(total))

    check("Stroke Count cannot hold it down",
          int(S.stroke_counts(np.full(20000, 0.001),
                              S.solve_density(np.full(20000, 0.001), 10, 64, 128)[0],
                              min_strokes=64, max_strokes=128).sum()) >= 1_280_000,
          "the reported 1.28M case, with Stroke Count asking for 10")

    check("an object with no faces does not raise",
          SB.predicted_total(FakeObj(np.zeros(0)), 10, 1, 128) == (0, 0))

    print("\nTHE MESSAGE NAMES THE RIGHT DIAL")
    cases = [
        ("min", SB._too_many(1_280_000, 200_000, 20_000, 64, 128), "Min per Face"),
        ("max", SB._too_many(400_000, 200_000, 20_000, 1, 4096), "Max per Face"),
        ("count", SB._too_many(300_000, 200_000, 100, 1, 128), "Stroke Count"),
    ]
    for dial, msg, want in cases:
        check("%s -> %s" % (dial, want), want in msg, msg[:64])

    # the panel renders livepreview.last_error()[:70]; the cause has to survive that
    for dial, msg, want in cases:
        head = msg.split(". ")[0]
        check("%s: cause fits the panel's 70 chars" % dial, len(head) <= 70,
              "%d chars" % len(head))

    print("\nBUDGETS")
    check("the preview gives up sooner than the bake", SB.PREVIEW_BUDGET < SB.STROKE_BUDGET,
          "%s vs %s" % ("{:,}".format(SB.PREVIEW_BUDGET), "{:,}".format(SB.STROKE_BUDGET)))
    check("but not below Stroke Count's own soft max of 50,000",
          SB.PREVIEW_BUDGET >= 50_000, "raising the main dial still previews")

    panel_lock()

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


def panel_lock():
    """Min/Max per Face are LOCKED in the panel while the live preview is on.

    They are the dials that detonate it: both are per-FACE, so they multiply by the face
    count and ignore Stroke Count -- Min 64 on a 20,000-face mesh is 1,280,000 strokes,
    and rebuilding that re-runs an O(N^2) kNN on Blender's main thread.

    Clamping them was tried first and did not hold, so they are simply greyed out while
    previewing. That is a UI property with no numpy behind it, so the only thing testable
    outside Blender is that the row is still built disabled -- which is exactly the line a
    later refactor would drop without noticing.
    """
    print("\nTHE PANEL LOCKS THE TWO DIALS WHILE PREVIEWING")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "ui", "panel.py")).read()
    tree = ast.parse(src)

    # Match the exact node, not a substring of the dump: "live" also occurs inside
    # "livepreview", so a loose test passes with the lock deleted -- it did, once.
    def is_row_enabled(n):
        t = n.targets[0]
        return (isinstance(t, ast.Attribute) and t.attr == "enabled"
                and isinstance(t.value, ast.Name) and t.value.id == "row")

    def is_row_prop(n, name):
        v = n.value
        return (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                and v.func.attr == "prop" and isinstance(v.func.value, ast.Name)
                and v.func.value.id == "row"
                and any(isinstance(a, ast.Constant) and a.value == name for a in v.args))

    nodes = list(ast.walk(tree))
    assigns = [n for n in nodes if isinstance(n, ast.Assign) and n.targets]
    exprs = [n for n in nodes if isinstance(n, ast.Expr)]

    locks = [n for n in assigns if is_row_enabled(n)]
    check("the row carries an enabled flag", len(locks) == 1, "%d found" % len(locks))
    check("and it is driven by the preview being on",
          bool(locks) and isinstance(locks[0].value, ast.UnaryOp)
          and isinstance(locks[0].value.op, ast.Not),
          "row.enabled = not live")

    for prop in ("min_strokes", "max_strokes"):
        check("%s is drawn into that row" % prop,
              any(is_row_prop(n, prop) for n in exprs))

    check("and the lock is explained on screen", "locked while previewing" in src.lower())

    # the properties themselves must NOT carry a clamp callback any more
    props = open(os.path.join(root, "props.py")).read()
    check("props.py no longer clamps", "_clamp" not in props,
          "clamping was removed in favour of the lock")


if __name__ == "__main__":
    sys.exit(main())
