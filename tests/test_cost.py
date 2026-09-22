"""The bake-time estimate, and the fact that it calibrates itself.

The old model had no Stroke Size term at all, which was most of why it read low: bigger
stamps overlap more, so more strokes survive each tile's rejection. It was also a fixed
constant measured on one machine, and a real bake came in at 3.31s where the fixture
predicted 2.5s. No constant fixes that, so the operator feeds the real time back.

Run directly: python3 autostroke/tests/test_cost.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import cost                                        # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


# (res, strokes, stroke_size) -> measured resolve seconds, from the 34-point grid
MEASURED = [
    (1024, 2000, 1.0, 0.46), (1024, 2000, 4.0, 0.76), (1024, 2000, 10.0, 0.92),
    (1024, 5000, 4.0, 1.25), (1024, 5000, 10.0, 1.56),
    (1024, 20000, 1.0, 2.20), (1024, 20000, 4.0, 3.21), (1024, 20000, 10.0, 4.01),
    (2048, 2000, 1.0, 1.81), (2048, 2000, 10.0, 2.84),
    (2048, 5000, 4.0, 3.13), (2048, 20000, 10.0, 7.73),
]


def main():
    print("\nBAKE TIME ESTIMATE\n")

    # It must never read SHORT on the data it was fitted to -- long is harmless, short
    # erodes trust in the readout.
    short = []
    for res, n, s, t in MEASURED:
        if cost.bake_seconds(res, n, s) < t:
            short.append((res, n, s))
    check("never under-predicts the measured grid", not short, str(short[:3]))

    # Every input has to move it, in the right direction.
    base = cost.bake_seconds(1024, 5000, 4.0)
    check("more texels costs more", cost.bake_seconds(2048, 5000, 4.0) > base * 1.5)
    check("more strokes costs more", cost.bake_seconds(1024, 20000, 4.0) > base * 1.5)
    check("bigger Stroke Size costs more -- the term that was missing",
          cost.bake_seconds(1024, 5000, 10.0) > base,
          "size 4 %.2fs -> size 10 %.2fs" % (base, cost.bake_seconds(1024, 5000, 10.0)))
    check("and it is a modest effect, not a dominant one",
          cost.bake_seconds(1024, 5000, 10.0) < base * 1.5,
          "%.2fx over the whole 1-10 range"
          % (cost.bake_seconds(1024, 5000, 10.0) / cost.bake_seconds(1024, 5000, 1.0)))

    # The direction field overtakes the resolve at high stroke counts; if that term were
    # dropped the estimate would go badly low exactly where bakes get slow.
    hi = cost.bake_seconds(1024, 40000, 4.0)
    check("high stroke counts are dominated by the direction field",
          cost.DIR_K * 40000 ** cost.DIR_N > 0.5 * hi / cost.DEFAULT_SCALE,
          "%.1fs of %.1fs" % (cost.DIR_K * 40000 ** cost.DIR_N, hi / cost.DEFAULT_SCALE))

    check("degenerate inputs do not explode",
          all(0 < cost.bake_seconds(*a) < 1e6
              for a in ((1, 0, 0.0), (8192, 1, 0.1), (1024, 10 ** 6, 10.0))))

    # ---- self-calibration -------------------------------------------------
    print()
    raw = cost.bake_seconds(1024, 4000, 10.0, scale=1.0)
    sc = cost.DEFAULT_SCALE
    for _ in range(6):
        sc = cost.calibration(3.31, raw, previous=sc)
    check("converges on a machine the fit did not see",
          abs(cost.bake_seconds(1024, 4000, 10.0, sc) - 3.31) < 0.15,
          "predicts %.2fs after 6 bakes, actual 3.31s"
          % cost.bake_seconds(1024, 4000, 10.0, sc))
    check("one freak bake cannot swing it",
          cost.calibration(300.0, raw, previous=1.25) <= 1.25 + 5.0,
          "a 300s outlier moves scale to %.2f, not 100+"
          % cost.calibration(300.0, raw, previous=1.25))
    check("a zero or negative measurement is ignored",
          cost.calibration(0.0, raw, previous=1.4) == 1.4
          and cost.calibration(3.0, 0.0, previous=1.4) == 1.4)
    check("it settles rather than oscillating",
          abs(cost.calibration(3.31, raw, previous=sc) - sc) < 0.02,
          "already at the fixed point")

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
