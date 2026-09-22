"""The seed/direction cache, which exists so a preview can be fast.

Neither seeds nor the direction field depend on resolution, brush, Stroke Size, Size
Variation, Global Rotation, Rotation Jitter or Stroke Cutoff -- every look dial except
Stroke Count and the per-face clamps. The direction field is ~N^1.7 (0.17s at 4,000
strokes, 2.88s at 20,000), so at high stroke counts reusing it IS the preview budget.

The danger is the opposite of slowness: a cache that hits when it should miss places
strokes for a mesh or a stroke count that is no longer current, and the result still looks
plausible. So most of this file is about when it must MISS.

Run directly: python3 autostroke/tests/test_cache.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types                                                  # noqa: E402
sys.modules.setdefault("bpy", types.SimpleNamespace(
    types=types.SimpleNamespace(Operator=object), data=None,
    path=types.SimpleNamespace(abspath=lambda p: p)))

from bridge import cache                                      # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def main():
    print("\nSEED / DIRECTION CACHE\n")
    cache.clear()
    SIG = "abc123"
    k = cache.key(SIG, 4000, 1, 128, 0.5)

    check("a cold cache misses", cache.get(k) is None)
    cache.put(k, {"position": [1, 2]}, {"strokes": 2}, "tangent", "curv")
    got = cache.get(k)
    check("and hits once filled", got is not None and got[0]["position"] == [1, 2])
    check("the same key hits again", cache.get(k) is not None)

    # Everything that changes the seeds must miss.
    print()
    for label, other in (
            ("the mesh changed", cache.key("different", 4000, 1, 128, 0.5)),
            ("Stroke Count changed", cache.key(SIG, 5000, 1, 128, 0.5)),
            ("Min per Face changed", cache.key(SIG, 4000, 2, 128, 0.5)),
            ("Max per Face changed", cache.key(SIG, 4000, 1, 64, 0.5)),
            ("aspect alpha changed", cache.key(SIG, 4000, 1, 128, 1.0))):
        check("misses when %s" % label, cache.get(other) is None)

    # And everything that does NOT must hit -- that is the whole point. Resolution is the
    # important one: a preview at 256 and a bake at 1024 have to share.
    print()
    check("resolution is NOT part of the key",
          "res" not in repr(k) and cache.get(cache.key(SIG, 4000, 1, 128, 0.5)) is not None,
          "so a 256 preview and a 1024 bake share seeds")
    check("Stroke Size, brush, rotation and cutoff are not in the key",
          len(k) == 5, "key is (mesh, count, min, max, alpha) and nothing else")

    # One slot, not an unbounded dict: seed arrays are large and an artist works on one
    # object at a time.
    print()
    cache.put(cache.key("other", 1, 1, 1, 0.5), {}, {}, None, None)
    check("a new entry evicts the old one", cache.get(k) is None,
          "one slot, so this cannot grow without bound")
    cache.clear()
    check("clear() empties it", cache.get(cache.key("other", 1, 1, 1, 0.5)) is None)

    # A float that differs only in type must not split the key.
    check("int and float clamps do not make different keys",
          cache.key(SIG, 4000, 1, 128, 0.5) == cache.key(SIG, 4000.0, 1.0, 128.0, 0.5))

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
