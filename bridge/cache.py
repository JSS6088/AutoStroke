"""Seeds and the direction field, kept between bakes.

Neither depends on resolution, brush, Stroke Size, Size Variation, Global Rotation,
Rotation Jitter or Stroke Cutoff -- every look dial except Stroke Count and the per-face
clamps. So while an artist is iterating on the look, both are being recomputed identically
every time.

That matters most exactly where it hurts: the direction field is ~N^1.7 in strokes, 0.17 s
at 4,000 and 2.88 s at 20,000. At high stroke counts it is the preview's whole budget.

In memory only, and deliberately: a stale seed set on disk would be far worse than
recomputing, and the cost of being wrong here is silently placing strokes for a mesh that
has changed. The key is the same evaluated-mesh hash the position map uses, so it sees a
moved vertex, a re-unwrap or a modifier change.
"""

_entry = {}
"""One slot, not a dict of many: an artist works on one object at a time, and an unbounded
cache of seed arrays is a memory leak with extra steps."""


def key(sig, target, min_strokes, max_strokes, alpha):
    """Everything build_seeds and the direction field actually depend on."""
    return (sig, int(target), int(min_strokes), int(max_strokes), float(alpha))


def get(k):
    if _entry.get("key") == k:
        return _entry["seeds"], _entry["stats"], _entry["tangent"], _entry["curv"]
    return None


def put(k, seeds, stats, tangent, curv):
    _entry.clear()
    _entry.update(key=k, seeds=seeds, stats=stats, tangent=tangent, curv=curv)


def clear():
    _entry.clear()
