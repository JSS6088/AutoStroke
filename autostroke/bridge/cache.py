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
    _reach.clear()
    _tri_map.clear()


# ---- reachable surface ------------------------------------------------------
# Which triangles each stroke may paint (core/reach.py). It depends on the seeds and on the
# radius each stroke actually paints with -- so on Stroke Size and Size Variation, but on
# no other look dial: rotation, jitter, cutoff and the brush set never rebuild it. One slot,
# for the same reason as the seeds.

_reach = {}


def reach_key(seed_key, cfg):
    """Everything the reach tables depend on beyond the seeds themselves."""
    return (seed_key, float(cfg.mask_scale), float(getattr(cfg, "size_random", 0.0)))


def reach_for(seed_key, cfg, build):
    """The cached reach tables for these seeds and radius settings, or `build()`'s result
    (core/reach.build_for_seeds). Shared by the bake and the preview, so both search
    identical candidate lists."""
    k = reach_key(seed_key, cfg)
    if _reach.get("key") != k:
        _reach.clear()
        _reach.update(key=k, value=build())
    return _reach["value"]


# ---- which triangle each texel belongs to ---------------------------------------
# A UV rasterisation of the mesh at the bake resolution (core/reach.raster_tri_ids). It
# depends on the mesh and the resolution only -- the same key as the position map -- so a
# look-only re-bake never redoes it.

_tri_map = {}


def tri_map_for(k, build):
    """The cached (H*W,) texel -> triangle map for key `k`, or `build()`'s result."""
    if _tri_map.get("key") != k:
        _tri_map.clear()
        _tri_map.update(key=k, value=build())
    return _tri_map["value"]
