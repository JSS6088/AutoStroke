"""Mesh + settings -> the seed arrays the baker consumes.

This is the seam the Geometry Nodes path used to occupy. Everything it produces matches
the contract resolve_uv already expects: position, UVMap, normal, radius, id.
"""

import numpy as np

from ..core import sampling as S
from . import mesh as M

STROKE_BUDGET = 200_000
"""Refuse to place more strokes than this.

Not a memory limit -- a TIME one. geometry.knn_indices is O(N^2): every seed is compared
against every other to find its 14 nearest. Measured 0.74 s at 10k seeds, 6.9 s at 30k,
26.2 s at 60k, which extrapolates to ~5 min at this budget and ~3.3 HOURS at 1.28M.

1.28M is not hypothetical. Min Strokes per Face is a floor applied to EVERY face, so it
multiplies by the face count and ignores Stroke Count entirely: 64 on a 20,000-face mesh
is 1,280,000 strokes no matter what the main dial says. The work then runs inside a timer
callback on Blender's main thread, with no progress and no Esc -- the UI simply stops
responding, and the only way out is to kill Blender.

So the count is checked BEFORE anything expensive allocates. stroke_counts is the last
cheap step, and it already knows the exact total.

Stroke Count's own soft maximum is 50,000, so this never stands in the way of the dial an
artist actually reaches for -- only of the two clamps, which multiply."""

PREVIEW_BUDGET = 60_000
"""The live preview rebuilds on every slider tick, so it gives up sooner.

Just above target_strokes' soft maximum, so raising Stroke Count to its limit still
previews; it is the multiplying clamps this stops."""


def _too_many(total, budget, n_faces, min_strokes, max_strokes):
    """Name the dial that caused it. The panel truncates to ~70 characters, so the
    cause has to come first and the advice second."""
    fmt = "{:,}".format
    if n_faces * min_strokes > budget:
        cause = "Min per Face %d x %s faces" % (min_strokes, fmt(n_faces))
        fix = "Lower Min Strokes per Face -- it applies to every face, so it multiplies."
    elif n_faces * max_strokes <= budget:
        cause = "Stroke Count %s" % fmt(total)
        fix = "Lower Stroke Count."
    else:
        cause = "Max per Face %d x %s faces" % (max_strokes, fmt(n_faces))
        fix = "Lower Max Strokes per Face, or Stroke Count."
    return ("%s strokes, limit %s -- %s. %s"
            % (fmt(total), fmt(budget), cause, fix))


def predicted_total(obj, target_strokes, min_strokes, max_strokes):
    """Strokes these settings will produce -- cheap enough for a slider's callback.

    Reads obj.data.polygons areas straight through foreach_get: no depsgraph, no
    to_mesh(). That makes it blind to modifiers which change the face count (Subdivision
    above all), so it can UNDER-count. build_seeds re-checks against the evaluated mesh
    for exactly that reason: this one keeps the slider out of trouble, that one is the
    guarantee.

    Counted the way the sampler counts -- same solve_density, same stroke_counts -- so
    the number here is the number the bake gets. Returns (total, n_faces).
    """
    me = getattr(obj, "data", None)
    if me is None or not hasattr(me, "polygons") or len(me.polygons) == 0:
        return 0, 0
    areas = np.empty(len(me.polygons), np.float32)
    me.polygons.foreach_get("area", areas)
    # density is per m^2, so object scale has to count -- as it does in read_triangles
    scale = np.array(obj.matrix_world.to_3x3()).__abs__().sum(0).prod() ** (2.0 / 3.0)
    areas = areas.astype(np.float64) * scale
    density, _reachable = S.solve_density(areas, target_strokes, min_strokes, max_strokes)
    counts = S.stroke_counts(areas, density, min_strokes=min_strokes,
                             max_strokes=max_strokes)
    return int(counts.sum()), len(areas)


def build_seeds(obj, target_strokes, min_strokes=1, max_strokes=128, alpha=0.5,
                budget=STROKE_BUDGET):
    """Place every stroke on `obj` and size it. Returns (seeds dict, stats dict).

    Takes the stroke COUNT the artist asked for and solves the per-m^2 density from it,
    so the two can never disagree. The clamps bend that relationship, which is why it is
    solved rather than divided -- see core.sampling.solve_density.

    Raises MeshError if the settings ask for more than `budget` strokes (see
    STROKE_BUDGET). Pass budget=0 to place them anyway.
    """
    m = M.read_triangles(obj)

    density, reachable = S.solve_density(m["face_area_world"], target_strokes,
                                         min_strokes, max_strokes)
    counts = S.stroke_counts(m["face_area_world"], density,
                             min_strokes=min_strokes, max_strokes=max_strokes)
    # Before sample_faces, and well before the direction field: this is the last point
    # where nothing large has been allocated and nothing O(N^2) has started.
    total = int(counts.sum())
    if budget and total > budget:
        raise M.MeshError(_too_many(total, budget, m["n_faces"], min_strokes, max_strokes))
    pos, leaf_area, leaf_aspect, face_id, within, tri_of_stroke = S.sample_faces(
        m["P"], m["Q"], m["R"], m["face_of_tri"], counts)
    if len(pos) == 0:
        raise M.MeshError("No strokes were placed -- raise Stroke Density")

    # tri_of_stroke comes straight from the sampler, so each stroke's UV and normal are
    # its own triangle's -- no nearest-centroid guessing near a quad's diagonal.
    t = tri_of_stroke
    uv = M.barycentric_uv(pos, m["P"][t], m["Q"][t], m["R"][t],
                          m["uvP"][t], m["uvQ"][t], m["uvR"][t])
    # The SHADING normal at the stroke, not its triangle's facet normal. The facet normal
    # paints a flat patch into a curved surface, which on a smooth-shaded mesh reads as
    # polygonal blotches. It also put the two ends of the crease guard in different
    # spaces: the texel side has always been Cycles' shading normal.
    nrm = M.barycentric_normal(pos, m["P"][t], m["Q"][t], m["R"][t],
                               m["nP"][t], m["nQ"][t], m["nR"][t])

    radius = S.stroke_radius(leaf_area, leaf_aspect, alpha=alpha)

    seeds = dict(
        position=pos.astype(np.float32),
        UVMap=uv,
        normal=nrm,
        radius=radius.astype(np.float32),
        id=np.arange(len(pos), dtype=np.int64),
    )
    stats = dict(
        strokes=len(pos), faces=m["n_faces"],
        density=density, target=int(target_strokes), reachable=int(reachable),
        area=float(m["face_area_world"].sum()),
        at_min=float((counts == min_strokes).mean()),
        at_max=float((counts == max_strokes).mean()),
        radius_med=float(np.median(radius)),
        aspect_boost=float(np.median(radius / np.maximum(np.sqrt(leaf_area), 1e-12))),
    )
    seeds["_face_id"] = face_id
    seeds["_within"] = within
    return seeds, stats
