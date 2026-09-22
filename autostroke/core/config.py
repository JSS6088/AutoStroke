"""Every tunable knob, in one place.

These were module-level globals in bake_indirection.py. Gathered into a dataclass so
the addon can build one per bake without mutating module state, and so the pure-numpy
core has no hidden dependencies on import-time configuration.
"""

from dataclasses import dataclass

INV_SQRT2 = 1.0 / (2.0 ** 0.5)
"""Stamp footprint: a square in the seed's tangent plane, half-side = radius/sqrt2
(diagonal = 2*radius, inscribed in the sphere used for broad phase). The mask image
maps onto that square; the mask's shape carves the actual stroke."""


@dataclass
class Config:
    # ---- texel / image ----------------------------------------------------
    bg_eps: float = 1e-6          # |pos| below this on all axes = background
    dilate_px: int = 16           # grow valid data into the gutter (seam hygiene)
    chunk: int = 0
    """Texels per spatial tile, 0 = derive it from the workload (core.baker.auto_chunk).
    This used to be a memory cap on the per-seed temporaries; it is not one any more,
    because a tile is now ~1-16k texels rather than 200,000. Override only to A/B the
    tile size -- output must not depend on it."""

    # ---- brush ------------------------------------------------------------
    mask_thresh: float = 0.5      # mask value above this = inside the stroke
    invert_mask: bool = False     # flip if the mask has black = inside
    mask_scale: float = 4.0       # multiplier on effective radius (sphere + footprint)
    """Why 4 and not 1: the brush image maps onto a square of side sqrt(2)*r_eff, but the
    brush only PAINTS part of that square. Across the shipped `standard` set the painted
    box is 0.69-0.94 of the image on its long axis and only 0.115-0.224 on its short one,
    so at 1.0 a mark is ~1.2r long and ~0.3r wide against a patch about r across -- the
    length is fine, the width leaves gaps. Marks meet across their width at 3.2x / 4.5x /
    3.7x / 6.1x on the four standard brushes (median 4.1)."""
    size_random: float = 0.0
    """Per-stroke size variation, 0 = none, 1 = each stroke lands between half and double
    mask_scale. Exponential, so halving is exactly as likely as doubling."""
    stamp_rotate_deg: float = 0.0 # "Global Rotation Angle": 90 runs strokes ACROSS the form
    rot_jitter_deg: float = 0.0
    """Per-stroke random rotation, +/- this many degrees, deterministic from the stroke's
    id. Breaks up the mechanical look of one silhouette repeated thousands of times.
    Capped at 45 (a 90 degree spread): past that strokes stop agreeing with the flow
    field they were just aligned to."""
    mask_auto_align: bool = True
    """Line up the brush SHAPE's long axis with the stroke direction. The mask's x axis
    maps to the frame's T and its y to B, so a vertically-drawn mask puts stroke LENGTH
    on B -- across the form, the opposite of what the direction field chose. Measuring
    the mask and folding the offset into the rotation makes any orientation work."""

    # ---- placement --------------------------------------------------------
    target_strokes: int = 5000
    """How many strokes the artist wants on this object. The per-m^2 density the sampler
    needs is SOLVED from this at bake time (core.sampling.solve_density), so there is only
    one dial and it cannot go stale when the model is scaled or re-topologised."""
    min_strokes: int = 1          # per face; 1 so no face is ever left empty
    max_strokes: int = 16         # per face; caps cost on big flat faces
    aspect_alpha: float = 0.5
    """r = sqrt(leaf area) * (leaf aspect / 2)^alpha. 1/2 is derived, not tuned: a sliver
    needs r ~ L/2 while sqrt(A) = sqrt(Lw/2), and their ratio is sqrt(aspect/2). Measured
    on a 1m x 1cm bevel at 8 strokes: 64% covered without it, 100% with."""

    # ---- crease guard -----------------------------------------------------
    crease_guard: bool = True     # dev/A-B only: no UI, and no reason to turn it off
    crease_angle_deg: float = 45.0
    """Reject a texel from a stamp once the surface has TURNED more than this far between
    the texel's own normal and the stroke's -- i.e. the stamp is reaching around a fold
    and would paint one flat normal on both sides of it. The texel goes to the next
    candidate; if every candidate rejects it, it stays uncovered and keeps the ORIGINAL
    surface normal.

    Uncovered is not a hole: ops/bake.py writes the true surface normal there, so the
    texel shades like the untouched model. An earlier version kept the least-wrong stroke
    instead, on the theory that a gap mid-stroke was worse -- it was not. That painted
    normals up to 179.6 deg from their own surface, which face into the model and render
    black; measured on the Sphinx, 1,267 such blobs, median 2 texels each, and every
    single one had been painted by a stroke rather than left bare.

    This is an angle rather than a distance off the stroke's tangent plane (the old
    k*radius test) because the cut has to land in the same place for every stroke that
    crosses one fold. A distance threshold scales with the stroke, so it did not:
    measured across the strokes crossing a fold on Body, the deviation at which each one
    stopped spread 84.9 deg. That is what read as strokes being chopped mid-shape. The
    angle brings it to 33.7 deg, and drops p90 per-texel normal error from 31.6 to
    23.8 deg."""

    # ---- direction --------------------------------------------------------
    direction_source: str = "curvature"   # "curvature" | "quaternion" (dev A/B only)
    quat_wxyz: bool = True        # Blender quaternions are (w,x,y,z)
    curv_k: int = 14              # neighbouring SEEDS in the shape-operator fit
    curv_smooth_iters: int = 20   # line-field smoothing passes (0 = raw, noisy)
    curv_smooth_mix: float = 0.5  # per-pass blend toward the neighbour average
    curv_normal_min: float = 0.5
    """Drop kNN neighbours whose normal disagrees more than this. Nearest-by-Euclidean
    reaches THROUGH thin features, so without it a seed can be "near" a point on the
    opposite surface and that bogus normal delta corrupts its fit."""
    curv_min_nb: int = 6          # below this many survivors, keep them all instead
    curv_max_bytes: int = 512 << 20

