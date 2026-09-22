"""Brush masks, per-cell randomness, and the texel resolver.

Moved from bake_indirection.py; module-level config globals became explicit `cfg`
parameters and the PIL/OpenEXR I/O was lifted out (see bridge/). Pure numpy -- no bpy.
"""

import numpy as np

from .config import INV_SQRT2
from .geometry import frame_from_tangent, quat_to_TB

def mask_from_rgba(arr, cfg):
    """(H,W[,C]) float array in [0,1] -> a single-channel brush mask.

    Uses ALPHA whenever it carries shape (LA and RGBA alike), else luminance. Callers
    supply the pixels, so this stays free of PIL and bpy alike.
    """
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:                       # (H,W) single channel
        m = arr
    elif arr.shape[2] in (2, 4):            # LA or RGBA -- alpha is the last channel
        a = arr[..., -1]
        # Prefer alpha whenever it carries shape. Brush PNGs put the stroke in alpha,
        # and it is typically antialiased where luminance is a hard 0/1 matte. The old
        # heuristic compared RGB variance instead, which mis-picked on LA images and
        # silently averaged luminance WITH alpha.
        m = a if a.std() > 1e-3 else arr[..., :-1].mean(-1)
    else:                                   # RGB (or anything else): luminance
        m = arr[..., :3].mean(-1)
    if cfg.invert_mask:
        m = 1.0 - m
    return np.ascontiguousarray(m, np.float32)


def mask_long_axis_deg(mask, cfg):
    """Angle of the brush shape's long axis, in degrees from the mask's +x."""
    ys, xs = np.nonzero(mask > cfg.mask_thresh)
    if len(xs) < 8:
        return 0.0
    cov = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
    if not np.isfinite(cov).all():
        return 0.0
    ev, evec = np.linalg.eigh(cov)
    if ev[1] <= ev[0] * 1.02:                  # round enough that no axis is meaningful
        return 0.0
    v = evec[:, int(np.argmax(ev))]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def mask_bbox(mask, cfg):
    """(mu0, mu1, mv0, mv1): the only part of the stamp square that can pass the mask.

    Outside the painted pixels the mask is <= mask_thresh by construction, and a bilinear
    blend of sub-threshold values is itself sub-threshold, so a sample out there CANNOT
    pass. Clipping the footprint to this box is therefore exact, not an approximation.

    It is worth a lot: a brush fills its image lengthwise but only a fifth of it across,
    so the painted box is 8-18% of the square on the shipped set. Measured on a real
    1024 bake, footprint texels fell from 22.2M to ~3.1M -- and each one of those was a
    bilinear mask sample, the single most expensive thing resolve_uv does.

    Expanded by one texel each way because sample_mask reads floor(x) and floor(x)+1, so
    a sample just outside the painted box can still pick up a painted neighbour.
    """
    ys, xs = np.nonzero(mask > cfg.mask_thresh)
    if len(xs) == 0:
        return 0.0, 1.0, 0.0, 1.0
    H, W = mask.shape[:2]
    return (max(0.0, (xs.min() - 1) / (W - 1.0)),
            min(1.0, (xs.max() + 1) / (W - 1.0)),
            max(0.0, 1.0 - (ys.max() + 1) / (H - 1.0)),
            min(1.0, 1.0 - (ys.min() - 1) / (H - 1.0)))


def sample_mask(mask, u, v):
    """Bilinear sample mask (Hm,Wm) at u,v in [0,1] (v=0 bottom). Returns values."""
    Hm, Wm = mask.shape
    x = np.clip(u, 0, 1) * (Wm - 1)
    y = np.clip(1 - v, 0, 1) * (Hm - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, Wm - 1); y1 = np.minimum(y0 + 1, Hm - 1)
    fx = x - x0; fy = y - y0
    return (mask[y0, x0]*(1-fx)*(1-fy) + mask[y0, x1]*fx*(1-fy) +
            mask[y1, x0]*(1-fx)*fy     + mask[y1, x1]*fx*fy)


def hash01(ids):
    """Map integer ids to a well-distributed [0,1) per-id random (splitmix64)."""
    x = ids.astype(np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x = x ^ (x >> np.uint64(31))
    return x.astype(np.float64) / np.float64(2.0 ** 64)


# distinct salts -> independent draws from the same hash, per stroke id
SALT_BRUSH = np.uint64(0x27d4eb2f165667c5)
SALT_SIZE = np.uint64(0x9e3779b97f4a7c15)
"""Own salts so a stroke's brush and its size are independent of its tone (hash01(id))
and its rotation jitter (hash01(id ^ 0x9e3779b9)). Sharing a salt would tie shape or
scale to tone -- every rough stroke also dark -- which reads as a pattern, the very
thing these variations exist to break up."""


def brush_index(seed_id, n_brushes):
    """Which brush each stroke draws, in [0, n_brushes). Deterministic from the id, so a
    stroke keeps its shape across re-bakes and the coverage preview agrees with the bake."""
    if n_brushes <= 1:
        return np.zeros(len(seed_id), np.int64)
    i = np.asarray(seed_id).astype(np.int64) ^ np.int64(SALT_BRUSH)
    return np.minimum((hash01(i) * n_brushes).astype(np.int64), n_brushes - 1)


def morton_order(P):
    """Sort index that puts points near each other in 3D near each other in memory.

    Texels arrive in UV scanline order, which is scattered across the model in 3D, so a
    block of them covers everything and no stroke can ever be excluded from it. After
    this, a block is a compact patch of surface -- which is the whole basis of the
    per-chunk rejection below.

    10 bits per axis is plenty: it only has to group texels, not distinguish them, and a
    1024^3 lattice is far finer than any chunk. The bbox is taken from the data, so a
    non-finite coordinate would poison the quantisation -- callers filter those out into
    `valid`, but clip anyway rather than trust it.
    """
    P = np.asarray(P, np.float64)
    lo, hi = P.min(0), P.max(0)
    q = np.clip((P - lo) / np.maximum(hi - lo, 1e-12) * 1023.0, 0, 1023).astype(np.uint64)

    def part1by2(x):                      # 0b...cba -> 0b...c00b00a, room to interleave
        x = x & np.uint64(0x3ff)
        x = (x | (x << np.uint64(16))) & np.uint64(0x030000FF)
        x = (x | (x << np.uint64(8))) & np.uint64(0x0300F00F)
        x = (x | (x << np.uint64(4))) & np.uint64(0x030C30C3)
        x = (x | (x << np.uint64(2))) & np.uint64(0x09249249)
        return x

    key = (part1by2(q[:, 0]) | (part1by2(q[:, 1]) << np.uint64(1))
           | (part1by2(q[:, 2]) << np.uint64(2)))
    return np.argsort(key, kind="stable")


def auto_chunk(n_texels, n_seeds):
    """Texels per spatial tile.

    Two costs pull against each other: the rejection test is O(chunks x seeds), so it
    wants FEW big chunks, while a surviving stroke still processes its whole chunk, which
    wants chunks no bigger than a stamp. Measured optima: ~1024 at 800K texels for 2K-20K
    strokes, ~4096 at 13.4M texels / 20K strokes. This fits all of them.

    Fitted to measurements, not derived. The valley is shallow -- 13.2 s / 15.1 s / 17.6 s
    across a 16x range of chunk at 4K/20K -- so being roughly right is enough.
    """
    return int(np.clip(np.sqrt(float(n_texels) * max(n_seeds, 1)) / 128.0, 1024, 16384))


def resolve_uv_blocking(*args, **kw):
    """Drain resolve_uv and return its result, for callers that cannot yield.

    The terminal test harness and any non-modal use go through here; the addon drives
    the generator itself so Blender stays responsive.
    """
    gen = resolve_uv(*args, **kw)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def resolve_uv(points, uv_self, seeds, seed_id, seed_uv, seed_r,
               seed_nrm, seed_rot, mask, cfg, seed_tan=None,
               stamp_rot_deg=0.0, pt_nrm=None, counters=None):
    """GENERATOR. Yields progress in 0..1 after each texel chunk; returns the result
    tuple via StopIteration.value (use resolve_uv_blocking if you just want the value).

    Per texel: candidates within seed radius (sphere), SMALLEST stroke first and then by
    id DESC; the first whose brush mask covers the texel wins. Mask sampled in the seed's
    tangent frame (from rotation), sized by radius*MASK_SCALE. No stamp -> keep
    own UV. `pt_nrm` is the true surface normal at each texel and drives the crease
    guard; without it the guard cannot run. `mask` is either one (H,W) mask or a LIST of
    them -- with a list each stroke draws one by brush_index(), and each brush's own long
    axis is what mask_auto_align lines up. Returns (uv_out (M,2), dbg_out (M,3) encoded
    normal, covered (M,), lum_out (M,)).

    Pass a dict as `counters` to have the work actually done accumulated into it:
    `tiles`, `tile_size`, `pairs` (stroke-tile pairs surviving rejection) and `footprint`
    (texels reaching the mask sample). test_footprint asserts on these, which is why they
    read from THIS function rather than from a copy that could drift from it."""
    M = len(points)
    # Spatial order (see morton_order): every array indexed by texel lives in this order
    # until the end, where the four outputs are put back. Without it the rejection below
    # excludes nothing, because a scanline-ordered chunk touches the whole model.
    perm = morton_order(points)
    points = np.asarray(points)[perm]
    uv_self = np.asarray(uv_self)[perm]
    if pt_nrm is not None:
        pt_nrm = np.asarray(pt_nrm)[perm]
    uv_out  = uv_self.copy()
    dbg_out = np.zeros((M, 3), np.float32)
    lum_out = np.full(M, 0.5, np.float32)          # per-cell random -> B channel (0.5 = neutral)
    covered = np.zeros(M, bool)

    # One mask or a set: everything downstream works off the list, so the single-brush
    # path is just a set of one and there is no second code path to keep in step.
    masks = [mask] if isinstance(mask, np.ndarray) else list(mask)
    if not masks:
        raise ValueError("resolve_uv needs at least one brush mask")
    brush_of = brush_index(seed_id, len(masks))

    lum_all = hash01(seed_id).astype(np.float32)   # per-seed random luminance (unordered)

    # Frames for every seed, built ONCE and unordered. `seed_tan` is whatever
    # direction_source resolved to; None means use the quaternion as-is.
    if seed_tan is not None:
        T_all, B_all = frame_from_tangent(seed_nrm.astype(np.float64), seed_tan)
    else:
        T_all, B_all = quat_to_TB(seed_rot, cfg)
    # In-plane rotation of every stamp frame: the Global Rotation Angle plus, optionally, a
    # per-stroke random offset. The offset is hashed from the stroke's id (with its own
    # salt, so it is independent of the luminance random on the same id), which keeps it
    # deterministic -- a stroke lands the same way on every re-bake.
    jitter = float(np.clip(getattr(cfg, "rot_jitter_deg", 0.0), 0.0, 45.0))
    # Line the brush SHAPE's long axis up with the stroke direction. Every brush in a set
    # is drawn at its own angle, so this has to be measured per brush and applied per
    # stroke -- a single offset taken from one brush would rotate all the others wrongly.
    align = (np.radians([mask_long_axis_deg(m, cfg) for m in masks])[brush_of]
             if cfg.mask_auto_align else 0.0)
    if stamp_rot_deg or jitter or cfg.mask_auto_align:
        ang = np.radians(np.broadcast_to(np.asarray(stamp_rot_deg, np.float64),
                                         (len(seed_id),))).copy()
        ang -= align
        if jitter:
            ang += (hash01(seed_id ^ np.int64(0x9e3779b9)) * 2.0 - 1.0) * np.radians(jitter)
        _ca, _sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
        T_all, B_all = T_all * _ca + B_all * _sa, -T_all * _sa + B_all * _ca
    # Per-stroke size variation, exponential so the spread is symmetric about the dial:
    # at size_random 1.0 the factor runs 0.5x..2x and halving is as likely as doubling,
    # where a linear 1 +/- r would put the ends at 0.5x and 1.5x and read as "mostly
    # bigger". This is the single place the effective radius is built, so the broad phase
    # (r2_ord) and the footprint (h_ord) both follow from it -- and so does the order.
    size_var = float(np.clip(getattr(cfg, "size_random", 0.0), 0.0, 1.0))
    r_eff = seed_r.astype(np.float64) * cfg.mask_scale
    if size_var:
        r_eff = r_eff * 2.0 ** ((hash01(seed_id ^ np.int64(SALT_SIZE)) * 2.0 - 1.0) * size_var)

    # Candidate order: the loop below awards each texel to the FIRST candidate that covers
    # it, so sorting smallest-first is exactly "a fine mark is never swallowed by a coarse
    # one". Sorted on the FINAL radius -- Stroke Size and size variation included -- because
    # the rule is about the mark that actually lands, not the patch of surface it came from.
    #
    # This replaced a two-tier split at the median radius. Tiers only guaranteed that SOME
    # stroke from the smaller half beat SOME stroke from the larger; within a tier the
    # winner fell to id, which is arbitrary. A straight sort says the same thing
    # monotonically and removes the priority attribute from the pipeline entirely.
    #
    # int64 cast first: seed ids span the full int32 range, and negating INT32_MIN
    # overflows back to itself, which would silently misorder those seeds.
    order = np.lexsort((-seed_id.astype(np.int64), r_eff))   # last key is primary
    s_ord   = seeds[order].astype(np.float64)
    suv_ord = seed_uv[order]
    snr_ord = seed_nrm[order]
    # The crease guard compares DIRECTIONS, so both sides have to be unit length. No-op
    # when they already are.
    guard = bool(cfg.crease_guard) and pt_nrm is not None
    if guard:
        nl = np.linalg.norm(snr_ord, axis=1, keepdims=True)
        snr_ord = snr_ord / np.maximum(nl, 1e-12)
        cos_max = float(np.cos(np.radians(cfg.crease_angle_deg)))
        # kept float32 and normalised per chunk: as float64 up front this is a full extra
        # copy of the normal map, 0.32 GB at 4K, for no gain
        pn_all = np.asarray(pt_nrm, np.float32)
    lum_ord = lum_all[order]                        # per-seed random luminance
    r_ord   = r_eff[order]
    r2_ord  = r_ord * r_ord                         # sphere broad phase scales with it
    h_ord   = r_ord * INV_SQRT2                      # footprint square, inscribed in sphere
    T_ord, B_ord = T_all[order], B_all[order]       # already rotated by STAMP_ROTATE_DEG
    brush_ord = brush_of[order]                     # which mask each candidate stamps with
    # Footprint bounds per candidate, in stroke-local units. The stamp square is what the
    # mask image maps onto, but only the brush's painted BOX inside it can ever pass --
    # see mask_bbox. Precomputed as four flat arrays so the inner loop indexes scalars
    # rather than unpacking a tuple 300,000 times.
    _bb = np.array([mask_bbox(m, cfg) for m in masks], np.float64)[brush_ord]
    _two_h = 2.0 * h_ord
    lu_lo, lu_hi = (_bb[:, 0] - 0.5) * _two_h, (_bb[:, 1] - 0.5) * _two_h
    lv_lo, lv_hi = (_bb[:, 2] - 0.5) * _two_h, (_bb[:, 3] - 0.5) * _two_h

    # World-space box around everything a stroke can paint, for the per-tile rejection.
    #
    # The sphere this replaces was loose twice over: it circumscribes the stamp SQUARE,
    # and the square is itself only 8-18% painted. Measured, the box halves the number of
    # (stroke, tile) pairs -- 212,657 -> 114,533 on a 1.04M-texel bake -- and each pair
    # costs three full-tile passes, so that is most of a 1.5x.
    #
    # The half-extent along the NORMAL is r_ord, not zero: the footprint test constrains
    # lu and lv only, and nothing but the sphere test below bounds how far off the tangent
    # plane a texel may sit. Bounding just the painted rectangle looks tighter and would
    # silently drop strokes.
    #
    # abs() because the box is oriented along (T, B, N) while a tile's box is axis-aligned:
    # projecting an oriented box onto an axis sums the absolute contributions. A stroke
    # lying diagonally to the object axes gets a looser box -- never looser than the
    # sphere, which is that loose in every direction at once.
    _ctr = (s_ord + T_ord * (0.5 * (lu_lo + lu_hi))[:, None]
            + B_ord * (0.5 * (lv_lo + lv_hi))[:, None])
    # snr_ord is only unit-length when the crease guard is on, and a SHORT normal here
    # would under-size the box and drop strokes. Normalise for this regardless.
    _n_unit = snr_ord / np.maximum(np.linalg.norm(snr_ord, axis=1, keepdims=True), 1e-12)
    _ext = (np.abs(T_ord) * (0.5 * (lu_hi - lu_lo))[:, None]
            + np.abs(B_ord) * (0.5 * (lv_hi - lv_lo))[:, None]
            + np.abs(_n_unit) * r_ord[:, None])
    box_lo, box_hi = _ctr - _ext, _ctr + _ext

    chunk = int(cfg.chunk) if getattr(cfg, "chunk", 0) else auto_chunk(M, len(s_ord))
    if counters is not None:
        counters["tile_size"] = chunk
    for a in range(0, M, chunk):
        b = min(a + chunk, M)
        p = points[a:b].astype(np.float64)
        done = np.zeros(len(p), bool)
        if guard:
            pn_chunk = pn_all[a:b].astype(np.float64)
            pn_chunk = pn_chunk / np.maximum(
                np.linalg.norm(pn_chunk, axis=1, keepdims=True), 1e-12)
        # Which strokes can reach this chunk at all? Point-to-box distance from each
        # stroke centre to the chunk's bounding box, against that stroke's own radius --
        # a sphere/AABB test, vectorised over every stroke at once.
        #
        # This is an EXCLUSION, not an approximation: a sphere that misses the box
        # contains no texel of the chunk, so the sphere test below would have rejected
        # every one of them anyway. It may over-include freely; under-including would
        # silently drop strokes at chunk boundaries and show up as faint seams on a grid.
        #
        # flatnonzero returns ASCENDING indices into arrays that are already in candidate
        # order, so survivors keep that order -- a subsequence of a sorted list is still
        # sorted. No per-chunk sort exists, and none is needed.
        _lo, _hi = p.min(0), p.max(0)
        _near = np.flatnonzero((box_lo <= _hi).all(1) & (_lo <= box_hi).all(1))
        if counters is not None:
            counters["pairs"] = counters.get("pairs", 0) + len(_near)
            counters["tiles"] = counters.get("tiles", 0) + 1
        for j in _near:
            if done.all():
                break
            dx = p[:, 0] - s_ord[j, 0]
            dy = p[:, 1] - s_ord[j, 1]
            dz = p[:, 2] - s_ord[j, 2]
            within = (dx*dx + dy*dy + dz*dz) < r2_ord[j]          # sphere broad phase
            # project offset into the seed's tangent frame
            lu = dx*T_ord[j, 0] + dy*T_ord[j, 1] + dz*T_ord[j, 2]
            lv = dx*B_ord[j, 0] + dy*B_ord[j, 1] + dz*B_ord[j, 2]
            h = h_ord[j]
            infoot = (within & (lu > lu_lo[j]) & (lu < lu_hi[j])
                      & (lv > lv_lo[j]) & (lv < lv_hi[j]) & ~done)
            if not infoot.any():
                continue
            # sample the mask only for footprint texels, then threshold
            mu = lu[infoot] / (2 * h) + 0.5
            mv = lv[infoot] / (2 * h) + 0.5
            passed = sample_mask(masks[brush_ord[j]], mu, mv) > cfg.mask_thresh
            hit_local = np.where(infoot)[0][passed]
            if counters is not None:
                counters["footprint"] = counters.get("footprint", 0) + int(infoot.sum())
            if guard and len(hit_local):
                # How far has the surface TURNED between the texel and the stroke's own
                # normal? Reject beyond crease_angle_deg: past that the stamp is reaching
                # around a fold and would paint one flat normal on both sides of it.
                #
                # A rejected texel is simply offered to the next candidate, and if every
                # candidate rejects it, it stays uncovered -- which is not a hole: the
                # uncovered path writes the TRUE surface normal, so the texel shades like
                # the untouched model. Keeping the least-wrong stroke there instead (which
                # this did briefly) painted normals up to 179.6 deg from the surface, and
                # those face into the model and render black.
                #
                # This used to test distance off the stroke's tangent plane instead, with
                # a threshold of k*radius. That is why strokes looked cut: the cut sits
                # wherever the accumulated offset reaches k*r, which scales with the
                # stroke, so every stroke crossing one fold stopped somewhere different
                # (measured spread across strokes: 84.9 deg, vs 33.7 for the angle). An
                # angle is a property of the SURFACE, so every stroke stops on the same
                # locus and the seam reads as the fold rather than as damage.
                pnh = pn_chunk[hit_local]
                cs = (pnh[:, 0] * snr_ord[j, 0] + pnh[:, 1] * snr_ord[j, 1]
                      + pnh[:, 2] * snr_ord[j, 2])
                hit_local = hit_local[cs > cos_max]
            if len(hit_local):
                uv_out[a:b][hit_local]  = suv_ord[j]
                dbg_out[a:b][hit_local] = snr_ord[j] * 0.5 + 0.5
                lum_out[a:b][hit_local] = lum_ord[j]
                done[hit_local] = True
        covered[a:b] = done
        yield b / float(M)
    inv = np.empty(M, np.int64)
    inv[perm] = np.arange(M)
    return uv_out[inv], dbg_out[inv], covered[inv], lum_out[inv]


def dilate_fill(img, valid, iters):
    """Grow valid pixels into invalid ones by copying from a written 4-neighbor."""
    img = img.copy(); valid = valid.copy()
    for _ in range(iters):
        if valid.all():
            break
        filled = valid.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = np.roll(np.roll(valid, dy, 0), dx, 1)
            take = src & ~valid                     # invalid px with a valid neighbor
            shifted = np.roll(np.roll(img, dy, 0), dx, 1)
            img[take] = shifted[take]
            filled |= take
        valid = filled
    return img
