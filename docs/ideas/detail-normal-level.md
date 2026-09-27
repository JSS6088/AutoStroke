# Idea (shelved): a detail level of strokes from one detail normal map

**Status: designed, not built.** Shelved in September 2026 in favour of presentation work.
**Before building, spike it first** (about an hour, throwaway):
- Run the splitting rule below on a flat plane with a real detail normal map.
- Render form-only against form + detail at normal viewing distance.
- Build this only if the difference is obvious.

## Why this and not texture-driven placement

The first idea was to make *any* texture map painterly by placing extra strokes wherever
that map's colour varies. It was dropped for two reasons:

- **One object has one stroke set.** There is one indirection map, and every texture read
  through it sees the same strokes. Albedo, roughness and detail maps would each want
  strokes in different places, and there is no way to give each map its own set.
- **Artists use several maps.** Refining per map makes the stroke count explode.

Making existing maps painterly *without* new placement still works and is cheap: route the
indirection pointer into any image node, and it paints one flat colour per stroke.

**What survives, reframed.** The tool stays about **normals**, with two levels:
- **Form level (today):** strokes placed per face by area × density, painting mesh normals.
- **Detail level:** extra, smaller strokes placed where **one detail normal map** (pores,
  weave, carving, a tiling fabric normal) varies, painting that relief as flat per-stroke
  normals.

One map keeps the stroke count bounded, and "form + detail" is intuitive for artists. With
no detail map set, everything stays byte-identical.

**Decisions made while designing:**
- Splitting is a plain top-down threshold on normal variance. The control is the
  tolerance, as a Detail Sensitivity slider; the resulting count is shown and capped by
  `STROKE_BUDGET`.
- Detail strokes follow the form direction field.
- Variance over each region's UV bounding box comes from summed-area tables. Mip pyramids
  were rejected: their square, power-of-two, grid-snapped blocks measured about 10× the
  region's texels, against 2× for the bounding box.
- The detail map is assumed to tile: every texel counts, so there is no usage mask. Baked
  (island-layout) detail maps are out of scope, because their padding would read as
  variance along seams.

## Design

### 1. What a detail stroke is: it lies on the form and paints the relief

A stroke's normal today does four jobs:
- its (T, B) frame;
- its 3D bounding box;
- the crease guard;
- the normal it writes.

Split them:
- `seed_nrm`, the **form normal** (barycentric shading normal, as today), keeps the first
  three. The detail stroke sits on the real surface and respects the form's folds. The
  crease guard keeps comparing form against form: Cycles' baked texel normals are form
  normals.
- `out_nrm`, the **painted normal**, is only what gets written.
  - **Every stroke paints the relief averaged at its own size.** A detail stroke paints
    its region's averaged detail normal (§3).
  - When a detail map is set, a **form stroke** also paints the detail normal averaged over
    its own responsibility region, via one window read. Otherwise a uniformly tilted part of
    the map (variance 0, mean not straight up) would be lost wherever no split happens.
  - With no detail map, `out_nrm = seed_nrm`, so nothing changes from today.
- **No split, no detail stroke.** The detail pass only emits the children of nodes that
  split. Its starting nodes are the form strokes' own regions and are never emitted. A
  region whose variance is ≤ `tol` at the first level adds nothing; its form stroke
  already covers it. A flat map adds 0 strokes.

The split is threaded through:
- `baker.resolve_uv(..., out_nrm=None)`: the output uses `out_nrm` when given.
- `gpu_resolve.gather`: gathers `out_nrm[idx]`. The GPU kernel is untouched, because it
  returns only the winner index.
- Preview: a **5th stroke row** holds `out_nrm`, and the FRAG shades from it. Row 2 stays
  the form normal used by the shared search. `STROKE_ROWS = 5`; the compute bake ignores
  the new row.

### 2. Where detail strokes go: split where the detail normals disagree

A **region** is an LEB triangle (§1.2's longest-edge bisection) of a mesh triangle.
**Splitting** it gives two equal-area halves, each a smaller stroke.

**Variance of unit normals** is `1 − |mean n|²`. Agreeing normals keep a long average;
disagreeing ones shrink it. It ranges 0..1 and needs only sums of `n.x, n.y, n.z` and a
count.

**Summed-area tables** are built at analysis size ≤ 2048: `Σ n` for x, y, z (3 float64
arrays).
- The sum over any rectangle is 4 reads. The texel count is just the window's area
  `w × h`, since the map tiles and every texel is used.
- A region's window is its UV triangle's bounding box in texels, after Tiling.
- Windows always wrap. One crossing the texture edge is split into up to 4 rectangles; a
  window spanning a full repeat uses the whole texture.

**The texel normals `n`** are the detail map decoded as tangent space: `2·rgb − 1`,
strength applied as `normalize(lerp((0,0,1), n, strength))`, then normalized.

**Splitting rule (top-down, as simple as that):**

```
for each node, starting from the form strokes' size in its triangle:
    var = window_var(node's UV bounding box)       # summed-area tables, 4 reads each
    if var > tol and node is above the floor:  split -> both children become strokes, recurse
    else:                                      stop
```

- The start depth is `ceil(log2 k_tri)`, where `k_tri` is the form strokes in that
  triangle. The floor is Min Detail Size, or 2 texels of UV extent.
- The walk is vectorised level by level over all active nodes and stops when none remain.
  Work is proportional to the strokes produced.
- Lowering `tol` only adds strokes, so it is nested.
- **Accepted limitation:** a tiny bump inside a region that is otherwise flat can dilute
  that region's variance below `tol` and be missed. Because the walk starts at form-stroke
  size, the dilution is bounded to within one form stroke.
- Results are cached per (mesh, map, strength, tiling, tol, Min Detail Size).

### 3. Each detail stroke's data

| field | value |
|---|---|
| position | child-triangle centroid (object space) |
| radius | `stroke_radius(child_area, child_aspect, density=density)` (§1.3 unchanged) |
| UV | barycentric (as today) |
| `seed_nrm` (form) | barycentric shading normal (as today) |
| `out_nrm` (painted) | mean tangent-space normal over the child's window → object space via the Normal Map node's convention: `normalize(T·x + B·y + N·z)` |
| direction | inherited from the form stroke owning its root region (see below) |
| id | continues after the form strokes |

- **Tangent frame:** `(T, sign)` from `mesh.calc_tangents(uvmap)` (MikkTSpace, as
  Blender's Normal Map node uses), interpolated barycentrically, with `B = sign·(N × T)`.
  If `calc_tangents` fails (n-gons), fall back to per-triangle UV-derivative tangents.
- **Direction:** form strokes' fit (§1.4) runs **on form strokes only**, so it is
  unchanged. A tiny, dense neighbourhood would make the curvature fit noisy.
  - Each detail stroke **inherits** the tangent of the form stroke that owns its root
    region: the region at depth `ceil(log2 k_tri)` whose path matches that form stroke's
    top path bits. The result is projected into the detail stroke's plane.
  - If no form stroke owns that region, it uses the double-angle average of its face's
    form strokes (every face has ≥ 1).
  - This is O(N). A brute-force 3-nearest lookup was measured at 5.9 s for 50k × 20k, so
    it is not used.

### 4. Controls: a "Surface Detail" box in the panel

- **Detail Normal**: an Image. Empty means off, with byte-identical output.
- **Strength**: 0–2, default 1 (as on the Normal Map node).
- **Tiling**: UV repeat count, default 1. It multiplies the bake UV map; the image is
  assumed to Repeat.
- **Detail Sensitivity**: 0–1, mapped to `tol`; higher means more splitting. The panel
  shows the resulting detail stroke count; exceeding the budget is refused as today.
- **Min Detail Size**: the floor, as a fraction of the median form stroke radius.

The map is read with the bake's UV map. Other coordinate setups are out of scope.

## Cost

Measured with numpy at 2048² (this machine, synthetic data), except where marked
*estimate*.

**One-time per (mesh, map, strength, tiling, sensitivity, Min Detail Size); cached:**

| Step | Time | Peak memory |
|---|---|---|
| Read the image from Blender (a >2K map is first reduced in C via `img.copy().scale()`) | 0.2–0.5 s *estimate* | 67 MB (RGBA float32) |
| Decode to unit normals | 0.04 s | 50 MB |
| Summed-area tables, 3 × float64 | 0.25 s | 101 MB |
| Window variances | 0.03 s per 200k windows (wrapping adds ≤ 4× reads on edge-crossers) | small |
| Split walk: LEB cut + barycentric UVs per level (visits ~2 × detail strokes) | < 0.2 s at 50k strokes *estimate* | a few MB |
| Detail-stroke direction (inherited, see §3) | O(N), negligible | small |
| **Total** | **~0.5–1 s** | **~220 MB peak** |

- Pixels, normals and tables are freed after the walk. Only the detail seeds stay cached,
  about 60 B each (3 MB at 50k).
- Moving Detail Sensitivity reruns only the walk: the tables are kept while the Surface
  Detail box is being edited.

**Ongoing, per detail stroke:**
- The same as a form stroke in resolution (GPU bake 0.09 s today; CPU fallback grows with
  strokes per tile).
- +16 B in the new 5th stroke row.
- The total stays within `STROKE_BUDGET` (200k).
- Preview: no extra texture (the relief is baked into `out_nrm` per stroke) and one extra
  row fetch per fragment.

**With no detail map:** zero cost; the path is skipped and the output is byte-identical.

**Risk to measure:** preview cell occupancy where many small strokes cluster. If hot
cells slow the per-fragment loop, size the grid from the detail population
(`upload_strokes(max_dim=...)`).

## Files

- `autostroke/core/detail.py` (new, pure numpy): `decode_normals`,
  `sum_tables`, `window_var` (with wrap), `split_walk(tris, start_depth, tol, floor)`,
  `detail_seeds(...)`
- `autostroke/core/sampling.py`: reuse `leb_leaves`, `tri_area`, `tri_aspect`,
  `stroke_radius`
- `autostroke/core/baker.py`: `resolve_uv(..., out_nrm=None)`, used only for the written
  normal
- `autostroke/gpu_resolve.py`: gather `out_nrm`
- `autostroke/bridge/mesh.py`: per-loop tangents + sign for the bake UV map (with
  fallback)
- `autostroke/bridge/images.py`: read the image, reduced to ≤ 2048 in C first
- `autostroke/bridge/seeds.py`: `build_seeds(..., detail=None)` appends detail seeds with
  `out_normal`; inherited directions; stats (detail count, variance floor hit)
- `autostroke/shaders.py`, `livepreview.py`: `STROKE_ROWS = 5`; row 4 = `out_nrm`; FRAG
  shades from row 4
- `autostroke/ops/bake.py`, `livepreview.py`: seed-cache keys gain the detail settings;
  pass `out_nrm`
- `autostroke/props.py`, `ui/panel.py`: the Surface Detail box
- `ALGORITHM.md`: new §1.5 "Detail level"; §2.3 notes the form/painted normal split
- Tests: new `test_detail.py`; extend `test_gpu_resolve.py`, `test_live.py`, and the
  byte-identity guards

## Verification

- `test_detail.py` (stub bpy, numpy):
  - `window_var` equals brute-force `1 − |mean n|²` over the same texels at random sizes
    and positions, including windows that wrap across the edge (compared against
    `np.roll`) and windows spanning a full repeat;
  - a flat normal map gives 0 detail strokes;
  - a checkerboard of two normals splits only along its edges, down to the floor;
  - stroke count is monotone in `tol`, and lowering `tol` keeps every existing stroke
    (nesting);
  - the floor stops splitting on pure noise.
  - Each check is mutation-tested.
- `test_gpu_resolve.py`: `gather` with `out_nrm` is still byte-identical to `resolve_uv`
  given the same `out_nrm`.
- No detail map: seeds, maps and stroke rows 0–3 are byte-identical to today, and every
  suite passes.
- Real Blender, on Body with a detail normal map:
  - the preview shows the painterly relief;
  - GPU and CPU bakes agree as before;
  - report counts, max cell occupancy, and the Cost-table timings and memory for a 2K and
    an 8K map;
  - render before/after; `.blend` untouched.
