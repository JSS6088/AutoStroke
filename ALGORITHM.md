# AutoStroke — the algorithm

Two halves. **Placement** decides where strokes sit on the mesh, how big each one is and
which way it points. **Resolution** decides which stroke owns each texel of the output.

## What it computes

An **indirection map**: for every texel, a UV pointer to the representative point of the
stroke that paints it (R,G), plus that stroke's tonal random (B). A shader reads the
pointer, then reads the colour map at that UV, so every texel of a stroke returns one flat
colour. A companion map stores each stroke's normal.

A texel is won by the **first stroke in candidate order** whose brush shape covers it and
whose normal agrees with the surface there — not by the nearest stroke.

## Notation

| symbol | meaning |
|---|---|
| `F`, `A_f` | faces, area of face `f` (world space — density is per m²) |
| `p` | stroke density, strokes per m² |
| `N` | strokes |
| `M` | valid texels (~0.84M at 1K, ~3.35M at 2K, ~13.4M at 4K) |
| `C` | chunk size, texels per spatial tile |
| `S` | strokes surviving rejection per tile, on average |
| `r` | effective radius = seed radius × Stroke Size × size variation |

---

# Part 1 — Placement

## 1.1 How many strokes per face

```
N_f = clamp( floor(A_f · p + u_f), min, max )        u_f = hash01(f) ∈ [0,1)
```

**Dithered, not rounded.** With rounding, every face of a near-uniform mesh crosses
`A·p = 0.5` at the same density, so the whole mesh jumps from 1 stroke to 2 at once and
every stroke shrinks by √2 together. The per-face offset staggers those crossings, so the
average follows density smoothly while each face stays deterministic.

`u_f` comes from splitmix64 on the face index — deterministic, and independent of every
other face.

**Density is solved, not divided.** The artist sets a total stroke count; `p` is found by
bisection (40 steps) because the clamps bend the relationship: on a low-poly mesh a few
large faces hit `max` and stop contributing, so `target / total_area` under-delivers. Total
strokes rise monotonically with `p`, which is what makes bisection valid. If the clamps
make the target unreachable, the solver returns the most the mesh can produce.

## 1.2 Where within a face

A face's `N_f` points are spread over **its** triangles by area, so a quad behaves like one
quad rather than two independent triangles. Within a triangle, positions come from
**longest-edge bisection** (LEB):

> At each level, cut the current longest edge at its midpoint. Bit `l` of the point's index
> picks which half to descend at level `l`. Recurse 32 levels; the point is the centroid of
> the leaf.

Both halves have exactly equal area — same apex, equal bases — so sampling stays
area-uniform at every depth. Two properties follow:

- **Nested.** A point's leaf depends only on its own index, never on how many points the
  face was asked for. Raising density adds strokes without moving existing ones.
- **Shape-aware.** The cut always crosses the longest side, so a sliver is divided along its
  length until the pieces are compact. A four-way midpoint split instead divides both
  directions every level, wasting half its subdivisions on a width with room for one row of
  points — measured **16 usable positions along a strip vs 90**.

Two details exist to protect that nesting:

**Choosing the triangle** uses the radical inverse (van der Corput base 2) of the stroke's
index, not `(k + 0.5)/n` — the latter depends on `n`, so raising density would reshuffle a
quad's strokes between its two triangles. Any prefix of the radical-inverse sequence is both
evenly spread and unchanged as the sequence grows.

**Filling the index's high bits.** For `i < N` only the low bits carry information; left as
zeros, every deep level takes the same side and the point drifts into a corner of its region
instead of sitting inside it. They are filled from `hash(i)` above `i`'s bit length — still
a function of `i` alone, so nesting holds.

## 1.3 How big each stroke is

```
r = sqrt(leaf_area) · (leaf_aspect / 2) ^ ½        aspect = L² / 2A
```

`sqrt(a)` is the side of a *square* of that area, so it matches a face's real extent only
when the face is compact. A sliver of length `L` and width `w` needs `r ≈ L/2`, while
`sqrt(a) = sqrt(Lw/2)`; their ratio is `sqrt(aspect/2)` — which is where the exponent ½ and
the divide-by-2 both come from. Neither is tuned. Measured on a 1 m × 1 cm bevel at 8
strokes: **64% covered without the correction, 100% with**. Exponent 1 covers no better and
triples the spill onto neighbouring faces.

Aspect is **longest edge over the height above it** (`L²/2A`), not longest-over-shortest: an
isosceles sliver has two long edges and a long base, so longest-over-shortest reports 2.0 on
a shape that is really 100:1 — skipping the correction on exactly the faces that need it.

Dividing by the ideal aspect (2) keeps compact faces near their uncorrected size; an
equilateral triangle's leaves sit at aspect 2.31, so a bare `sqrt(aspect)` would enlarge
every stroke by 1.5×.

The **leaf** whose area and aspect are used is the point's *responsibility region* — the LEB
leaf at depth `ceil(log2(points in this triangle))`, which is just deep enough to separate
this triangle's points. Position uses depth 32. Two different depths on purpose: size should
track how much surface a stroke is responsible for, position should not depend on how many
points were asked for.

**The size cap.** Max Strokes per Face bounds *count*, not size. Once a face's natural count
(area × density) passes that cap, `leaf_area = face_area / max_strokes` keeps growing exactly
as fast as the face does, and `r` above has no ceiling — measured, a 1024× range of face area
under a fixed cap of 128 produced a 32× range of stroke radius, unbounded.

The fix asymptotes toward `R = 1/√density` rather than inventing a new limit: on any face with
room to reach the target density (not clamped by Min/Max), `leaf_area → 1/density` in the
limit and `r` already converges to exactly `R` on its own — measured within 0.3% a thousand
times past the reference area, before the cap ever bites. `R` is therefore the size the rest
of the mesh is *already* using; a capped face asymptoting toward it looks proportionate
instead of erupting.

```
r_final = R · tanh(r / R)
```

`tanh(x) ≈ x` for `x ≪ 1`, so a normal, uncapped stroke (`r ≪ R`) is touched by a fraction of a
percent — measured 0.3% at `r = R/10`. As `r` approaches or passes `R` the curve bends over
smoothly and never exceeds it, which is the point: two capped faces of slightly different area
should look like a smooth falloff, not a hard clip that shows as a visible size discontinuity
between them. `density` is optional on `stroke_radius` — omit it and the formula above is
untouched, which is what every caller that predates this did and still does.

## 1.4 Which way each stroke points

Strokes run along the surface's **least-curved direction**, so they follow the form rather
than a UV axis. Per seed:

1. **kNN over seeds** (K = 14). Neighbours are other strokes, never texels. The distance
   matrix is (rows × seeds), so it is chunked by a byte budget rather than a row count.
2. **Reject neighbours across a gap.** Nearest-by-Euclidean reaches *through* thin features,
   so a seed can be "near" a point on the opposite surface and that bogus normal delta
   corrupts the fit. Neighbours whose normal disagrees by more than 60° are dropped; if
   fewer than 6 survive, keep them all rather than fit on nothing.
3. **Fit the shape operator.** Express neighbour offsets `u` and neighbour normal deltas `v`
   in the seed's local tangent frame and solve the 2×2 system `v ≈ u·S` — "step this way,
   the normal tilts that way". Symmetrise `S`, take `eigh`: the eigenvectors are the
   principal directions, the eigenvalues the principal curvatures, and the eigenvector with
   the **smaller |eigenvalue|** is the direction along which the normal changes least.

   Done in closed form over all seeds at once (explicit 2×2 inverse, `einsum` for `uᵀu` and
   `uᵀv`) rather than a per-seed least-squares. Neighbour masking is a 0/1 weight: a zeroed
   row contributes nothing to either product, so "drop that neighbour" needs no special case.
   A collinear patch gives `det ≈ 0` and falls back to an arbitrary in-plane direction.

4. **Smooth into a coherent field** — 20 passes, each blending halfway toward the neighbour
   average. Two traps this exists to avoid:

   - **Direction is modulo 180°** (`T` and `−T` are the same stroke), so averaging raw
     vectors lets agreeing neighbours cancel to zero. Fixed by the double-angle
     representation `(cos 2θ, sin 2θ)`, in which `T` and `−T` are the same point.
   - **Every seed has its own frame**, so bare angles are not comparable. Each neighbour's
     direction is projected into seed `i`'s tangent plane and measured against `i`'s `e₁`
     first — a discrete stand-in for parallel transport.

   Many small passes rather than one large `K`: confident directions propagate outward while
   real feature structure survives, and umbilic regions (which have no direction of their
   own) inherit from their neighbours.

Finally each stroke gets an orthonormal in-plane frame `(T, B)`: project the fitted tangent
into the surface plane, normalise, `B = N × T`. A tangent that is only approximately
perpendicular still yields a valid frame; degenerate input falls back to an arbitrary
perpendicular rather than producing NaNs.

---

# Part 2 — Texel resolution

`core/baker.resolve_uv`. Brute force is `Θ(M·N)` — every texel against every stroke.

## 2.1 Per-stroke setup, once

1. **Brush choice** — `brush_index(seed_id, n)` hashes each stroke to one brush of the set.
2. **Rotate the frame** — Global Rotation, minus the brush's own long-axis alignment, plus
   per-stroke jitter, as one combined rotation.
3. **Effective radius** — `r = seed_radius · mask_scale · 2^(u · size_random)`. The single
   place `r` is built, so `r²` (sphere test), `h = r/√2` (footprint half-side) and the
   candidate order all follow from it.
4. **Candidate order** — `lexsort((−seed_id, r))`: **smallest stroke first**, id descending
   on ties. This order *is* the tie-break rule, and it is why a fine mark is never swallowed
   by a coarse one. Sorted on the **final** radius, because the rule is about the mark that
   lands, not the patch it came from.
5. **Bounds** — each brush's painted bbox → per-stroke `lu`/`lv` limits, and the world AABB
   of each stroke's paintable region (§2.5).

`Θ(N log N)`, plus `Θ(brush pixels)` to measure each brush once.

## 2.2 Spatial ordering

Quantise every texel's 3D position to 10 bits per axis, interleave the bits into a **Morton
code**, argsort. Texels arrive in UV scanline order, which is scattered in 3D, so a raw
block of them covers the whole model; after this a block is a compact lump of surface.
**Without this step, the rejection in §2.3 excludes nothing.** `Θ(M log M)`; 0.58 s at
M = 3.35M.

## 2.3 Per tile

For each contiguous block of `C` texels:

**Reject distant strokes** — AABB overlap against every stroke at once:

```python
_near = np.flatnonzero((box_lo <= _hi).all(1) & (_lo <= box_hi).all(1))
```

`Θ(N)` per tile. `flatnonzero` returns **ascending** indices into arrays already in
candidate order, so survivors keep that order for free — a subsequence of a sorted list is
still sorted. There is no per-tile sort and none is needed.

**Then, for each surviving stroke in candidate order**, a funnel:

| stage | test | purpose |
|---|---|---|
| offset | `d = p − s[j]` | vector from stroke to each texel |
| sphere | `d·d < r²` | bounds the normal direction (§2.5) |
| project | `lu = d·T`, `lv = d·B` | 3D → stroke-local 2D |
| footprint | `lu, lv` in the painted box, and `~done` | the tight rectangular bound |
| reach | the texel's triangle is in the stroke's reach set | only surface reachable across the mesh (§2.10) |
| brush | `sample_mask(...) > thresh` | the silhouette itself |
| crease | `dot(texel_normal, stroke_normal) > cos θ` | reject strokes reaching around a fold |

Survivors claim the texel: write UV, normal, luminance, set `done`. **`~done` is what
enforces candidate order** — a texel already won is invisible to every later stroke.

The crease test is an **angle**, not a distance off the tangent plane: a distance threshold
scales with the stroke, so every stroke crossing one fold stopped somewhere different
(spread **84.9°** — this is what read as strokes being chopped mid-shape). An angle is a
property of the surface, so every stroke stops on the same locus: spread **33.7°**, p90
per-texel normal error 31.6° → 23.8°.

A rejected texel falls through to the next candidate; if all reject it, it stays uncovered
and is filled with the **true surface normal**, shading like the untouched model. Keeping
the least-wrong stroke instead painted normals up to 179.6° off their own surface — those
face into the model and render black.

## 2.4 Restore order

Un-permute the four outputs. Uncovered texels keep their own UV (pass-through).

## 2.5 The bounds that make it fast

Both are **exclusions, not approximations**: each provably contains everything that could
pass the test after it, so output is byte-identical to brute force.

**The footprint is the painted box, not the stamp square.** The mask image maps onto a
square of side `√2·r`, but a brush paints a sliver of it — on the shipped set the painted
bbox is **8.4–18.4%** of the square. Outside the painted pixels the mask is ≤ threshold by
construction, and a bilinear blend of sub-threshold values is sub-threshold, so a sample out
there *cannot* pass. The box is expanded one texel each way because `sample_mask` reads
`floor(x)` and `floor(x)+1` — that margin rescues 1,080 passing samples on a 256px brush.
Measured: footprint texels **22.2M → 1.70M (13×)**.

**The rejection bound is a box, not a sphere.** The paintable region is an **OBB**: the
painted rectangle in `(T, B)`, and the sphere's reach along `N`. The normal half-extent must
be the full `r`, not zero — the footprint test constrains `lu` and `lv` only, and nothing
else bounds how far off the tangent plane a texel may sit. Bounding just the rectangle looks
tighter and loses strokes (measured: 111). That OBB is collapsed once per stroke into its
world AABB — `|T|·h_u + |B|·h_v + |N|·r`, the tightest axis-aligned bound on an oriented box
— and the per-tile test is AABB vs AABB. Measured: stroke-tile pairs **306,458 → 180,718**;
brute force confirms 0 of 6,117 rejected strokes would have painted anything.

**What the sphere test is still for.** Its one unique job is bounding the *normal*
direction, since the footprint constrains `lu` and `lv` only — without it a texel straight
"below" a stroke, through the mesh, would pass. In practice it rejects almost nothing: for a
real brush at `r = 1` the painted rectangle is 0.32r × 1.18r, but the sphere does not bite
until ~0.79r along the normal, so geometry must fold back inside that narrow rectangle while
standing most of a radius off the tangent plane. Measured **0 of 775,801** texels inside the
painted box rejected by it on a real bake. It stays because it defines current output and
because the stage computing it is mostly computing `d` — which the projection needs anyway.

## 2.6 Complexity

| | brute force | now |
|---|---|---|
| per-stroke setup | `Θ(N log N)` | `Θ(N log N)` |
| spatial sort | — | `Θ(M log M)` |
| rejection | — | `Θ((M/C)·N)` |
| stamp work | `Θ(M·N)` | `Θ(M·S)` |

`Θ(M·N)` → `Θ(M log M + (M/C)·N + M·S)`. `S` is `O(1)` as quality rises: more strokes over a
fixed surface are *smaller* strokes, so the number overlapping any one texel stays flat.

**Choosing C.** Two costs pull opposite ways: rejection is `(M/C)·N` and wants large `C`; a
surviving stroke still processes its whole tile and wants small `C`.

```python
C = clip(sqrt(M · N) / 128, 1024, 16384)
```

Fitted, not derived. The valley is shallow — 1,024 → 1.91 s, 2,048 → 1.85 s, 4,096 → 2.17 s,
32,768 → 4.44 s — so being roughly right is enough.

## 2.7 Measured

Resolve, on a real 1024 bake:

| version | resolve |
|---|---|
| brute force | ~70 s (2K) |
| Morton + sphere rejection | 6.98 s |
| + footprint clipped to the painted box | ~4.4 s |
| + box-vs-box rejection | **2.93 s** |

Whole bake **7.41 s → 3.31 s**.

## 2.8 Tried and declined

| idea | result | why it loses |
|---|---|---|
| short-circuit the funnel (subset after the sphere test) | **1.10×**, byte-identical | removes arithmetic, but arithmetic is not the cost; and after box rejection the average candidate reaches **63.6%** of its tile |
| true OBB vs tile AABB (separating axis) | **40% fewer pairs**, test costs 2.52 s vs 0.09 s | 15 axes ≈ 60 numpy ops per tile instead of 2; loses by ~8× |
| cheaper mask sampling (256px, nearest) | 1.4×, but changes output | the mask is only 15% of resolve now |

The common thread: **numpy dispatch, not arithmetic, is the remaining cost.** An op on 1,024
elements costs 0.34 µs, of which 0.33 ns/element is work — so ~2.7M calls is ~0.9 s, about a
third of resolve. Anything that adds calls to save work loses.

## 2.9 The next order of magnitude

A **per-texel gather**: bin strokes spatially, and for each texel test only its own cell.
`Θ(M·k)` against `Θ(M·S)` today. Dispatch being the bottleneck argues *for* it — it collapses
~180,000 small loop iterations into ~1,000 large ones. Two obstacles:

- **Cell size with non-uniform radii.** Radius spans 10–100× on real meshes, the classic
  failure of a uniform grid. Fixes: a hierarchical grid, or one grid at `2·median r` with
  each stroke inserted into every cell it overlaps, or one grid plus a short oversize list.
- **Candidate order resists vectorisation.** A texel takes the first stroke in order passing
  mask *and* crease — a variable-length, early-terminating per-texel loop. The vectorised
  form is expand-and-reduce: build `(texel, stroke)` pairs, evaluate all at once, sort by
  `(texel, rank)`, take the first passing pair per texel. That materialises `M·k` pairs, so
  it needs blocking.

The trigger is 8K maps or ~100,000 strokes. (The GPU bake, Part 3, is this per-texel
gather.)

## 2.10 Reachable surface

**The bug.** Every test above is measured in 3D or in the stroke's own plane. None of them
knows which *surface* a stroke was placed on:
- the sphere test is distance only;
- the painted box and brush mask are measured in the stroke's plane;
- the crease test passes whenever the normals agree.

So a stroke painted any nearby surface facing the same way: a second layer a few mm away, a
panel floating over a hull, an eyeball in its socket, the far side of a fold. The normal
map mostly hid it, because the stolen normal is nearly right. The indirection map did not:
colour read through the pointer came from the wrong part. Measured at 1K, texels won by a
stroke on a *different connected part* of the mesh:

| object | without reach | with reach |
|---|---|---|
| two planes 2 mm apart (one mesh) | 498,807 of 997,724 | **0** |
| Suzanne (eyes in sockets) | 9,921 of 791,712 | **0** |
| Body | 16,913 of 760,143 | **0** |

**The rule.** A stroke may paint only triangles it can reach **across the surface**:
triangles inside its sphere that connect back to its own triangle through triangles also
inside the sphere. Stacked layers are both inside the sphere but share no path. The two
sides of a fold connect only around the fold, which lies outside the sphere. No threshold is
involved. Distance-off-plane, UV-distance and curvature-fit tests were considered and
rejected: each needs a tolerance and fails when the gap between surfaces is smaller than it,
and UV distance cuts every stroke at every UV seam.

**Welded by position.** Triangles are neighbours if they share a vertex *position*, not a
vertex index. Imported meshes routinely split vertices along UV seams and hard edges;
joining by index would cut the surface into its UV islands and stop strokes at every seam.
Neighbours share any welded vertex, not just an edge, so fans and T-junctions stay one
surface. `tri_adjacency` builds the CSR with the repeat/offset trick.

**The flood fill** runs for all strokes at once. Each ring expands the frontier of
`(stroke, triangle)` pairs to their neighbours, keeps those whose **exact** closest point is
inside the stroke's sphere (Ericson's closest-point-on-triangle, vectorised), and drops pairs
already visited. Exact rather than a centroid-plus-circumradius bound: on a low-poly mesh the
slack of a big triangle can bridge a fold. The sphere is `baker.effective_radius`, the same
radius the search paints with, so the reach sets can never be sized differently from it.
`test_reach` proves the vectorised fill equals a plain per-stroke BFS, pair for pair.

**Which triangle a texel is on.** The mesh's UV triangles are rasterised at the bake
resolution, once per mesh and resolution. Where UVs overlap, the triangle whose interpolated
3D position matches the baked position wins, so the answer agrees with what Cycles baked.
Cycles' bake margin marks texels outside every triangle as valid; they take a neighbour's
triangle, grown by 4-neighbour steps. That takes `2 × margin` passes, not `margin`: one
4-neighbour step per pass reaches only half as far diagonally, and measured on Suzanne the
single-margin version left 10,932 margin texels with no triangle.

**Cost**, measured at 1K with 3,500 strokes:

| object | triangles | reach pairs | build | texel map |
|---|---|---|---|---|
| stacked planes | 2,304 | 101,566 | 0.07 s | 0.35 s |
| Suzanne | 968 | 92,505 | 0.07 s | 0.27 s |
| Body | 4,146 | 246,280 | 0.18 s | 0.25 s |

Both are cached: the texel map per mesh and resolution, the reach sets per seed set, Stroke
Size and Size Variation. Rotation, jitter, cutoff and brush changes rebuild neither. On the
CPU the reach test is one sorted-membership check per candidate, before the mask sample: a
few percent on resolve.

**Coverage.** Texels only other parts' strokes used to reach now fall back to the true
surface normal, as any uncovered texel does: Body 89.2% covered where it was 90.7%.

**Limit.** Parts that intersect without sharing vertices, such as an arm modelled into a
torso, are separate surfaces: strokes stop where they cross. Merging them (Merge by
Distance, a boolean union) makes them one surface.

---

# Part 3 — The GPU search: preview and bake

The same search, run on the GPU. One GLSL function, `find_stroke(p, n, tri)` in
`shaders.py`, does it: the strokes that can reach the point's triangle (§2.10) → sphere,
box, brush-mask and crease tests → winner.
Two shaders include that text verbatim: the preview's fragment shader and the bake's compute
shader. The artist therefore judges the same code that ships.

## 3.1 Preview

A fragment shader, so parameter changes are interactive. It produces no map at all:

```
fragment → object-space position + normal + its triangle (varyings; no baked maps)
         → the strokes that can reach that triangle (§2.10)
         → loop them, running the same tests as Part 2
         → keep the SMALLEST passing stroke, shade with its normal
```

Dropping resolution instead would have been the wrong axis: the output is a pointer table
inspected at full size, so coarse texels degrade exactly what is being judged. Rendering to
a texture and reading it back is also out — `img.pixels.foreach_set` costs ~0.3 s of Python
for 4M floats, enough to defeat "live" on its own.

**Candidates per triangle.** Each triangle carries the list of strokes that can reach it,
straight from the reach sets. The list is complete by construction, so a fragment reads only
its own triangle's list. It is also the rule itself: a stroke outside it may not paint there.
Each corner of the unindexed preview batch carries its loop-triangle index as a flat
varying, the same index the reach sets use.

This replaced a uniform 3D grid, where a stroke went into every cell its §2.5 box overlapped.
That grid was complete too (measured 67 / 71 / 79 strokes per cell at 1,000 / 4,000 / 20,000
strokes), but a cell holds every layer passing through it, which is exactly the bug of
§2.10. Per-triangle lists hold what reaches that surface. They grow on low-poly meshes, where
one triangle is large next to the strokes (Suzanne: mean 96, max 227 strokes per triangle;
Body: 59 and 127). The GPU bake still resolves a 1K map in about 0.06 s there.

## 3.2 Bake (the default)

The bake is not drawing anything; it maps arrays to arrays. So it is a **compute shader**:
read buffers in, one write buffer out, one thread per texel. It needs no rasterization of
its own, because the CPU path already has everything a texel needs: its 3D position and
normal from the cached Cycles maps, plus the `valid` mask. `gpu_resolve.resolve` takes
exactly `resolve_uv`'s inputs and returns exactly its 4-tuple. The position bake, cache,
gutter, dilation, EXR writing and material build are therefore shared, unchanged.

```
once per object   stroke rows, per-triangle table, list, brush atlas   (upload_strokes)
per chunk         ≤ 4096×1024 valid texels: position + triangle id (.w), normal
                  → two RGBA32F textures
                  output R32F texture prefilled with −2
dispatch          each thread: find_stroke → imageStore(winner index, or −1)
read back         any −2 left ⇒ the dispatch did not run ⇒ raise
gather (numpy)    uv = seed_uv[i], lum = hash01(seed_id[i]), nrm = normalize(seed_nrm[i])
```

**The shader returns only the winner's index.** Everything a winner writes is a function of
the stroke alone, so numpy gathers it with `resolve_uv`'s own definitions. The two resolvers
can then disagree only about *which* stroke wins a texel, never about what it writes.
`test_gpu_resolve` proves the gather rebuilds `resolve_uv`'s output byte for byte from its
winners. All brush-mask sampling still happens on the GPU: the brush shape is encoded in
which texels receive index `j`.

**The CPU and GPU run the same algorithm in a different loop order.** Numpy has a fixed
per-call cost, so the CPU runs stroke-major over Morton tiles (Part 2). The GPU already runs
one thread per texel, so it runs texel-major, gathering its triangle's candidates. The
same tests are applied to the same candidates.

**Aligned with the CPU wherever it is cheap:**
- *Brush mask.* It is sampled with a hand-rolled bilinear over `texelFetch`, using
  `sample_mask`'s pixel-centre convention, from a full-resolution atlas. This replaces
  hardware filtering: compute shaders have no screen derivatives, and the Python API
  exposes no filter-mode setter.
- *Tie-break.* An exact radius tie goes to the higher index, as in `resolve_uv`. The preview
  gets this too, because the function is shared.

**Measured** on the same cached maps at 1K, winner index per texel, with reach (§2.10):

| object | same winner | CPU resolve | GPU resolve |
|---|---|---|---|
| Body | 747,095 / 747,095 | 2.81 s | 0.061 s |
| Suzanne | 789,855 / 789,855 | 2.86 s | 0.066 s |
| stacked planes | 925,459 / 925,459 | 1.94 s | 0.129 s |

Through the real operator on a two-object batch (Body + Suzanne), a whole bake took 2.1 s
on the GPU against 8.1 s on the CPU, and the written maps were identical, texel for texel.
Before reach, the same batch differed in 13 and 1 texels of 1,048,576: float32 against
float64 at a brush silhouette or crease threshold. Such differences can still appear on
other meshes and graphics drivers; for a visual output that is fine.

**Limits.** No texture is wider than 4096 or taller than 1024. Metal caps textures at 16384
and treats exceeding it as a hard process abort, not a catchable error. Python's API has no
storage buffers, so every array is a texture used as a 2D array, element `i` at
`(i % 4096, i / 4096)`. When reading back, `np.array(Buffer)` walks a 2D buffer in the wrong
axis order. The readback therefore flattens the buffer and uses `np.frombuffer`.

## 3.3 CPU fallback

`resolve_uv`, the numpy resolver of Part 2, is the fallback. It runs:
- when the artist picks **CPU**;
- when there is no GPU context (`--background`);
- when the GPU path raises anywhere: shader compile, dispatch, or a readback that fails the
  sentinel/range checks.

On failure the job restarts on the CPU from scratch and the rest of the batch stays on the
CPU, with one warning. The report names the device each job used. The CPU time model
(`est_scale`) learns only from CPU jobs.
