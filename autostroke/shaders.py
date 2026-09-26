"""The stroke search in GLSL -- one source, used by the live preview AND the GPU bake.

The placement rule used to exist twice on the GPU side of things: once in the preview's
fragment shader, and (briefly) once in a bake prototype. Two copies of a rule drift, and
every preview/bake mismatch so far came from exactly that. So the rule lives here, once, as
a GLSL function both shaders include verbatim, and tests/test_live.py asserts that they do.

No imports, no bpy: this module is plain strings so it can be read by the tests and by both
shader builders without dragging a GPU context into either.

What the search must reproduce is core/baker.resolve_uv, per texel:
  - candidates are the strokes binned into this point's grid cell (the grid is only a
    speed-up -- it never excludes a stroke that could win);
  - sphere:   |p - centre|^2 < r^2          (bounds thickness off the stroke's plane)
  - box:      lu, lv inside the painted rectangle, in the stroke's own (T, B) frame
  - mask:     the brush image, bilinear, > 0.5 -- computed by HAND with texelFetch using
              sample_mask()'s exact convention, so it is independent of GPU sampler state
              and filtering precision, and clamped so it cannot bleed into the next brush
              in the atlas;
  - crease:   dot(surface normal, stroke normal) > cos(Stroke Cutoff);
  - winner:   the SMALLEST passing radius; on an exact tie the HIGHER index, which is
              resolve_uv's lexsort((-id, r)) order. Seed ids are np.arange, so index == id.
"""

STROKE_TILE_W = 4096
"""Width of every stroke-data texture. Apple's Metal caps a 2D texture at 16384 texels wide
and treats a violation as a native assertion that kills Blender outright -- not an error
Python can catch. A texture whose width grew with the stroke count did exactly that past
16384 strokes, so every table here is a fixed-width grid instead: element i lives at
(i % 4096, i / 4096)."""

STROKE_ROWS = 4
"""RGBA rows per stroke in the stroke table (see livepreview.pack_strokes)."""

STROKE_SEARCH = """
vec4 stroke_row(int i, int row)
{
    ivec2 tc = ivec2(i % STROKE_TILE_W, (i / STROKE_TILE_W) * STROKE_ROWS + row);
    return texelFetch(u_stroke, tc, 0);
}

/* (start, count) of the grid cell containing object-space point p, into u_list */
ivec2 cell_range(vec3 p)
{
    ivec3 gdim = u_dim_n.xyz;
    ivec3 c = ivec3(floor((p - u_grid_lo.xyz) * u_grid_inv.xyz));
    c = clamp(c, ivec3(0), gdim - 1);
    int cell = c.x + gdim.x * (c.y + gdim.y * c.z);
    int cell_w = u_misc.y;
    vec2 sc = texelFetch(u_cell, ivec2(cell % cell_w, cell / cell_w), 0).rg;
    return ivec2(int(sc.x), int(sc.y));
}

/* core/baker.sample_mask, verbatim in GLSL: m = 0 and m = 1 land on the first and last
   pixel CENTRES of the brush, v runs bottom-up, bilinear by hand. */
float brush_mask(int bi, vec2 m)
{
    int cols = u_misc.z;
    int tile = textureSize(u_brush, 0).x / cols;
    float x = clamp(m.x, 0.0, 1.0) * float(tile - 1);
    float y = clamp(1.0 - m.y, 0.0, 1.0) * float(tile - 1);
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    int x1 = min(x0 + 1, tile - 1);
    int y1 = min(y0 + 1, tile - 1);
    float fx = x - float(x0);
    float fy = y - float(y0);
    ivec2 o = ivec2((bi % cols) * tile, (bi / cols) * tile);
    float a = texelFetch(u_brush, o + ivec2(x0, y0), 0).r;
    float b = texelFetch(u_brush, o + ivec2(x1, y0), 0).r;
    float c = texelFetch(u_brush, o + ivec2(x0, y1), 0).r;
    float d = texelFetch(u_brush, o + ivec2(x1, y1), 0).r;
    return a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy)
         + c * (1.0 - fx) * fy + d * fx * fy;
}

/* The placement rule. Returns the winning stroke's index, or -1 if no stroke paints p. */
int find_stroke(vec3 p, vec3 n_in)
{
    vec3 n = (dot(n_in, n_in) > 1e-24) ? normalize(n_in) : vec3(0.0);
    int nstroke = u_dim_n.w;
    int list_w = u_misc.x;
    float cos_cut = u_grid_lo.w;
    ivec2 sc = cell_range(p);

    float best_r = 1e30;
    int best = -1;
    for (int k = 0; k < sc.y; ++k) {
        int li = sc.x + k;
        int i = int(texelFetch(u_list, ivec2(li % list_w, li / list_w), 0).r);
        if (i < 0 || i >= nstroke) continue;

        vec4 r0 = stroke_row(i, 0);                      /* centre.xyz, radius */
        float r = r0.w;
        /* smallest radius wins; exact tie -> higher index, as resolve_uv's order has it */
        if (r > best_r || (r == best_r && i < best)) continue;

        vec3 d = p - r0.xyz;
        if (dot(d, d) >= r * r) continue;                /* sphere */

        vec4 r1 = stroke_row(i, 1);                      /* T.xyz, luminance */
        vec4 r2 = stroke_row(i, 2);                      /* normal.xyz, brush index */
        vec4 r3 = stroke_row(i, 3);                      /* painted box lu/lv bounds */
        vec3 T = r1.xyz;
        vec3 SN = r2.xyz;
        vec3 B = cross(SN, T);
        float lu = dot(d, T);
        float lv = dot(d, B);
        if (lu <= r3.x || lu >= r3.y || lv <= r3.z || lv >= r3.w) continue;   /* box */

        float two_h = 2.0 * r * 0.7071067811865476;      /* 2 * r / sqrt(2) */
        vec2 m = vec2(lu / two_h + 0.5, lv / two_h + 0.5);
        if (brush_mask(int(r2.w), m) <= 0.5) continue;   /* mask */

        if (dot(n, SN) <= cos_cut) continue;             /* crease */

        best_r = r;
        best = i;
    }
    return best;
}
"""


def with_tokens(src):
    """Substitute the layout constants into GLSL source. Done as plain text replacement,
    not uniforms: the preview's push-constant budget is exactly full (128 bytes), and a
    constant we choose ourselves needs no uniform slot."""
    return (src.replace("STROKE_TILE_W", str(STROKE_TILE_W))
               .replace("STROKE_ROWS", str(STROKE_ROWS)))


SEARCH = with_tokens(STROKE_SEARCH)
"""The search, ready to prepend to a shader's own main()."""
