"""Live viewport preview, drawn on the GPU.

Why this is a separate renderer rather than a faster bake: the bake resolves a texture, and
even at its fastest that is ~1-3 s for a full-resolution map. Dropping resolution to buy
latency was tried and is the wrong axis -- the output is a pointer table inspected at full
size, so coarse texels degrade exactly what the artist is judging.

So this does not produce a map at all. It draws the object in the viewport with a fragment
shader that finds each pixel's winning stroke directly. That skips the texture entirely,
and with it the readback trap: pulling a rendered map back through img.pixels.foreach_set
costs ~0.3 s of Python for 4M floats, which on its own would make "live" impossible.

    fragment -> object-space position + normal (varyings, no baked maps needed)
             -> look up its cell in a uniform 3D grid of strokes
             -> loop that cell, run the same tests resolve_uv runs
             -> keep the SMALLEST passing stroke, shade with its normal

Strokes are inserted into every cell their bounding box overlaps, so a fragment only ever
has to read its OWN cell -- the cell contains every stroke that could reach it. That is
what keeps the shader loop short.

This is a second implementation of the placement rules, and it is deliberately only
required to LOOK right: `core/baker.resolve_uv` remains the only thing that produces the
maps you ship. Treat disagreements as a preview bug, never as a reason to change the baker.
"""

import traceback

import numpy as np
import bpy

from .bridge import cache as seed_cache, mesh as mesh_bridge, seeds as seed_bridge
from .core import baker, geometry
from .core.config import INV_SQRT2

_handle = None          # the draw-handler token, or None when the preview is off
_state = None           # everything the handler needs; rebuilt when settings change
_error = ""             # last failure, shown in the panel instead of crashing


# ---------------------------------------------------------------------------
# Shaders. No #version line: Blender injects its own and rejects one here.
# ---------------------------------------------------------------------------

# Blender 5.0 removed direct GPUShader(vert, frag) construction, so the shader is built
# through GPUShaderCreateInfo. That changes the rules: uniforms, attributes and varyings
# are declared on the info object and must NOT appear in the GLSL, or they are declared
# twice and the compile fails.
#
# Push constants are limited (128 bytes), and a MAT4 alone is 64 of them -- so the odds
# and ends are packed into two vec4s and two ivec4s rather than passed individually:
#
#   u_grid_lo  = (grid origin .xyz, cos of the Stroke Cutoff)
#   u_grid_inv = (1 / cell size .xyz, depth offset along the normal)
#   u_dim_n    = (grid dims .xyz, stroke count)
#   u_misc     = (list width, cell width, brush atlas columns, view mode)
#
# That totals exactly 128: MAT4 64 + two VEC4 32 + two IVEC4 32. If a backend rejects it
# for size, the fix is to drop u_misc -- the texture widths are constants we choose, so
# they can be hardcoded in the GLSL, and columns/mode fit in the spare .w slots above.

VERT = """
void main()
{
    v_pos = pos;                       /* OBJECT space: strokes live here too */
    v_nrm = nrm;
    /* Blender has already rasterised this same mesh by the time a POST_VIEW handler
       runs, so drawing it again at identical depths z-fights -- a fine stipple across
       every flat area. Nudging the rasterised position a hair OUT along the normal puts
       our surface just outside Blender's so it wins consistently.

       Along the normal rather than in clip space on purpose: a clip-space z bias has to
       know whether the depth buffer is reversed, and this does not. v_pos keeps the true
       position, so stroke lookup is unaffected by the nudge. */
    gl_Position = u_mvp * vec4(pos + nrm * u_grid_inv.w, 1.0);
}
"""

STROKE_TILE_W = 4096
"""Fixed width for the stroke texture, same reasoning and same value as list_tex/cell_tex
below: MTLTextureDescriptor on Apple GPUs (and plenty of others) caps a 2D texture at
16384 texels wide. stroke_tex used to be exactly (stroke count, 4) -- width equal to the
RAW stroke count -- so any preview past 16384 strokes (well inside PREVIEW_BUDGET's
60,000) hit that cap and crashed the whole process: a native Metal assertion failure, not
a Python exception, so nothing in this addon could have caught or reported it. Tiling into
a fixed-width grid, the same way list_tex already had to be, removes the ceiling instead
of trying to guess a safe one. Hardcoded into the GLSL below (STROKE_TILE_W token,
substituted after this string) rather than passed as a uniform -- the push-constant
budget is already exactly 128 bytes with nothing spare, and a width we choose ourselves
needs no uniform slot at all."""

FRAG = """
vec4 stroke_row(int i, int row)
{
    ivec2 tc = ivec2(i % STROKE_TILE_W, (i / STROKE_TILE_W) * 4 + row);
    return texelFetch(u_stroke, tc, 0);
}

void main()
{
    vec3 n = normalize(v_nrm);
    ivec3 gdim = u_dim_n.xyz;
    int   nstroke = u_dim_n.w;
    int   list_w = u_misc.x;
    int   cell_w = u_misc.y;
    int   cols = u_misc.z;
    int   mode = u_misc.w;
    float cos_cut = u_grid_lo.w;

    ivec3 c = ivec3(floor((v_pos - u_grid_lo.xyz) * u_grid_inv.xyz));
    c = clamp(c, ivec3(0), gdim - 1);
    int cell = c.x + gdim.x * (c.y + gdim.y * c.z);
    vec2 sc = texelFetch(u_cell, ivec2(cell % cell_w, cell / cell_w), 0).rg;
    int start = int(sc.x);
    int count = int(sc.y);

    if (mode == 2) {                        /* occupancy, for debugging the binning */
        float f = clamp(float(count) / 64.0, 0.0, 1.0);
        fragColor = vec4(f, 1.0 - f, 0.0, 1.0);
        return;
    }

    float best_r = 1e30;
    vec3  best_n = vec3(0.0);
    float best_l = 0.5;
    bool  found  = false;

    for (int k = 0; k < count; ++k) {
        int li = start + k;
        int i = int(texelFetch(u_list, ivec2(li % list_w, li / list_w), 0).r);
        if (i < 0 || i >= nstroke) continue;

        vec4 r0 = stroke_row(i, 0);          /* pos.xyz, r_eff */
        float r = r0.w;
        if (r >= best_r) continue;           /* cannot win: skip the rest */

        vec3 d = v_pos - r0.xyz;
        if (dot(d, d) >= r * r) continue;    /* sphere */

        vec4 r1 = stroke_row(i, 1);          /* T.xyz, luminance */
        vec4 r2 = stroke_row(i, 2);          /* normal.xyz, brush index */
        vec4 r3 = stroke_row(i, 3);          /* lu_lo, lu_hi, lv_lo, lv_hi */
        vec3 T = r1.xyz;
        vec3 SN = r2.xyz;
        vec3 B = cross(SN, T);
        float lu = dot(d, T);
        float lv = dot(d, B);
        if (lu <= r3.x || lu >= r3.y || lv <= r3.z || lv >= r3.w) continue;

        float two_h = 2.0 * r * 0.7071067811865476;   /* 1/sqrt(2) */
        vec2 uv = vec2(lu / two_h + 0.5, lv / two_h + 0.5);
        int bi = int(r2.w);
        vec2 tile = vec2(float(bi % cols), float(bi / cols));
        vec2 auv = (tile + vec2(uv.x, 1.0 - uv.y)) / float(cols);
        if (texture(u_brush, auv).r <= 0.5) continue;

        if (dot(n, SN) <= cos_cut) continue;          /* crease guard */

        best_r = r; best_n = SN; best_l = r1.w; found = true;
    }

    vec3 shade_n = found ? normalize(best_n) : n;
    if (mode == 1) {
        fragColor = vec4(shade_n * 0.5 + 0.5, 1.0);
        return;
    }
    float lit = clamp(dot(shade_n, vec3(0.0, 0.0, 1.0)) * 0.5 + 0.5, 0.0, 1.0);
    float tone = found ? mix(0.85, 1.0, best_l) : 1.0;
    fragColor = vec4(vec3(lit * tone), 1.0);
}
"""

FRAG = FRAG.replace("STROKE_TILE_W", str(STROKE_TILE_W))


def make_shader():
    """Build the shader the 5.0 way, declaring the interface on the info object."""
    import gpu
    iface = gpu.types.GPUStageInterfaceInfo("autostroke_live_iface")
    iface.smooth('VEC3', "v_pos")
    iface.smooth('VEC3', "v_nrm")

    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC3', "nrm")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('VEC4', "u_grid_lo")
    info.push_constant('VEC4', "u_grid_inv")
    info.push_constant('IVEC4', "u_dim_n")
    info.push_constant('IVEC4', "u_misc")
    info.sampler(0, 'FLOAT_2D', "u_stroke")
    info.sampler(1, 'FLOAT_2D', "u_cell")
    info.sampler(2, 'FLOAT_2D', "u_list")
    info.sampler(3, 'FLOAT_2D', "u_brush")
    info.vertex_source(VERT)
    info.fragment_source(FRAG)
    return gpu.shader.create_from_info(info)




# ---------------------------------------------------------------------------
# Packing: strokes and the spatial grid, built in numpy and uploaded once
# ---------------------------------------------------------------------------

def pack_strokes(seeds, tan, masks, cfg):
    """(N, 4, 4) float32: four RGBA rows per stroke, matching the shader's stroke_row().

    Row 0  position.xyz, effective radius
    Row 1  T.xyz, per-stroke luminance
    Row 2  stroke normal.xyz, brush index
    Row 3  lu_lo, lu_hi, lv_lo, lv_hi   (the painted box, in stroke-local units)

    Built with the same helpers the baker uses -- brush_index, mask_bbox,
    frame_from_tangent -- so the preview's placement follows the bake's rather than
    drifting from it.
    """
    sid = seeds["id"].astype(np.int64)
    pos = seeds["position"].astype(np.float64)
    nrm = seeds["normal"].astype(np.float64)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    brush = baker.brush_index(sid, len(masks))
    lum = baker.hash01(sid).astype(np.float64)

    T, B = geometry.frame_from_tangent(nrm, tan if tan is not None else nrm)
    align = (np.radians([baker.mask_long_axis_deg(m, cfg) for m in masks])[brush]
             if cfg.mask_auto_align else 0.0)
    ang = np.full(len(sid), np.radians(cfg.stamp_rotate_deg), np.float64) - align
    jitter = float(np.clip(cfg.rot_jitter_deg, 0.0, 45.0))
    if jitter:
        ang += (baker.hash01(sid ^ np.int64(0x9e3779b9)) * 2.0 - 1.0) * np.radians(jitter)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    T = T * ca + B * sa

    size_var = float(np.clip(cfg.size_random, 0.0, 1.0))
    r = seeds["radius"].astype(np.float64) * cfg.mask_scale
    if size_var:
        r = r * 2.0 ** ((baker.hash01(sid ^ np.int64(baker.SALT_SIZE)) * 2.0 - 1.0) * size_var)

    bb = np.array([baker.mask_bbox(m, cfg) for m in masks], np.float64)[brush]
    two_h = 2.0 * r * INV_SQRT2
    out = np.zeros((len(sid), 4, 4), np.float32)
    out[:, 0, :3] = pos
    out[:, 0, 3] = r
    out[:, 1, :3] = T
    out[:, 1, 3] = lum
    out[:, 2, :3] = nrm
    out[:, 2, 3] = brush
    out[:, 3, 0] = (bb[:, 0] - 0.5) * two_h
    out[:, 3, 1] = (bb[:, 1] - 0.5) * two_h
    out[:, 3, 2] = (bb[:, 2] - 0.5) * two_h
    out[:, 3, 3] = (bb[:, 3] - 0.5) * two_h
    return out, pos, r, T, nrm


def build_grid(pos, r, T, nrm, bb_local, max_dim=48):
    """Bin strokes into a uniform grid, by the SAME box the baker rejects tiles with.

    A stroke goes into every cell its box overlaps, so a fragment only reads its own cell:
    that cell already holds every stroke that could possibly reach it. Querying a 3x3x3
    neighbourhood would be the alternative and is 27x the shader work for no gain.
    """
    B = np.cross(nrm, T)
    ctr = (pos + T * (0.5 * (bb_local[:, 0] + bb_local[:, 1]))[:, None]
           + B * (0.5 * (bb_local[:, 2] + bb_local[:, 3]))[:, None])
    ext = (np.abs(T) * (0.5 * (bb_local[:, 1] - bb_local[:, 0]))[:, None]
           + np.abs(B) * (0.5 * (bb_local[:, 3] - bb_local[:, 2]))[:, None]
           + np.abs(nrm) * r[:, None])
    lo, hi = ctr - ext, ctr + ext

    gmin = lo.min(0) - 1e-5
    gmax = hi.max(0) + 1e-5
    span = np.maximum(gmax - gmin, 1e-6)
    # cell no smaller than the biggest stroke, or one stroke lands in thousands of cells
    cell = np.maximum(2.0 * ext.max(0), span / max_dim)
    dim = np.maximum(np.ceil(span / cell).astype(np.int64), 1)
    dim = np.minimum(dim, max_dim)
    cell = span / dim

    i0 = np.clip(((lo - gmin) / cell).astype(np.int64), 0, dim - 1)
    i1 = np.clip(((hi - gmin) / cell).astype(np.int64), 0, dim - 1)

    # Expand stroke -> cells without a Python loop. Looping took 262 ms at 20,000
    # strokes, which is what a slider drag would have paid on every tick.
    span = i1 - i0 + 1                       # cells this stroke covers, per axis
    per = span.prod(1)                       # ...and in total
    n_pairs = int(per.sum())
    sidx = np.repeat(np.arange(len(pos)), per)
    # index of each pair WITHIN its own stroke's block, then unflattened to x/y/z
    off = np.arange(n_pairs) - np.repeat(np.cumsum(per) - per, per)
    sx = span[sidx]
    cx = i0[sidx, 0] + off % sx[:, 0]
    cy = i0[sidx, 1] + (off // sx[:, 0]) % sx[:, 1]
    cz = i0[sidx, 2] + off // (sx[:, 0] * sx[:, 1])
    cid = cx + dim[0] * (cy + dim[1] * cz)
    order = np.argsort(cid, kind="stable")
    pairs = np.stack([cid[order], sidx[order]], 1)
    n_cells = int(dim.prod())
    counts = np.bincount(pairs[:, 0], minlength=n_cells)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (gmin.astype(np.float32), (1.0 / cell).astype(np.float32),
            dim.astype(np.int32), starts.astype(np.float32),
            counts.astype(np.float32), pairs[:, 1].astype(np.float32))


def pack_brush_atlas(masks):
    """The brush set as one square atlas, cols x cols tiles. One texture, one sampler."""
    cols = int(np.ceil(np.sqrt(len(masks))))
    tile = max(m.shape[0] for m in masks)
    tile = min(tile, 512)                     # the silhouette is all the preview needs
    atlas = np.zeros((cols * tile, cols * tile), np.float32)
    for i, m in enumerate(masks):
        if m.shape[0] != tile:
            yi = (np.arange(tile) * (m.shape[0] / tile)).astype(np.int64)
            xi = (np.arange(tile) * (m.shape[1] / tile)).astype(np.int64)
            m = m[np.clip(yi, 0, m.shape[0] - 1)][:, np.clip(xi, 0, m.shape[1] - 1)]
        r, c = divmod(i, cols)
        atlas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = m
    return atlas, cols


# ---------------------------------------------------------------------------
# GPU upload and the draw handler.
#
# Everything below needs a running Blender with a GPU, so none of it is covered by the
# test suite -- unlike the packing above, which is. It is written defensively for that
# reason: any failure disables the preview and reports a message rather than raising
# inside a draw callback, where an exception can spam the console every redraw.
# ---------------------------------------------------------------------------

def _flat_texture(values, width=4096):
    """A 1-D float array as an R32F 2-D texture, since 1-D textures are not portable."""
    import gpu
    n = max(len(values), 1)
    h = int(np.ceil(n / float(width)))
    buf = np.zeros(width * h, np.float32)
    buf[:len(values)] = values
    tex = gpu.types.GPUTexture((width, h), format='R32F',
                               data=gpu.types.Buffer('FLOAT', width * h, buf.ravel()))
    return tex, width


def stroke_tile_buffer(packed, width=STROKE_TILE_W):
    """Pure-numpy half of _pack_stroke_texture: the (4*tile_h, width, 4) array a
    width-capped RGBA32F texture needs, with stroke i's row r placed at
    (col=i % width, row=(i // width) * 4 + r) -- matching the shader's stroke_row().
    Split out from the GPU upload so this indexing can be tested without bpy/gpu, the
    same way pack_strokes/build_grid are.
    """
    n = max(len(packed), 1)
    tile_h = int(np.ceil(n / float(width)))
    out = np.zeros((4 * tile_h, width, 4), np.float32)
    n_real = len(packed)
    col = np.arange(n_real) % width
    base_row = (np.arange(n_real) // width) * 4
    for r in range(4):
        out[base_row + r, col] = packed[:, r]
    return out


def _pack_stroke_texture(packed, width=STROKE_TILE_W):
    """(n,4,4) stroke rows -> a width-capped RGBA32F texture, tiled 4 rows per stroke.

    A texture width equal to the raw stroke count -- what this used to be -- exceeds
    MTLTextureDescriptor's max width (16384) past 16384 strokes, well inside
    PREVIEW_BUDGET's 60,000, and crashes the whole process with a native Metal
    assertion: not a Python exception, so nothing here could have caught or reported
    it. Tiling into a fixed-width grid instead, the same fix list_tex already needed
    via _flat_texture, removes the ceiling instead of trying to guess a safe one.
    """
    import gpu
    out = stroke_tile_buffer(packed, width)
    h = out.shape[0]
    tex = gpu.types.GPUTexture(
        (width, h), format='RGBA32F',
        data=gpu.types.Buffer('FLOAT', width * h * 4, out.ravel()))
    return tex


def _build(context, obj):
    """Everything the shader needs, from the same seeds the bake would use."""
    import gpu
    from .ops.bake import config_from_settings, load_brush, mesh_signature

    st = context.scene.autostroke
    from .core.config import Config
    masks, _set = load_brush(st, Config())
    cfg = config_from_settings(st, masks)

    sig = mesh_signature(obj)
    ckey = seed_cache.key(sig, cfg.target_strokes, cfg.min_strokes, cfg.max_strokes,
                          cfg.aspect_alpha)
    hit = seed_cache.get(ckey)
    if hit is not None:
        seeds, _stats, tan, _curv = hit
    else:
        seeds, stats = seed_bridge.build_seeds(
            obj, cfg.target_strokes, min_strokes=cfg.min_strokes,
            max_strokes=cfg.max_strokes, alpha=cfg.aspect_alpha,
            budget=seed_bridge.PREVIEW_BUDGET)
        tan = None
        curv = None
        if cfg.direction_source == "curvature":
            tan, curv = geometry.compute_direction_field(
                seeds["position"].astype(np.float32), seeds["normal"].astype(np.float32), cfg)
        seed_cache.put(ckey, seeds, stats, tan, curv)

    packed, pos, r, T, nrm = pack_strokes(seeds, tan, masks, cfg)
    bbl = packed[:, 3, :].astype(np.float64)
    gmin, ginv, dim, starts, counts, lst = build_grid(pos, r, T, nrm, bbl)

    n = len(pos)
    stroke_tex = _pack_stroke_texture(packed)
    cell_w = 4096
    cell_h = int(np.ceil(len(counts) / float(cell_w)))
    cbuf = np.zeros((cell_w * cell_h, 2), np.float32)
    cbuf[:len(counts), 0] = starts
    cbuf[:len(counts), 1] = counts
    cell_tex = gpu.types.GPUTexture(
        (cell_w, cell_h), format='RG32F',
        data=gpu.types.Buffer('FLOAT', cell_w * cell_h * 2, cbuf.ravel()))
    list_tex, list_w = _flat_texture(lst)

    atlas, cols = pack_brush_atlas(masks)
    brush_tex = gpu.types.GPUTexture(
        atlas.shape, format='R32F',
        data=gpu.types.Buffer('FLOAT', atlas.size, np.ascontiguousarray(atlas).ravel()))

    me = obj.evaluated_get(context.evaluated_depsgraph_get()).to_mesh()
    try:
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        co = np.empty(nv * 3, np.float32); me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        tri_v = np.empty(nt * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tri_v)
        tri_v = tri_v.reshape(-1, 3)

        # SHADING normals, per loop corner -- the same source bridge/mesh.py hands the
        # baker. Per-VERTEX normals are always smooth-averaged, so across a sharp edge,
        # custom split normals or Auto Smooth they disagree with what the bake sees. This
        # normal is the crease test's input (dot(n, SN) > cos_cut), so using the smoothed
        # one made the preview stop strokes in different places than the bake, everywhere
        # the mesh has a hard edge.
        #
        # One vertex can carry several corner normals, so the buffer cannot stay indexed
        # by vertex: expand to three unshared corners per triangle. That costs 3x the
        # vertex memory (~7 MB at 100k triangles) and is why the batch is unindexed.
        try:
            corner_n = np.empty(len(me.loops) * 3, np.float32)
            me.corner_normals.foreach_get("vector", corner_n)
            tri_l = np.empty(nt * 3, np.int32)
            me.loop_triangles.foreach_get("loops", tri_l)
            vnrm = corner_n.reshape(-1, 3)[tri_l.reshape(-1, 3)].reshape(-1, 3)
        except (AttributeError, RuntimeError, ValueError):
            vn = np.empty(nv * 3, np.float32)        # older build: smoothed, as before
            me.vertices.foreach_get("normal", vn)
            vnrm = vn.reshape(-1, 3)[tri_v].reshape(-1, 3)
        vpos = co[tri_v].reshape(-1, 3)
    finally:
        obj.evaluated_get(context.evaluated_depsgraph_get()).to_mesh_clear()

    from gpu_extras.batch import batch_for_shader
    shader = make_shader()
    batch = batch_for_shader(shader, 'TRIS', {"pos": vpos, "nrm": vnrm})
    # 0.1% of the bounding-box diagonal: enough to win the depth test, far too small to
    # show as the silhouette creeping outwards.
    diag = float(np.linalg.norm(vpos.max(0) - vpos.min(0)))
    depth_offset = max(diag, 1e-6) * 0.001

    return dict(shader=shader, batch=batch, obj=obj, n=n, depth_offset=depth_offset,
                stroke=stroke_tex, cell=cell_tex, lst=list_tex, brush=brush_tex,
                grid_lo=gmin, grid_inv=ginv, grid_dim=dim,
                cell_w=cell_w, list_w=list_w, cols=cols)


def _draw():
    global _error
    if _state is None:
        return
    try:
        import gpu
        s = _state
        obj = s["obj"]
        if obj is None or obj.name not in bpy.data.objects:
            return
        st = bpy.context.scene.autostroke
        mvp = bpy.context.region_data.perspective_matrix @ obj.matrix_world
        sh = s["shader"]
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        gpu.state.face_culling_set('BACK')
        sh.bind()
        sh.uniform_float("u_mvp", mvp)
        sh.uniform_sampler("u_stroke", s["stroke"])
        sh.uniform_sampler("u_cell", s["cell"])
        sh.uniform_sampler("u_list", s["lst"])
        sh.uniform_sampler("u_brush", s["brush"])
        # packed to stay inside the 128-byte push-constant budget; see make_shader()
        lo = s["grid_lo"]
        inv = s["grid_inv"]
        dim = s["grid_dim"]
        # read live from the scene: Stroke Cutoff is one uniform, so it costs nothing to
        # change and should not go through a rebuild
        cos_cut = float(np.cos(st.crease_angle))     # the prop is already radians
        sh.uniform_float("u_grid_lo",
                         (float(lo[0]), float(lo[1]), float(lo[2]), cos_cut))
        sh.uniform_float("u_grid_inv",
                         (float(inv[0]), float(inv[1]), float(inv[2]),
                          s["depth_offset"]))
        sh.uniform_int("u_dim_n",
                       (int(dim[0]), int(dim[1]), int(dim[2]), int(s["n"])))
        sh.uniform_int("u_misc",
                       (int(s["list_w"]), int(s["cell_w"]), int(s["cols"]),
                        int(st.live_mode)))
        s["batch"].draw(sh)
        gpu.state.depth_mask_set(False)
        gpu.state.face_culling_set('NONE')
    except Exception:                       # never raise inside a draw callback
        _error = traceback.format_exc().strip().splitlines()[-1]
        disable()


def enabled():
    return _handle is not None


def last_error():
    return _error


# ---- auto refresh ---------------------------------------------------------
# A slider drag fires its update callback on every tick. Rebuilding on each one would be
# ~10 ms of numpy plus four texture uploads, so changes are coalesced: the first one arms
# a timer and later ones are absorbed until it fires.
_pending = False
_DEBOUNCE = 0.12


def _flush():
    global _pending
    _pending = False
    if enabled():
        rebuild(bpy.context)
        _redraw()
    return None                 # one shot; returning a number would repeat it


def request_rebuild():
    """Called from property update callbacks. Cheap, and safe to call in a tight loop."""
    global _pending
    if not enabled() or _pending:
        return
    _pending = True
    try:
        bpy.app.timers.register(_flush, first_interval=_DEBOUNCE)
    except Exception:
        _pending = False        # no timer available: fall back to the Refresh button


def redraw_only():
    """For settings the shader reads live -- Stroke Cutoff, the view mode. No rebuild."""
    if enabled():
        _redraw()


def rebuild(context):
    """Re-upload after a change. Cheap for look dials; seeds come from the cache."""
    global _state, _error
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        disable()
        return False
    try:
        _state = _build(context, obj)
        _error = ""
        return True
    except mesh_bridge.MeshError as e:
        # Settings this one object cannot satisfy -- too many strokes, no UV map. The
        # PREVIOUS state is kept and the preview left ON: an artist who drags a slider
        # too far sees the reason and drags back, rather than having to notice the
        # preview switched itself off and turn it on again.
        _error = str(e)
        _redraw()
        return False
    except Exception:
        _error = traceback.format_exc().strip().splitlines()[-1]
        print("AutoStroke live preview failed:\n" + traceback.format_exc())
        disable()
        return False


def enable(context):
    global _handle
    if _handle is not None:
        return True
    if not rebuild(context):
        return False
    _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    _redraw()
    return True


def disable():
    global _handle, _state
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    _state = None
    _redraw()


def _redraw():
    for w in getattr(bpy.context, "window_manager", None).windows if bpy.context else []:
        for area in w.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# ---------------------------------------------------------------------------
# Operators and registration
# ---------------------------------------------------------------------------

class AUTOSTROKE_OT_live(bpy.types.Operator):
    bl_idname = "autostroke.live"
    bl_label = "Live Preview"
    bl_description = ("Draw the strokes straight into the viewport while you tweak. Not a "
                      "bake: it shades the object directly, so it ignores your material "
                      "and its lighting. Press Bake when you are happy")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        if enabled():
            disable()
            self.report({'INFO'}, "AutoStroke: live preview off")
            return {'FINISHED'}
        if not enable(context):
            self.report({'ERROR'}, "Live preview failed: %s" % (last_error() or "unknown"))
            return {'CANCELLED'}
        self.report({'INFO'}, "AutoStroke: live preview on")
        return {'FINISHED'}


class AUTOSTROKE_OT_live_refresh(bpy.types.Operator):
    bl_idname = "autostroke.live_refresh"
    bl_label = "Refresh Preview"
    bl_description = "Re-upload the strokes after changing a setting"

    @classmethod
    def poll(cls, context):
        return enabled()

    def execute(self, context):
        if not rebuild(context):
            self.report({'ERROR'}, "Live preview failed: %s" % (last_error() or "unknown"))
            return {'CANCELLED'}
        _redraw()
        return {'FINISHED'}


classes = (AUTOSTROKE_OT_live, AUTOSTROKE_OT_live_refresh)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    disable()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
