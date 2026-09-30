"""Live viewport preview, drawn on the GPU.

Why this is a separate renderer rather than a faster bake: the bake resolves a texture, and
even at its fastest that is ~1-3 s for a full-resolution map. Dropping resolution to buy
latency was tried and is the wrong axis -- the output is a pointer table inspected at full
size, so coarse texels degrade exactly what the artist is judging.

So this does not produce a map at all. It draws the object in the viewport with a fragment
shader that finds each pixel's winning stroke directly. That skips the texture entirely,
and with it the readback trap: pulling a rendered map back through img.pixels.foreach_set
costs ~0.3 s of Python for 4M floats, which on its own would make "live" impossible.

    fragment -> object-space position + normal + its triangle (varyings, no baked maps)
             -> the strokes that can reach that triangle across the surface (core/reach.py)
             -> loop them, run the same tests resolve_uv runs
             -> keep the SMALLEST passing stroke, shade with its normal

The search itself is shaders.SEARCH, shared verbatim with the GPU bake, and the candidate
lists come from the same cached reach tables the bake uses -- so the preview shows what
the bake produces.
"""

import traceback

import numpy as np
import bpy

from .bridge import cache as seed_cache, mesh as mesh_bridge, seeds as seed_bridge
from .core import baker, geometry, reach
from .core.config import INV_SQRT2
from .shaders import SEARCH, STROKE_ROWS, STROKE_TILE_W

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
#   u_grid_lo  = (unused .xyz, cos of the Stroke Cutoff)
#   u_grid_inv = (unused .xyz, depth offset along the normal)
#   u_dim_n    = (triangle count, unused, unused, stroke count)
#   u_misc     = (list width, cell width, brush atlas columns, view mode)
#
# The names predate per-triangle candidate lists, when .xyz carried a 3D grid; kept so the
# shared search reads the same uniforms in both shaders.
#
# That totals exactly 128: MAT4 64 + two VEC4 32 + two IVEC4 32. If a backend rejects it
# for size, the fix is to drop u_misc -- the texture widths are constants we choose, so
# they can be hardcoded in the GLSL, and columns/mode fit in the spare .w slots above.

VERT = """
void main()
{
    v_pos = pos;                       /* OBJECT space: strokes live here too */
    v_nrm = nrm;
    v_tri = tri;                       /* flat: the loop-triangle this corner belongs to */
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

# The stroke search itself lives in shaders.py, shared verbatim with the GPU bake: the
# preview is required to show what the bake will produce, and one copy of a rule cannot
# drift from itself. Only what is specific to DRAWING stays here -- the occupancy debug
# view and the headlight shading of the winner.
FRAG = SEARCH + """
void main()
{
    vec3 n = normalize(v_nrm);
    int mode = u_misc.w;

    if (mode == 2) {                        /* occupancy, for debugging the binning */
        float f = clamp(float(tri_range(int(v_tri + 0.5)).y) / 32.0, 0.0, 1.0);
        fragColor = vec4(f, 1.0 - f, 0.0, 1.0);
        return;
    }

    int w = find_stroke(v_pos, v_nrm, int(v_tri + 0.5));
    bool found = w >= 0;
    vec3 shade_n = found ? normalize(stroke_row(w, 2).xyz) : n;
    if (mode == 1) {
        fragColor = vec4(shade_n * 0.5 + 0.5, 1.0);
        return;
    }
    float lit = clamp(dot(shade_n, vec3(0.0, 0.0, 1.0)) * 0.5 + 0.5, 0.0, 1.0);
    float tone = found ? mix(0.85, 1.0, stroke_row(w, 1).w) : 1.0;
    fragColor = vec4(vec3(lit * tone), 1.0);
}
"""


def make_shader():
    """Build the shader the 5.0 way, declaring the interface on the info object."""
    import gpu
    iface = gpu.types.GPUStageInterfaceInfo("autostroke_live_iface")
    iface.smooth('VEC3', "v_pos")
    iface.smooth('VEC3', "v_nrm")
    iface.flat('FLOAT', "v_tri")

    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC3', "nrm")
    info.vertex_in(2, 'FLOAT', "tri")
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
# Packing: strokes and their candidate lists, built in numpy and uploaded once
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


def pack_brush_atlas(masks, max_tile=512):
    """The brush set as one square atlas, cols x cols tiles. One texture, one sampler.

    `max_tile` caps each brush's tile: 512 is plenty for the preview's silhouette, while
    the GPU bake passes the full brush resolution so its mask test samples the same
    pixels the CPU bake does."""
    cols = int(np.ceil(np.sqrt(len(masks))))
    tile = max(m.shape[0] for m in masks)
    tile = min(tile, max_tile)
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
    same way pack_strokes is.
    """
    n = max(len(packed), 1)
    tile_h = int(np.ceil(n / float(width)))
    out = np.zeros((STROKE_ROWS * tile_h, width, 4), np.float32)
    n_real = len(packed)
    col = np.arange(n_real) % width
    base_row = (np.arange(n_real) // width) * STROKE_ROWS
    for r in range(STROKE_ROWS):
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


def upload_strokes(masks, cfg, seeds, tan, rs, atlas_tile=512):
    """The four read-only tables the stroke search reads, uploaded to the GPU.

    Shared by the live preview and the GPU bake so both search IDENTICAL data: the stroke
    table (STROKE_ROWS RGBA rows per stroke), per TRIANGLE the (start, count) of the strokes
    that can reach it, the flat list of those stroke indices, and the brush atlas. `rs` is
    core/reach.build()'s result (via bridge/cache.reach_for). Every table is a fixed-width
    grid (see shaders.STROKE_TILE_W) so no dimension grows with the mesh or the stroke
    count past what Metal accepts.

    `atlas_tile` caps each brush's atlas tile: the preview keeps it small (it re-uploads on
    every slider tick); the bake passes full-resolution brushes.
    """
    import gpu
    packed, pos, r, T, nrm = pack_strokes(seeds, tan, masks, cfg)

    stroke_tex = _pack_stroke_texture(packed)
    n_tris = int(rs["n_tris"])
    cell_w = 4096
    cell_h = int(np.ceil(max(n_tris, 1) / float(cell_w)))
    cbuf = np.zeros((cell_w * cell_h, 2), np.float32)
    cbuf[:n_tris, 0] = rs["tri_start"]
    cbuf[:n_tris, 1] = rs["tri_count"]
    cell_tex = gpu.types.GPUTexture(
        (cell_w, cell_h), format='RG32F',
        data=gpu.types.Buffer('FLOAT', cell_w * cell_h * 2, cbuf.ravel()))
    list_tex, list_w = _flat_texture(rs["tri_strokes"].astype(np.float32))

    atlas, cols = pack_brush_atlas(masks, max_tile=atlas_tile)
    brush_tex = gpu.types.GPUTexture(
        atlas.shape, format='R32F',
        data=gpu.types.Buffer('FLOAT', atlas.size, np.ascontiguousarray(atlas).ravel()))

    return dict(stroke=stroke_tex, cell=cell_tex, lst=list_tex, brush=brush_tex,
                n=len(pos), n_tris=n_tris, cell_w=cell_w, list_w=list_w, cols=cols)


def bind_stroke_tables(shader, up):
    """Bind upload_strokes()'s tables to a shader that includes shaders.SEARCH. Uniforms
    that differ per caller (cos_cut, the .w slots) are set by the caller afterwards."""
    shader.uniform_sampler("u_stroke", up["stroke"])
    shader.uniform_sampler("u_cell", up["cell"])
    shader.uniform_sampler("u_list", up["lst"])
    shader.uniform_sampler("u_brush", up["brush"])


def _build(context, obj):
    """Everything the shader needs, from the same seeds the bake would use."""
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

    rs = seed_cache.reach_for(ckey, cfg, lambda: reach.build_for_seeds(seeds, cfg))
    up = upload_strokes(masks, cfg, seeds, tan, rs)

    # The mesh as the bake sees it (render settings), the same one the seeds and the reach
    # tables were built from -- see mesh_bridge.render_state.
    with mesh_bridge.render_state(obj):
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
            # Each corner carries its loop-triangle index, so a fragment knows which triangle
            # it is on -- the key into the reach lists. Same loop_triangles order as the seeds'
            # triangles (bridge/mesh.read_triangles), so the index means the same thing.
            vtri = np.repeat(np.arange(nt, dtype=np.float32), 3)
        finally:
            obj.evaluated_get(context.evaluated_depsgraph_get()).to_mesh_clear()

    from gpu_extras.batch import batch_for_shader
    shader = make_shader()
    if nt != up["n_tris"]:
        raise mesh_bridge.MeshError(
            "%s changed since its strokes were placed -- press Refresh" % obj.name)
    batch = batch_for_shader(shader, 'TRIS', {"pos": vpos, "nrm": vnrm, "tri": vtri})
    # 0.1% of the bounding-box diagonal: enough to win the depth test, far too small to
    # show as the silhouette creeping outwards.
    diag = float(np.linalg.norm(vpos.max(0) - vpos.min(0)))
    depth_offset = max(diag, 1e-6) * 0.001

    return dict(up, shader=shader, batch=batch, obj=obj, depth_offset=depth_offset)


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
        bind_stroke_tables(sh, s)
        # packed to stay inside the 128-byte push-constant budget; see make_shader()
        # read live from the scene: Stroke Cutoff is one uniform, so it costs nothing to
        # change and should not go through a rebuild
        cos_cut = float(np.cos(st.crease_angle))     # the prop is already radians
        sh.uniform_float("u_grid_lo", (0.0, 0.0, 0.0, cos_cut))
        sh.uniform_float("u_grid_inv", (0.0, 0.0, 0.0, s["depth_offset"]))
        sh.uniform_int("u_dim_n", (int(s["n_tris"]), 0, 0, int(s["n"])))
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
    except (mesh_bridge.MeshError, reach.ReachError) as e:
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
