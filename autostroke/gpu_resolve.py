"""The texel resolve on the GPU, as a compute shader -- a drop-in for baker.resolve_uv.

Why it exists: the live preview already runs the placement rule on the GPU, per pixel. The
bake ran a separate CPU implementation of the same rule, so what an artist judged in the
viewport was not literally what they shipped. Here the bake runs the preview's own search
(shaders.SEARCH, included verbatim in both) over the texture's texels instead of the
screen's pixels, so the two cannot drift apart. The CPU resolver stays as the fallback.

Why a compute shader and not a draw: a bake draws nothing. It processes arrays -- per
texel, a 3D position and surface normal (the Cycles-baked maps the CPU path already uses)
in, one answer per texel out. Nothing is rasterized, so there is no culling, no UV-space
drawing and no framebuffer orientation to get wrong; both ends of every buffer are laid out
by this module.

What the shader returns is only the WINNING STROKE'S INDEX per texel (or -1). Everything a
winner contributes -- its UV pointer, its luminance, its normal -- is a per-stroke constant
(flat per cell), and baker.resolve_uv defines each one from the seed arrays. gather() looks
them up with those exact definitions, so the two resolvers can differ only in WHICH stroke
wins a texel, never in what a given winner writes. The brush shape itself is decided on the
GPU, inside the search: it is encoded in which texels receive index j.

Layout, and the limits it respects:
  - Blender's Python GPU API exposes textures and small uniform buffers, not storage
    buffers, so every array is a texture used as a 2D array: element i at
    (i % 4096, i / 4096).
  - Texels go through in chunks of up to 4096 x 1024 (~4.2M): at 8K a whole map's valid
    texels are ~54M, and positions + normals alone would be ~1.7 GB in one go. Chunks also
    give the modal operator a progress point, so the bar moves and Esc works.
  - Nothing is ever wider than 4096. Metal treats a texture past 16384 wide as a native
    assertion that kills Blender -- not an exception anything here could catch.

`gpu` and `bpy` are imported inside functions only, so the pure helpers below import and
test without Blender.
"""

import numpy as np

from . import shaders
from .core import baker

CHUNK_W = 4096
CHUNK_H = 1024
CHUNK = CHUNK_W * CHUNK_H

SENTINEL = -2.0
"""What every output slot holds before the dispatch. Every thread overwrites its own slot
with a winner index (>= 0) or -1, so a -2 surviving where a thread should have run means
the dispatch did not run -- a GPU that is 'bugging out' silently. Without this, a dead
dispatch could read back as zeros, i.e. 'stroke 0 wins everywhere', and ship a broken map."""

class GPUResolveError(RuntimeError):
    """The GPU produced something that cannot be trusted; the caller falls back to CPU."""


COMPUTE = shaders.SEARCH + ("""
void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= CHUNK_W) return;
    int i = p.x + p.y * CHUNK_W;
    if (i >= u_misc.w) return;               /* padding past this chunk: keep sentinel */
    vec4 pt = texelFetch(u_pts, p, 0);          /* position .xyz, triangle id .w */
    vec3 nrm = texelFetch(u_pnrm, p, 0).xyz;
    imageStore(u_out, p, vec4(float(find_stroke(pt.xyz, nrm, int(pt.w)))));
}
""".replace("CHUNK_W", str(CHUNK_W)))


# ---------------------------------------------------------------------------
# Pure numpy: layout, the sentinel check, the gather. Tested in tests/test_gpu_resolve.py.
# ---------------------------------------------------------------------------

def chunks(m, size=CHUNK):
    """[(a, b), ...] covering range(m) in order, each at most `size` long."""
    return [(a, min(a + size, m)) for a in range(0, m, size)]


def pack_chunk(xyz, w=None, width=CHUNK_W):
    """(m, 3) floats -> (h, width, 4) float32, element i at row i // width, col i % width.
    The fourth channel carries `w` when given (the texel's triangle id -- exact as a float
    below 2**24) and is zero otherwise; padding past m is zero and never read."""
    m = len(xyz)
    h = max(int(np.ceil(m / float(width))), 1)
    out = np.zeros((h * width, 4), np.float32)
    out[:m, :3] = xyz
    if w is not None:
        out[:m, 3] = w
    return out.reshape(h, width, 4)


def indices_from_readback(flat, m, n_strokes):
    """The first m readback values as int64 winner indices, or raise GPUResolveError.

    Rejects: a surviving sentinel (a thread never ran), anything non-finite, anything not
    a whole number, anything outside [-1, n_strokes)."""
    v = np.asarray(flat[:m], np.float64)
    if len(v) < m:
        raise GPUResolveError("GPU readback returned %d values, expected %d" % (len(v), m))
    if (v == SENTINEL).any():
        raise GPUResolveError("%d texels were never written by the GPU"
                              % int((v == SENTINEL).sum()))
    if not np.isfinite(v).all():
        raise GPUResolveError("GPU returned non-finite stroke indices")
    idx = np.rint(v).astype(np.int64)
    if (np.abs(v - idx) > 1e-3).any() or (idx < -1).any() or (idx >= n_strokes).any():
        raise GPUResolveError("GPU returned stroke indices outside [-1, %d)" % n_strokes)
    return idx


def gather(idx, uv_self, seed_uv, seed_id, seed_nrm, normalize_normals):
    """Winner index per texel -> resolve_uv's exact return tuple.

    Uses baker.resolve_uv's own definitions: a winner j writes seed_uv[j], luminance
    hash01(seed_id[j]), and its normal * 0.5 + 0.5 -- normalised only when the crease
    guard is on, because that is the only case in which resolve_uv normalises it. Texels
    no stroke paints keep their own UV, luminance 0.5 and a zero normal, exactly as there.
    """
    covered = idx >= 0
    w = idx[covered]
    uv_out = np.array(uv_self, copy=True)
    uv_out[covered] = seed_uv[w]
    lum_out = np.full(len(idx), 0.5, np.float32)
    lum_out[covered] = baker.hash01(seed_id).astype(np.float32)[w]
    nrm = seed_nrm
    if normalize_normals:
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    dbg_out = np.zeros((len(idx), 3), np.float32)
    dbg_out[covered] = nrm[w] * 0.5 + 0.5
    return uv_out, dbg_out, covered, lum_out


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def make_compute_shader():
    """The search as a compute kernel: same push constants and samplers as the preview
    (so shaders.SEARCH reads the same names), plus the chunk's positions and normals in,
    one R32F image of winner indices out."""
    import gpu
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('VEC4', "u_grid_lo")      # .w: cos(Stroke Cutoff)
    info.push_constant('VEC4', "u_grid_inv")     # unused here (preview: depth offset)
    info.push_constant('IVEC4', "u_dim_n")       # triangle count, -, -, stroke count
    info.push_constant('IVEC4', "u_misc")        # list width, cell width, atlas cols, n
    info.sampler(0, 'FLOAT_2D', "u_stroke")
    info.sampler(1, 'FLOAT_2D', "u_cell")
    info.sampler(2, 'FLOAT_2D', "u_list")
    info.sampler(3, 'FLOAT_2D', "u_brush")
    info.sampler(4, 'FLOAT_2D', "u_pts")
    info.sampler(5, 'FLOAT_2D', "u_pnrm")
    info.image(6, 'R32F', 'FLOAT_2D', "u_out", qualifiers={'WRITE'})
    info.local_group_size(16, 16)
    info.compute_source(COMPUTE)
    return gpu.shader.create_from_info(info)


def _texture(arr, fmt, channels):
    import gpu
    h, w = arr.shape[:2]
    return gpu.types.GPUTexture(
        (w, h), format=fmt,
        data=gpu.types.Buffer('FLOAT', w * h * channels, np.ascontiguousarray(arr).ravel()))


def resolve(points, uv_self, pt_nrm, seeds, seed_tan, masks, cfg, pt_tri, rs):
    """GENERATOR with baker.resolve_uv's contract: yields progress 0..1 after each chunk,
    returns (uv_out, dbg_out, covered, lum_out) via StopIteration.value.

    `points`, `uv_self`, `pt_nrm`, `pt_tri` are the VALID texels only, exactly as
    ops/bake.py passes them to resolve_uv; `rs` is core/reach.build()'s result, the same
    reach tables the CPU resolver filters with. Raises GPUResolveError (or whatever the GPU
    API raises) on any failure; the caller treats every exception as "fall back to the CPU
    resolver".
    """
    if len(pt_tri) != len(points):
        raise GPUResolveError("texel triangle ids do not match the texels")
    if rs["n_tris"] >= 1 << 24:
        raise GPUResolveError("more triangles than a float texel can index exactly")
    import gpu
    from . import livepreview
    from .bridge.brushes import BRUSH_MAX_PX

    masks = [masks] if isinstance(masks, np.ndarray) else list(masks)
    M = len(points)
    guard = bool(cfg.crease_guard) and pt_nrm is not None
    # guard off -> a cutoff no dot product can fall under, so the crease test never rejects
    cos_cut = float(np.cos(np.radians(cfg.crease_angle_deg))) if guard else -2.0
    nrm_in = pt_nrm if pt_nrm is not None else np.zeros((M, 3), np.float32)

    up = livepreview.upload_strokes(masks, cfg, seeds, seed_tan, rs,
                                    atlas_tile=BRUSH_MAX_PX)
    shader = make_compute_shader()

    idx = np.empty(M, np.int64)
    for a, b in chunks(M):
        m = b - a
        pts = pack_chunk(points[a:b], pt_tri[a:b])
        h = pts.shape[0]
        pts_tex = _texture(pts, 'RGBA32F', 4)
        nrm_tex = _texture(pack_chunk(nrm_in[a:b]), 'RGBA32F', 4)
        out_tex = _texture(np.full((h, CHUNK_W), SENTINEL, np.float32), 'R32F', 1)

        shader.bind()
        livepreview.bind_stroke_tables(shader, up)
        shader.uniform_sampler("u_pts", pts_tex)
        shader.uniform_sampler("u_pnrm", nrm_tex)
        shader.image("u_out", out_tex)
        shader.uniform_float("u_grid_lo", (0.0, 0.0, 0.0, cos_cut))
        shader.uniform_float("u_grid_inv", (0.0, 0.0, 0.0, 0.0))
        shader.uniform_int("u_dim_n", (int(up["n_tris"]), 0, 0, int(up["n"])))
        shader.uniform_int("u_misc", (int(up["list_w"]), int(up["cell_w"]), int(up["cols"]), m))
        gpu.compute.dispatch(shader, CHUNK_W // 16, (h + 15) // 16, 1)

        buf = out_tex.read()
        # A Buffer converted with np.array() is walked in the wrong axis order for a 2D
        # texture (measured on Metal: it came back transposed). Flattening its dimensions
        # first and reading the memory directly is exact -- and zero-copy.
        buf.dimensions = h * CHUNK_W
        flat = np.frombuffer(buf, dtype=np.float32, count=h * CHUNK_W)
        idx[a:b] = indices_from_readback(flat, m, int(up["n"]))
        yield b / float(M)

    return gather(idx, uv_self, seeds["UVMap"].astype(np.float32),
                  seeds["id"].astype(np.int64), seeds["normal"].astype(np.float32), guard)
