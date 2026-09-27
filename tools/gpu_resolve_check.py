"""GPU vs CPU texel resolve, measured on a real object in a real Blender.

Compute shaders need a GPU context, which `blender --background` does not have, so this
runs with a window. It opens briefly, measures, writes a report and exits -- it never saves
the .blend.

    Blender --factory-startup PainterlyTexture.blend --python tools/gpu_resolve_check.py \
            -- <report.txt> [resolution]

--factory-startup keeps any INSTALLED copy of the add-on from loading, so the repo copy can
register without its classes colliding with the installed ones.

What it reports:
  1. plumbing -- a trivial compute kernel writes each slot's own index; read back exactly?
     (Proves dispatch + imageStore + readback layout before any stroke logic is trusted.)
  2. parity -- baker.resolve_uv (CPU) and gpu_resolve.resolve (GPU) on the SAME Cycles-
     baked position/normal maps and the SAME seeds: covered-flag agreement, and among
     texels both cover, how often they picked the same winner (same UV pointer and same
     luminance -- a winner's two identifying per-stroke constants).
  3. timings for both resolvers.
  4. reachable surface (core/reach.py): how many texels the UNRESTRICTED search gave to a
     stroke on a different connected part of the mesh (the bug reach fixes), and the same
     count with reach on (must be 0), plus reach build time and strokes per triangle.

Object "__stacked__" builds, in memory only, two stacked 1 m planes 2 mm apart as one mesh
with separate UV islands -- the smallest case of the bug.
"""

import os
import sys
import time
import traceback

import bpy

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = ARGV[0] if ARGV else os.path.join(os.path.dirname(__file__), "gpu_resolve_report.txt")
RES = int(ARGV[1]) if len(ARGV) > 1 else 1024
OBJ = ARGV[2] if len(ARGV) > 2 else None          # optional: object name to test
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lines = []


def log(s=""):
    lines.append(str(s))
    print(s)


def plumbing():
    import numpy as np
    import gpu
    W, H = 4096, 64
    n = W * H
    info = gpu.types.GPUShaderCreateInfo()
    info.image(0, 'R32F', 'FLOAT_2D', "u_out", qualifiers={'WRITE'})
    info.push_constant('IVEC4', "u_dim")
    info.local_group_size(16, 16)
    info.compute_source("""
void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= u_dim.x || p.y >= u_dim.y) return;
    imageStore(u_out, p, vec4(float(p.x + p.y * u_dim.x)));
}
""")
    sh = gpu.shader.create_from_info(info)
    tex = gpu.types.GPUTexture((W, H), format='R32F',
                               data=gpu.types.Buffer('FLOAT', n, np.full(n, -2.0, np.float32)))
    sh.bind()
    sh.image("u_out", tex)
    sh.uniform_int("u_dim", (W, H, 0, 0))
    gpu.compute.dispatch(sh, W // 16, H // 16, 1)
    buf = tex.read()
    buf.dimensions = n
    got = np.frombuffer(buf, dtype=np.float32, count=n)
    ok = np.array_equal(got, np.arange(n, dtype=np.float32))
    log("1. plumbing (%s slots): %s" % ("{:,}".format(n), "OK" if ok else "WRONG"))
    return ok


def stacked_planes():
    """Two 1 m planes 2 mm apart, one mesh, each on its own half of the UV square."""
    n = 24
    verts, faces = [], []
    for layer, z in enumerate((0.0, 0.002)):
        base = len(verts)
        for j in range(n + 1):
            for i in range(n + 1):
                verts.append((i / n - 0.5, j / n - 0.5, z))
        for j in range(n):
            for i in range(n):
                a = base + j * (n + 1) + i
                faces.append((a, a + 1, a + n + 2, a + n + 1))
    me = bpy.data.meshes.new("__stacked__")
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            x, y, z = me.vertices[me.loops[li].vertex_index].co
            uv.data[li].uv = ((x + 0.5) * 0.48 + (0.5 if z > 0.001 else 0.01),
                              (y + 0.5) * 0.96 + 0.02)
    obj = bpy.data.objects.new("__stacked__", me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def components(adj, T):
    """Connected-component label per triangle, by min-label propagation."""
    import numpy as np
    start, nbr = adj
    lab = np.arange(T)
    has = np.diff(start) > 0
    while True:
        m = lab.copy()
        m[has] = np.minimum(lab[has], np.minimum.reduceat(lab[nbr], start[:-1][has]))
        m = m[m]                                  # pointer jumping
        if np.array_equal(m, lab):
            return lab
        lab = m


def pick_object():
    from autostroke.ops import setup as setup_ops
    if OBJ == "__stacked__":
        return stacked_planes()
    if OBJ:
        o = bpy.data.objects.get(OBJ)
        return o if o is not None and not setup_ops.validate(o) else None
    obj = bpy.context.view_layer.objects.active
    if obj is not None and not setup_ops.validate(obj):
        return obj
    for o in bpy.context.view_layer.objects:
        if not setup_ops.validate(o):
            return o
    return None


def parity():
    import numpy as np
    from autostroke import gpu_resolve
    from autostroke.core import baker, geometry, reach
    from autostroke.core.config import Config
    from autostroke.bridge import position as pos_bridge, seeds as seed_bridge
    from autostroke.ops.bake import config_from_settings, load_brush

    obj = pick_object()
    if obj is None:
        from autostroke.ops import setup as setup_ops
        ok = [o.name for o in bpy.data.objects if not setup_ops.validate(o)]
        log("2. parity: no valid mesh object%s; bakeable objects here: %s"
            % (" named %r" % OBJ if OBJ else "", ", ".join(ok) or "none"))
        return
    st = bpy.context.scene.autostroke
    masks, set_name = load_brush(st, Config())
    cfg = config_from_settings(st, masks)
    log("2. parity on '%s' at %d, brush set '%s' (%d brushes)"
        % (obj.name, RES, set_name, len(masks)))

    t = time.perf_counter()
    pos_img, _ = pos_bridge.bake(obj, RES, channel="position")
    nrm_img, _ = pos_bridge.bake(obj, RES, channel="normal")
    log("   cycles position + normal bake: %.2fs" % (time.perf_counter() - t))

    seeds, sstats = seed_bridge.build_seeds(
        obj, cfg.target_strokes, min_strokes=cfg.min_strokes,
        max_strokes=cfg.max_strokes, alpha=cfg.aspect_alpha)
    tan = None
    if cfg.direction_source == "curvature":
        tan, _ = geometry.compute_direction_field(
            seeds["position"].astype(np.float32), seeds["normal"].astype(np.float32), cfg)
    log("   strokes: %s" % "{:,}".format(len(seeds["id"])))

    # exactly the texel setup ops/bake.py does
    H, W, _ = pos_img.shape
    flat = pos_img.reshape(-1, 3)
    valid = (np.abs(flat) > cfg.bg_eps).any(1) & np.isfinite(flat).all(1)
    ys, xs = np.divmod(np.arange(H * W), W)
    uv_self = np.empty((H * W, 2), np.float32)
    uv_self[:, 0] = (xs + 0.5) / W
    uv_self[:, 1] = 1.0 - (ys + 0.5) / H
    flat_n = nrm_img.reshape(-1, 3)
    pts, uvs, pns = flat[valid], uv_self[valid], flat_n[valid]
    log("   valid texels: %s" % "{:,}".format(len(pts)))

    # exactly the reach setup ops/bake.py does
    m = seeds["_mesh"]
    t = time.perf_counter()
    tri_map = reach.raster_tri_ids(m["uvP"], m["uvQ"], m["uvR"], m["P"], m["Q"], m["R"],
                                   H, W, pos_map=flat, valid=valid,
                                   margin=pos_bridge.BAKE_MARGIN)
    t_raster = time.perf_counter() - t
    t = time.perf_counter()
    rs = reach.build_for_seeds(seeds, cfg)
    t_reach = time.perf_counter() - t
    tri = tri_map[valid]
    log("   texel -> triangle map: %.2fs  (%d valid texels without a triangle)"
        % (t_raster, int((tri < 0).sum())))
    log("   reach tables: %.2fs  (%s triangles, %s pairs, strokes per triangle mean %.1f, "
        "max %d)" % (t_reach, "{:,}".format(rs["n_tris"]), "{:,}".format(rs["pairs"]),
                     rs["tri_count"].mean(), rs["tri_count"].max()))

    def cpu_run(with_reach):
        kw = dict(pt_tri=tri, stroke_reach=(rs["stroke_start"], rs["stroke_tris"])) \
            if with_reach else {}
        return baker.resolve_uv_blocking(
            pts, uvs, seeds["position"].astype(np.float32), seeds["id"].astype(np.int64),
            seeds["UVMap"].astype(np.float32), seeds["radius"].astype(np.float32),
            seeds["normal"].astype(np.float32), None, masks, cfg, tan, cfg.stamp_rotate_deg,
            pt_nrm=pns, **kw)

    # which stroke won each texel, from its UV pointer + luminance (unique per stroke here)
    key_uv = seeds["UVMap"].astype(np.float32)
    lut = {(float(u), float(v)): i for i, (u, v) in enumerate(key_uv)}

    def winners(out):
        w = np.array([lut.get((float(a), float(b)), -1) for a, b in out[0]])
        return np.where(out[2], w, -1)

    comp = components(reach.tri_adjacency(reach.weld(m["P"], m["Q"], m["R"])), rs["n_tris"])

    def cross(out):
        w = winners(out)
        ok = (w >= 0) & (tri >= 0)
        return int((comp[seeds["_tri"][w[ok]]] != comp[tri[ok]]).sum()), int(ok.sum())

    t = time.perf_counter()
    old = cpu_run(False)
    t_old = time.perf_counter() - t
    t = time.perf_counter()
    cpu = cpu_run(True)
    t_cpu = time.perf_counter() - t
    log("4. texels won by a stroke on ANOTHER connected part: without reach %s of %s,"
        " with reach %s of %s" % (*("{:,}".format(x) for x in cross(old)),
                                  *("{:,}".format(x) for x in cross(cpu))))
    log("   CPU resolve without reach %.2fs, with reach %.2fs" % (t_old, t_cpu))

    t = time.perf_counter()
    gen = gpu_resolve.resolve(pts, uvs, pns, seeds, tan, masks, cfg, tri, rs)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        gpu_out = stop.value
    t_gpu = time.perf_counter() - t

    cu, cd, cc, cl = cpu
    gu, gd, gc, gl = gpu_out
    agree_cov = float((cc == gc).mean())
    both = cc & gc
    same = both & (cu == gu).all(1) & (cl == gl)
    log("   covered: CPU %.2f%%  GPU %.2f%%  same flag on %.3f%% of texels"
        % (100 * cc.mean(), 100 * gc.mean(), 100 * agree_cov))
    log("   same winner on %.3f%% of texels both cover (%s of %s)"
        % (100 * same.sum() / max(both.sum(), 1), "{:,}".format(int(same.sum())),
           "{:,}".format(int(both.sum()))))
    log("   winner-normal max diff where same winner: %.2e"
        % (float(np.abs(cd[same] - gd[same]).max()) if same.any() else 0.0))
    log("   GPU winners on another connected part: %s of %s"
        % tuple("{:,}".format(x) for x in cross(gpu_out)))
    log("3. timing: CPU resolve %.2fs   GPU resolve %.3fs   (x%.0f)"
        % (t_cpu, t_gpu, t_cpu / max(t_gpu, 1e-6)))


def run():
    try:
        sys.path.insert(0, REPO)
        import autostroke
        autostroke.register()
        log("backend: %s" % __import__("gpu").platform.backend_type_get())
        if plumbing():
            parity()
    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc())
    finally:
        with open(OUT, "w") as f:
            f.write("\n".join(lines) + "\n")
        os._exit(0)          # never prompt to save: this must not touch the .blend
    return None


bpy.app.timers.register(run, first_interval=1.0)
