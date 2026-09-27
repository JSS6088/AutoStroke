"""The GPU resolver's CPU-side half: layout, the sentinel check, and the gather.

gpu_resolve.resolve() runs the stroke search in a compute shader and reads back only each
texel's WINNING STROKE INDEX. Everything else -- the UV pointer, luminance and normal a
winner writes -- is filled in here, in numpy. So the GPU and CPU resolvers can only ever
disagree about WHICH stroke wins a texel, never about what a given winner writes, and only
if gather() reproduces baker.resolve_uv's definitions exactly. That is the main thing this
file proves: it runs the real CPU resolver, recovers its winners, and requires gather() to
rebuild resolve_uv's output byte for byte.

The GPU itself cannot run here (no GPU context outside Blender); tools/gpu_resolve_check.py
measures GPU/CPU parity in a real Blender instead.

Run directly: python3 autostroke/tests/test_gpu_resolve.py
"""

import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def stub_bpy():
    """Just enough bpy for `import autostroke.gpu_resolve` (the package __init__ imports
    the bpy-facing modules). Nothing here touches the stub at runtime."""
    class _Type:
        pass

    def _prop(*a, **kw):
        return ("prop", a, kw)

    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(**{n: type(n, (_Type,), {}) for n in (
        "PropertyGroup", "Operator", "Panel", "Scene", "Object")})
    bpy.props = types.SimpleNamespace(**{n: _prop for n in (
        "BoolProperty", "EnumProperty", "FloatProperty", "IntProperty",
        "PointerProperty", "StringProperty")})
    bpy.utils = types.SimpleNamespace(register_class=lambda c: None,
                                      unregister_class=lambda c: None)
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    bpy.data = types.SimpleNamespace(objects={}, images={}, materials={}, filepath="")
    bpy.context = None
    bpy.app = types.SimpleNamespace(background=True)
    for name, mod in (("bpy", bpy), ("bpy.types", bpy.types), ("bpy.props", bpy.props),
                      ("bpy.utils", bpy.utils)):
        sys.modules[name] = mod
    for name in ("gpu", "gpu_extras", "gpu_extras.batch", "bmesh", "mathutils"):
        sys.modules.setdefault(name, types.ModuleType(name))


def load():
    stub_bpy()
    sys.path.insert(0, os.path.dirname(ROOT))
    from autostroke import gpu_resolve, shaders          # noqa: E402
    from autostroke.core import baker                     # noqa: E402
    from autostroke.core.config import Config             # noqa: E402
    return gpu_resolve, shaders, baker, Config


def scene(n_pts=6000, n_seeds=60, seed=3):
    """Few strokes with a wide radius spread over a curved sheet, so texels are contested
    and the crease guard actually has something to reject."""
    rng = np.random.default_rng(seed)
    u, v = rng.random(n_pts) * 4.0, rng.random(n_pts) * 4.0
    pts = np.stack([u, v, 0.3 * np.sin(u) * np.cos(v)], 1).astype(np.float32)
    du = np.stack([np.ones_like(u), np.zeros_like(u), 0.3 * np.cos(u) * np.cos(v)], 1)
    dv = np.stack([np.zeros_like(u), np.ones_like(u), -0.3 * np.sin(u) * np.sin(v)], 1)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    su, sv = rng.random(n_seeds) * 4.0, rng.random(n_seeds) * 4.0
    pos = np.stack([su, sv, 0.3 * np.sin(su) * np.cos(sv)], 1).astype(np.float32)
    sn = np.stack([-0.3 * np.cos(su) * np.cos(sv), 0.3 * np.sin(su) * np.sin(sv),
                   np.ones_like(su)], 1)
    # deliberately NOT unit length: the gather must normalise exactly when resolve_uv does
    sn = (sn / np.linalg.norm(sn, axis=1, keepdims=True)
          * (1.0 + rng.random(n_seeds))[:, None]).astype(np.float32)
    return dict(pts=pts, pn=n.astype(np.float32), pos=pos, nrm=sn,
                uv=np.stack([np.arange(n_seeds) / n_seeds + 0.001,      # unique per seed
                             np.full(n_seeds, 0.5)], 1).astype(np.float32),
                rad=(0.04 + 0.30 * rng.random(n_seeds)).astype(np.float32),
                ids=np.arange(n_seeds, dtype=np.int64),
                uv_self=rng.random((n_pts, 2)).astype(np.float32))


def main():
    gpu_resolve, shaders, baker, Config = load()
    gather_matches_cpu(gpu_resolve, baker, Config)
    layout(gpu_resolve)
    readback(gpu_resolve)
    kernel_source(gpu_resolve, shaders)
    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


def gather_matches_cpu(gpu_resolve, baker, Config):
    print("\nGATHER REBUILDS resolve_uv's OUTPUT FROM ITS WINNERS, BYTE FOR BYTE")
    s = scene()
    mask = np.ones((64, 64), np.float32)
    for label, cfg in (("crease guard on", Config(crease_angle_deg=35.0)),
                       ("crease guard off", Config(crease_guard=False))):
        cpu = baker.resolve_uv_blocking(
            s["pts"], s["uv_self"], s["pos"], s["ids"], s["uv"], s["rad"], s["nrm"],
            None, mask, cfg, s["nrm"].astype(np.float64), 0.0, pt_nrm=s["pn"])
        # recover the CPU's winner per texel from its UV pointer (unique per seed)
        key = {round(float(u[0]), 6): i for i, u in enumerate(s["uv"])}
        idx = np.array([key.get(round(float(u[0]), 6), -1) for u in cpu[0]])
        idx = np.where(cpu[2], idx, -1)
        guard = bool(cfg.crease_guard)
        got = gpu_resolve.gather(idx, s["uv_self"], s["uv"], s["ids"], s["nrm"], guard)
        names = ("uv_out", "dbg_out", "covered", "lum_out")
        same = [a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
                for a, b in zip(cpu, got)]
        check("%s: all four outputs identical" % label, all(same),
              ", ".join(n for n, ok in zip(names, same) if not ok) or
              "%d of %d texels covered" % (int(cpu[2].sum()), len(idx)))

    # and the fixture is not vacuous: the normalise branch really changes the answer here
    s2 = scene()
    idx = np.zeros(len(s2["pts"]), np.int64)
    on = gpu_resolve.gather(idx, s2["uv_self"], s2["uv"], s2["ids"], s2["nrm"], True)[1]
    off = gpu_resolve.gather(idx, s2["uv_self"], s2["uv"], s2["ids"], s2["nrm"], False)[1]
    check("  (the normalise branch is exercised: outputs differ)",
          not np.array_equal(on, off))


def layout(gpu_resolve):
    print("\nCHUNKS AND TEXTURE LAYOUT")
    C = gpu_resolve.CHUNK
    for m in (0, 1, 4095, 4096, 4097, C - 1, C, C + 1, 3 * C + 17):
        ch = gpu_resolve.chunks(m)
        covered = sum(b - a for a, b in ch)
        contiguous = all(ch[k][1] == ch[k + 1][0] for k in range(len(ch) - 1))
        ok = (covered == m and contiguous and all(0 < b - a <= C for a, b in ch)
              and (not ch or (ch[0][0] == 0 and ch[-1][1] == m)))
        check("chunks(%d): every texel exactly once, none over %d" % (m, C), ok,
              "%d chunk(s)" % len(ch))
    check("8K worth of texels (~54M) stays within the chunk cap",
          all(b - a <= C for a, b in gpu_resolve.chunks(54_000_000)))

    W = gpu_resolve.CHUNK_W
    for m in (1, 5, W - 1, W, W + 3, 3 * W + 100):
        xyz = np.random.default_rng(m).random((m, 3)).astype(np.float32)
        p = gpu_resolve.pack_chunk(xyz)
        i = np.arange(m)
        ok = (p.shape == (int(np.ceil(m / W)), W, 4)
              and np.array_equal(p[i // W, i % W, :3], xyz)
              and not p[..., 3].any()
              and not p.reshape(-1, 4)[m:].any())
        check("pack_chunk(%d): element i at (i // %d, i %% %d), padding zero" % (m, W, W),
              ok, "shape %s" % (p.shape,))
    tri = np.array([0, 7, 16_777_215, 123_456], np.int64)          # up to 2**24 - 1
    p = gpu_resolve.pack_chunk(np.zeros((4, 3), np.float32), tri)
    check("pack_chunk carries the triangle id in .w, exactly, up to 2**24 - 1",
          np.array_equal(p.reshape(-1, 4)[:4, 3].astype(np.int64), tri)
          and not p.reshape(-1, 4)[4:].any())
    check("a chunk's texture is never wider than 4096 (Metal caps at 16384)",
          gpu_resolve.pack_chunk(np.zeros((gpu_resolve.CHUNK, 3), np.float32)).shape[1] <= 4096)


def readback(gpu_resolve):
    print("\nREADBACK: A GPU THAT IS SILENTLY WRONG IS REFUSED, NOT SHIPPED")
    good = np.array([-1, 0, 3, 7, -1, 2], np.float32)
    idx = gpu_resolve.indices_from_readback(np.concatenate([good, [-2, -2]]), 6, 8)
    check("valid indices come back as int64, padding ignored",
          idx.dtype == np.int64 and idx.tolist() == [-1, 0, 3, 7, -1, 2])

    def refuses(flat, m, n):
        try:
            gpu_resolve.indices_from_readback(np.asarray(flat, np.float32), m, n)
            return False
        except gpu_resolve.GPUResolveError:
            return True

    check("a surviving sentinel (thread never ran) is refused", refuses([0, -2, 1], 3, 8))
    check("an all-sentinel readback (dispatch did nothing) is refused",
          refuses(np.full(100, -2.0), 100, 8))
    check("NaN is refused", refuses([0, np.nan, 1], 3, 8))
    check("an index >= stroke count is refused", refuses([0, 8, 1], 3, 8))
    check("an index below -1 is refused", refuses([0, -5, 1], 3, 8))
    check("a non-integer index is refused", refuses([0, 1.5, 1], 3, 8))
    check("a readback shorter than the chunk is refused", refuses([0, 1], 3, 8))
    check("GPUResolveError is a RuntimeError (the bake's fallback catches it)",
          issubclass(gpu_resolve.GPUResolveError, RuntimeError))


def kernel_source(gpu_resolve, shaders):
    print("\nTHE KERNEL RUNS THE SHARED SEARCH, NOT A COPY OF IT")
    check("compute source contains shaders.SEARCH verbatim",
          shaders.SEARCH in gpu_resolve.COMPUTE)
    check("...and calls find_stroke once per texel, with the texel's triangle",
          gpu_resolve.COMPUTE.count("find_stroke(pt.xyz, nrm, int(pt.w))") == 1
          and "vec4 pt = texelFetch(u_pts, p, 0);" in gpu_resolve.COMPUTE)
    check("layout tokens were substituted (no raw token left)",
          "STROKE_TILE_W" not in gpu_resolve.COMPUTE
          and "STROKE_ROWS" not in gpu_resolve.COMPUTE
          and "CHUNK_W" not in gpu_resolve.COMPUTE)
    # no implicit-LOD sampling anywhere in the search: compute shaders have no screen
    # derivatives, and the mask must use sample_mask's exact bilinear, not GPU filtering
    check("the search samples only with texelFetch (no texture())",
          "texture(" not in shaders.SEARCH)
    check("the tie-break matches resolve_uv (higher index wins an exact tie)",
          "r == best_r && i < best" in shaders.SEARCH)


if __name__ == "__main__":
    sys.exit(main())
