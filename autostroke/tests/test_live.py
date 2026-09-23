"""The numpy half of the live preview: stroke packing and the spatial grid.

The shader itself cannot be tested here -- it needs a GPU and a running Blender. What CAN
be tested is the part most likely to be quietly wrong: the binning. The shader reads only
a fragment's OWN cell, which is only correct if every stroke was inserted into every cell
its bounding box overlaps. Get that wrong and strokes vanish in bands, which reads as a
placement bug rather than a binning bug.

Run directly: python3 autostroke/tests/test_live.py
"""

import ast
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import baker, geometry                              # noqa: E402
from core.config import Config, INV_SQRT2                     # noqa: E402


def _load_pure():
    """The pure-numpy half of livepreview.py, lifted out of the shipping source.

    Importing the module itself would drag in bpy and gpu through its relative imports.
    Reimplementing these here would pass while the real thing was broken, so they are
    exec'd from the file instead.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "livepreview.py")
    tree = ast.parse(open(path).read())
    want = {"pack_strokes", "build_grid", "pack_brush_atlas", "stroke_tile_buffer"}
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    if len(keep) != len(want):
        raise SystemExit("livepreview.py no longer defines %s"
                         % sorted(want - {n.name for n in keep}))
    # STROKE_TILE_W is stroke_tile_buffer's default arg value, evaluated when the
    # function def below executes -- it has to exist in ns first.
    const = [n for n in tree.body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "STROKE_TILE_W"]
    if not const:
        raise SystemExit("livepreview.py no longer defines STROKE_TILE_W")
    ns = {"np": np, "baker": baker, "geometry": geometry, "INV_SQRT2": INV_SQRT2}
    exec(compile(ast.Module(body=const + keep, type_ignores=[]), "<livepreview>", "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in want | {"STROKE_TILE_W"}})


LP = _load_pure()

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


AREA = 16.0          # the fixture surface is 4 x 4


def fake_seeds(n=400, seed=2):
    """Radius scaled to the stroke count, the way core.sampling actually sizes seeds.

    This matters: holding radius fixed while raising the count makes strokes overlap
    absurdly, which made the grid look hopeless (10,260 strokes in one cell at 20,000
    strokes) when the real sampler shrinks strokes as it places more of them.
    """
    rng = np.random.default_rng(seed)
    u, v = rng.random(n) * 4.0, rng.random(n) * 4.0
    pos = np.stack([u, v, 0.4 * np.sin(u) * np.cos(v)], 1).astype(np.float32)
    du = np.stack([np.ones_like(u), np.zeros_like(u), 0.4 * np.cos(u) * np.cos(v)], 1)
    dv = np.stack([np.zeros_like(u), np.ones_like(u), -0.4 * np.sin(u) * np.sin(v)], 1)
    nrm = np.cross(du, dv)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    base = np.sqrt(AREA / n) * 0.5
    return dict(position=pos, normal=nrm.astype(np.float32),
                radius=(base * (0.7 + 0.6 * rng.random(n))).astype(np.float32),
                id=np.arange(n, dtype=np.int64),
                UVMap=np.stack([rng.random(n), rng.random(n)], 1).astype(np.float32))


def bars(k, px=64):
    out = []
    for i in range(k):
        y, x = (np.mgrid[0:px, 0:px] / (px - 1.0)) - 0.5
        out.append(((np.abs(x) < 0.30 - 0.05 * i) & (np.abs(y) < 0.12)).astype(np.float32))
    return out


def main():
    print("\nLIVE PREVIEW (numpy half)\n")
    cfg = Config(mask_scale=6.0)
    s = fake_seeds()
    masks = bars(4)
    packed, pos, r, T, nrm = LP.pack_strokes(s, s["normal"].astype(np.float64), masks, cfg)

    check("one RGBA row group per stroke", packed.shape == (len(pos), 4, 4), str(packed.shape))
    check("row 0 carries position and radius",
          np.allclose(packed[:, 0, :3], pos, atol=1e-5)
          and np.allclose(packed[:, 0, 3], r, atol=1e-5))
    check("T is unit length and in the tangent plane",
          np.abs(np.linalg.norm(packed[:, 1, :3], axis=1) - 1).max() < 1e-4
          and np.abs((packed[:, 1, :3] * nrm).sum(1)).max() < 1e-4)
    check("brush index is in range",
          packed[:, 2, 3].min() >= 0 and packed[:, 2, 3].max() <= len(masks) - 1)
    check("the painted box is non-empty and inside the stamp square",
          (packed[:, 3, 1] > packed[:, 3, 0]).all() and (packed[:, 3, 3] > packed[:, 3, 2]).all()
          and (packed[:, 3, 1] <= r * 2 * INV_SQRT2 * 0.5 + 1e-5).all())

    # ---- the load-bearing one: the grid must be conservative ---------------
    print()
    bbl = packed[:, 3, :].astype(np.float64)
    gmin, ginv, dim, starts, counts, lst = LP.build_grid(pos, r, T, nrm, bbl)
    n_cells = int(np.prod(dim))
    check("grid dims are sane", n_cells > 0 and n_cells == len(counts),
          "%dx%dx%d = %s cells" % (dim[0], dim[1], dim[2], "{:,}".format(n_cells)))
    check("every stroke is binned at least once",
          len(np.unique(lst.astype(np.int64))) == len(pos),
          "%d of %d strokes appear" % (len(np.unique(lst.astype(np.int64))), len(pos)))

    # For a sample of points that a stroke could paint, that stroke must be in the cell
    # the shader will look in -- otherwise the shader silently misses it.
    B = np.cross(nrm, T)
    rng = np.random.default_rng(7)
    misses = tested = 0
    for j in rng.choice(len(pos), 120, replace=False):
        a = rng.uniform(bbl[j, 0], bbl[j, 1], 40)
        b = rng.uniform(bbl[j, 2], bbl[j, 3], 40)
        p = pos[j] + T[j] * a[:, None] + B[j] * b[:, None]
        p = p[((p - pos[j]) ** 2).sum(1) < r[j] ** 2]        # only points it can paint
        for q in p:
            c = np.clip(((q - gmin) * ginv).astype(np.int64), 0, dim - 1)
            cid = int(c[0] + dim[0] * (c[1] + dim[1] * c[2]))
            here = lst[int(starts[cid]):int(starts[cid]) + int(counts[cid])].astype(np.int64)
            tested += 1
            if j not in here:
                misses += 1
    check("a stroke is in the cell of every point it can paint", misses == 0,
          "%s points checked, %d missed" % ("{:,}".format(tested), misses))
    check("cells stay short enough for a shader loop", counts.max() <= 512,
          "largest cell holds %d strokes, mean %.1f" % (counts.max(), counts.mean()))

    # The property that actually matters: the loop must NOT grow with stroke count. The
    # grid refines as strokes shrink, so it should not -- but only if the seeds shrink,
    # which is why the fixture scales radius with n.
    worst = []
    for n in (1000, 4000, 20000):
        sn = fake_seeds(n)
        pk, pp, rr, TT, NN = LP.pack_strokes(sn, sn["normal"].astype(np.float64), masks, cfg)
        _, _, _, _, cts, _ = LP.build_grid(pp, rr, TT, NN, pk[:, 3, :].astype(np.float64))
        worst.append((n, int(cts.max())))
    check("the shader loop does not grow with stroke count",
          worst[-1][1] < 3 * worst[0][1],
          " ".join("%s->%d" % (format(n, ","), c) for n, c in worst))

    # ---- brush atlas -------------------------------------------------------
    print()
    atlas, cols = LP.pack_brush_atlas(masks)
    check("atlas is square and tiles the set", atlas.shape[0] == atlas.shape[1]
          and cols * cols >= len(masks), "%dx%d, %d cols" % (*atlas.shape, cols))
    tile = atlas.shape[0] // cols
    ok = True
    for i, m in enumerate(masks):
        rr, cc = divmod(i, cols)
        sub = atlas[rr * tile:(rr + 1) * tile, cc * tile:(cc + 1) * tile]
        if abs(float((sub > 0.5).mean()) - float((m > 0.5).mean())) > 0.02:
            ok = False
    check("each tile still holds its own brush", ok,
          "painted fraction preserved through the resample")
    check("a single brush still works", LP.pack_brush_atlas(bars(1))[1] == 1)

    debounce()
    shading_normals()
    stroke_texture_width()

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


def stroke_texture_width():
    """stroke_tex's width never grows with stroke count.

    It used to: width was exactly the stroke count, and Metal's MTLTextureDescriptor
    caps a 2D texture at 16384 texels wide (other backends have their own caps, often
    similar). A preview past 16384 strokes -- comfortably inside PREVIEW_BUDGET's
    60,000 -- crashed the whole Blender process with a native GPU assertion: not a
    Python exception, so nothing in this addon could catch or report it. Fixed by
    tiling into a fixed-width grid, the same way list_tex already needed to be.
    """
    print("\nSTROKE TEXTURE WIDTH NEVER GROWS WITH STROKE COUNT")
    W = LP.STROKE_TILE_W
    for n in (1, W - 1, W, W + 1, 20000, 60000):        # 60000 = PREVIEW_BUDGET
        packed = np.random.default_rng(n).random((n, 4, 4)).astype(np.float32)
        out = LP.stroke_tile_buffer(packed)
        check("n=%-6d -> texture width %d (never %d)" % (n, out.shape[1], n),
              out.shape[1] == W, "shape %s" % (out.shape,))
        check("n=%-6d -> height is a multiple of 4 (one stroke = 4 texel rows)" % n,
              out.shape[0] % 4 == 0, "shape %s" % (out.shape,))

    print("\nSTROKE TEXTURE WIDTH: THE PACKING IS LOSSLESS")
    for n in (1, 5, 4096, 4097, 8500):
        packed = np.random.default_rng(n + 1).random((n, 4, 4)).astype(np.float32)
        out = LP.stroke_tile_buffer(packed)
        ok = True
        # spot-check every stroke for small n, a sample for large n, matching the
        # shader's own stroke_row(): (i % W, (i // W) * 4 + row)
        idx = range(n) if n <= 200 else np.random.default_rng(0).integers(0, n, 200)
        for i in idx:
            col, base = i % W, (i // W) * 4
            for r in range(4):
                if not np.array_equal(out[base + r, col], packed[i, r]):
                    ok = False
                    break
            if not ok:
                break
        check("n=%d: every stroke reads back at the shader's own (i%%W, (i//W)*4+row)"
              % n, ok)

    check("an empty stroke set does not raise",
          LP.stroke_tile_buffer(np.zeros((0, 4, 4), np.float32)).shape[1] == W)




def debounce():
    """A slider fires its update callback on every tick, so changes must coalesce.

    Rebuilding per tick is ~10 ms of numpy plus four texture uploads; at 60 ticks a
    second that is a stutter. Only the coalescing is tested here -- whether Blender's
    timer actually fires is not something a stub can tell us.
    """
    print()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "livepreview.py")
    tree = ast.parse(open(path).read())
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in {"request_rebuild", "_flush"}]
    if len(keep) != 2:
        check("request_rebuild and _flush still exist", False)
        return

    registered = []
    ns = {
        "bpy": types.SimpleNamespace(
            app=types.SimpleNamespace(timers=types.SimpleNamespace(
                register=lambda fn, first_interval=0: registered.append(fn))),
            context=None),
        "enabled": lambda: ns["_on"],
        "rebuild": lambda ctx: ns.setdefault("_rebuilds", 0) or ns.update(
            _rebuilds=ns.get("_rebuilds", 0) + 1),
        "_redraw": lambda: None,
        "_pending": False,
        "_on": True,
        "_DEBOUNCE": 0.12,
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<lp>", "exec"), ns)

    ns["_on"] = False
    ns["request_rebuild"]()
    check("a change with the preview off does nothing", registered == [])

    ns["_on"] = True
    for _ in range(50):                      # a slider drag
        ns["request_rebuild"]()
    check("50 changes arm exactly one rebuild", len(registered) == 1,
          "%d timers registered" % len(registered))

    registered[0]()                          # the timer fires
    check("the rebuild runs once", ns.get("_rebuilds", 0) == 1)
    check("and the next change arms a fresh one", (ns["request_rebuild"]() or True)
          and len(registered) == 2, "%d total" % len(registered))

def shading_normals():
    """The preview feeds the crease test CORNER normals, as the baker does.

    Per-vertex normals are always smooth-averaged. On a hard edge that average points
    between the two faces, so `dot(n, SN)` lands near the cutoff on BOTH sides and the
    preview stops strokes where the bake does not -- systematically, everywhere the mesh
    has a sharp edge, custom split normals or Auto Smooth.

    Corner normals mean one vertex can carry several, so the vertex buffer has to be
    expanded to three unshared corners per triangle. A reshape mistake there scrambles
    the mesh silently, so the indexing is exec'd from the shipped source rather than
    retyped here.
    """
    print("\nSHADING NORMALS")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "livepreview.py")
    src = open(path).read()
    tree = ast.parse(src)
    build = next(n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "_build")

    check("_build reads corner_normals", "corner_normals" in ast.dump(build))
    check("the batch is unindexed (corners are unshared)",
          "batch_for_shader" in src and "indices=tri" not in src)

    # Lift the two expansion statements out of the shipped function. `vnrm` is assigned
    # twice -- once from corner normals, once in the older-build fallback -- so take the
    # corner-normal one, which is the path being tested.
    assigns = [n for n in ast.walk(build)
               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)]
    stmts = [n for n in assigns if n.targets[0].id == "vpos"]
    stmts += [n for n in assigns if n.targets[0].id == "vnrm"
              and "corner_n" in ast.dump(n.value)]
    got = {n.targets[0].id for n in stmts}
    check("both expansions found in source", got == {"vpos", "vnrm"} and len(stmts) == 2,
          "%d statements: %s" % (len(stmts), ", ".join(sorted(got))))
    if got != {"vpos", "vnrm"} or len(stmts) != 2:
        return

    # a folded quad: two triangles meeting at a 90 deg HARD edge, split normals there
    co = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                   [0, 1, 0], [1, 1, 1], [0, 1, 1]], np.float32)
    tri_v = np.array([[0, 1, 2], [2, 4, 5]], np.int32)
    tri_l = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    nA = np.array([0, 0, 1.0], np.float32)
    nB = np.array([0, -1.0, 0], np.float32)
    corner_n = np.stack([nA, nA, nA, nB, nB, nB]).astype(np.float32)

    ns = {"np": np, "co": co, "tri_v": tri_v, "tri_l": tri_l, "corner_n": corner_n}
    for st in stmts:
        exec(compile(ast.Module(body=[st], type_ignores=[]), "<livepreview>", "exec"), ns)
    vpos, vnrm = ns["vpos"], ns["vnrm"]

    check("one vertex per triangle corner", vpos.shape == (6, 3) and vnrm.shape == (6, 3),
          "%s / %s" % (vpos.shape, vnrm.shape))
    check("position follows the triangle's own vertex",
          all(np.array_equal(vpos[3 * t + c], co[tri_v[t, c]])
              for t in range(2) for c in range(3)))
    check("normal follows the triangle's own loop",
          all(np.array_equal(vnrm[3 * t + c], corner_n[tri_l[t, c]])
              for t in range(2) for c in range(3)))

    # the shared vertex keeps a DIFFERENT normal on each side of the fold
    check("the hard edge stays hard", not np.array_equal(vnrm[2], vnrm[3]),
          "%s vs %s" % (vnrm[2], vnrm[3]))

    # and that is what the crease test sees
    smoothed = nA + nB
    smoothed = smoothed / np.linalg.norm(smoothed)
    cut = float(np.cos(np.radians(45.0)))
    check("smoothed normal would reject a stroke the bake keeps",
          float(smoothed @ nA) <= cut < float(vnrm[2] @ nA),
          "old %.3f vs new %.3f, cutoff %.3f" % (smoothed @ nA, vnrm[2] @ nA, cut))
    check("and still rejects on the far face", float(vnrm[3] @ nA) <= cut,
          "%.3f" % (vnrm[3] @ nA))


if __name__ == "__main__":
    sys.exit(main())
