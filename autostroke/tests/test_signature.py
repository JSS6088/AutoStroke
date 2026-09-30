"""The position/normal cache key must notice the mesh changing.

It used to be `verts | faces | uv_layer_NAME | resolution`, read off the BASE mesh. That
cannot see a vertex move, a re-unwrap onto the same layer, or any modifier change that
leaves base counts alone -- and those maps are cached while the seeds are not, so a missed
change resolves fresh strokes against a texel->3D mapping for geometry that is gone.

`bpy` is stubbed with a fake evaluated mesh, so this drives the real `signature()` without
Blender. Run directly: python3 autostroke/tests/test_signature.py
"""

import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


# ---- the smallest fake Blender that signature() can read ------------------
class Attr:
    """Stands in for me.vertices / me.loops / uv_layers.active.data."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def foreach_get(self, _name, out):
        out[:] = self.data.ravel()


class FakeMesh:
    def __init__(self, co, loop_v, uv, has_uv=True):
        self.vertices = Attr(co)
        self.loops = Attr(loop_v)
        self.uv_layers = types.SimpleNamespace(
            active=types.SimpleNamespace(data=Attr(uv)) if has_uv else None)


class FakeObj:
    type = 'MESH'
    name = "Fake"
    modifiers = ()      # viewport == render: mesh.render_state has nothing to switch

    def __init__(self, mesh):
        self._mesh = mesh
        self.cleared = 0

    def evaluated_get(self, _deps):
        return self

    def to_mesh(self):
        return self._mesh

    def to_mesh_clear(self):
        self.cleared += 1


def install_stub():
    sys.modules["bpy"] = types.SimpleNamespace(
        context=types.SimpleNamespace(evaluated_depsgraph_get=lambda: None),
        types=types.SimpleNamespace(Operator=object),
        data=None, path=types.SimpleNamespace(abspath=lambda p: p))


def mesh(nv=200, seed=0):
    rng = np.random.default_rng(seed)
    co = rng.random((nv, 3)).astype(np.float32)
    loop_v = np.arange(nv * 2, dtype=np.int32) % nv
    uv = rng.random((nv * 2, 2)).astype(np.float32)
    return co, loop_v, uv


def main():
    print("\nCACHE SIGNATURE\n")
    install_stub()
    from bridge import mesh as MB

    co, loop_v, uv = mesh()

    def sig(co=co, loop_v=loop_v, uv=uv, extra="res=2048"):
        return MB.signature(FakeObj(FakeMesh(co, loop_v, uv)), extra=extra)

    base = sig()
    check("stable when nothing changes", sig() == base, base)

    # The case the old key was blind to, and the reason for all of this.
    moved = co.copy(); moved[7, 1] += 1e-4
    check("ONE vertex moved by 1e-4 changes it", sig(co=moved) != base,
          "the old key could not see this at all")

    uv2 = uv.copy(); uv2[3, 0] += 1e-4
    check("one UV coordinate changed -> changes", sig(uv=uv2) != base,
          "a re-unwrap onto the same layer")

    lv2 = loop_v.copy(); lv2[5], lv2[6] = lv2[6], lv2[5]
    check("connectivity changed, coordinates untouched -> changes",
          sig(loop_v=lv2) != base, "an edge flip")

    check("resolution changed -> changes", sig(extra="res=4096") != base)

    # Everything the old key DID catch must still be caught.
    check("vertex count changed -> changes", sig(co=np.vstack([co, co[:1]])) != base)
    check("loop count changed -> changes",
          sig(loop_v=np.concatenate([loop_v, loop_v[:1]])) != base)

    # A moved vertex must not collide with a UV edit, or the two invalidate as one.
    check("different edits give different digests",
          len({sig(co=moved), sig(uv=uv2), sig(loop_v=lv2), base}) == 4)

    # to_mesh_clear has to run or every bake leaks a mesh datablock.
    o = FakeObj(FakeMesh(co, loop_v, uv))
    MB.signature(o)
    check("the temporary mesh is released", o.cleared == 1, "to_mesh_clear x%d" % o.cleared)

    bad = FakeObj(FakeMesh(co, loop_v, uv, has_uv=False))
    try:
        MB.signature(bad)
        ok = False
    except MB.MeshError:
        ok = True
    check("a mesh with no UV map raises MeshError", ok)
    check("a non-mesh raises MeshError",
          _raises(lambda: MB.signature(types.SimpleNamespace(type='EMPTY', name="E"))))

    # Cost: this is paid on every bake, including cache hits.
    big = mesh(500_000, seed=1)
    o = FakeObj(FakeMesh(*big))
    t = time.time()
    MB.signature(o)
    dt = time.time() - t
    check("500k vertices hash in well under 50 ms", dt < 0.05, "%.1f ms" % (1000 * dt))

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


def _raises(fn):
    from bridge import mesh as MB
    try:
        fn()
    except MB.MeshError:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
