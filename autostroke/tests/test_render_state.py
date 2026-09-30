"""Strokes, reach tables, the live preview and the Cycles bake read the SAME mesh.

The bug this guards (a user's StairSubject.blend): a Bevel modifier enabled for render
only. Cycles bakes with render settings, so the position/normal maps described the
bevelled mesh (12,176 triangles); the depsgraph Python reads uses viewport settings, so
the strokes, their reach sets and the preview described the unbevelled one (784). Most
texels then saw strokes placed on a different surface and rejected them: 45% coverage,
against 88% once both read the same mesh. The preview looked right, so preview and bake
disagreed.

bridge/mesh.render_state switches every modifier to what the bake evaluates -- render
visibility, render subdivision levels, geometry nodes off as position.disabled_geometry_
nodes has them -- for the duration of a mesh read, then restores it exactly. It is lifted
from the shipping source and run against fake modifiers.

Run directly: python3 autostroke/tests/test_render_state.py
"""

import ast
import os
import sys
import types
from contextlib import contextmanager

FAILED = []


def check(name, ok, detail=""):
    print("   %-62s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def load():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "bridge", "mesh.py")
    tree = ast.parse(open(path).read())
    want = {"RENDER_LEVELS", "render_changes", "render_state"}
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in want)
            or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in want)]
    updates = []
    bpy = types.SimpleNamespace(context=types.SimpleNamespace(
        view_layer=types.SimpleNamespace(update=lambda: updates.append(1))))
    ns = {"bpy": bpy, "contextmanager": contextmanager}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<mesh>", "exec"), ns)
    missing = want - set(ns)
    if missing:
        raise SystemExit("bridge/mesh.py no longer defines %s" % sorted(missing))
    return ns, updates


def mod(type_, viewport=True, render=True, **levels):
    return types.SimpleNamespace(type=type_, show_viewport=viewport, show_render=render,
                                 **levels)


def state(obj):
    return [(m.type, m.show_viewport, getattr(m, "levels", None)) for m in obj.modifiers]


def main():
    ns, updates = load()
    render_state = ns["render_state"]
    print("\nTHE MESH IS READ AS THE BAKE SEES IT, THEN PUT BACK")

    plain = types.SimpleNamespace(modifiers=[mod('BEVEL'), mod('WEIGHTED_NORMAL'),
                                             mod('SUBSURF', levels=2, render_levels=2)])
    with render_state(plain) as changed:
        pass
    check("viewport == render: nothing touched, no depsgraph update",
          not changed and not updates)

    stair = types.SimpleNamespace(modifiers=[mod('BEVEL', viewport=False, render=True),
                                             mod('WEIGHTED_NORMAL')])
    before = state(stair)
    with render_state(stair) as changed:
        inside = state(stair)
    check("render-only Bevel is ON while the mesh is read (the StairSubject case)",
          changed and inside[0][1] is True)
    check("...and OFF again afterwards", state(stair) == before)

    sub = types.SimpleNamespace(modifiers=[mod('SUBSURF', levels=1, render_levels=3)])
    with render_state(sub):
        inside = state(sub)
    check("Subdivision reads at its render level", inside[0][2] == 3)
    check("...and the viewport level comes back", state(sub)[0][2] == 1)

    multi = types.SimpleNamespace(modifiers=[mod('MULTIRES', levels=0, render_levels=2)])
    with render_state(multi):
        inside = state(multi)
    check("Multires reads at its render level too", inside[0][2] == 2 and state(multi)[0][2] == 0)

    off = types.SimpleNamespace(modifiers=[mod('SUBSURF', viewport=True, render=False,
                                               levels=1, render_levels=3)])
    with render_state(off):
        inside = state(off)
    check("a modifier disabled for render is off, its levels left alone",
          inside[0][1] is False and inside[0][2] == 1 and state(off)[0][1] is True)

    gn = types.SimpleNamespace(modifiers=[mod('NODES', viewport=True, render=True)])
    with render_state(gn):
        inside = state(gn)
    check("geometry nodes are off, as the Cycles bake has them",
          inside[0][1] is False and state(gn)[0][1] is True)

    boom = types.SimpleNamespace(modifiers=[mod('BEVEL', viewport=False, render=True),
                                            mod('SUBSURF', levels=1, render_levels=2)])
    before = state(boom)
    try:
        with render_state(boom):
            raise RuntimeError("read failed")
    except RuntimeError:
        pass
    check("settings come back even when the read raises", state(boom) == before)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
