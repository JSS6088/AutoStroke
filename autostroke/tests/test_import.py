"""The addon imports and registers.

Three times now a text splice has removed something the rest of the file still referenced
-- twice in livepreview.py, once taking props._redraw with it. The last one shipped, and
the only symptom was `name '_redraw' is not defined` when enabling the addon in Blender.

Every other suite tests pure-numpy code, so none of them import a module that touches bpy
-- which is all the ones that break this way. This one stubs bpy just far enough to import
the package and run register()/unregister(), which is where property annotations are
evaluated and classes are registered. It does not test behaviour; it tests that the addon
loads at all.

Run directly: python3 autostroke/tests/test_import.py
"""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def stub_bpy():
    """Just enough bpy for import time and register(). Deliberately minimal: a faithful
    mock would be its own bug surface, and anything this cannot express is something the
    real Blender has to check anyway."""
    registered = []

    class _Type:
        """Stands in for bpy.types.* base classes, and accepts the attributes register()
        assigns (Scene.autostroke = PointerProperty(...))."""

    def _prop(*a, **kw):
        return ("prop", a, kw)

    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(
        PropertyGroup=type("PropertyGroup", (_Type,), {}),
        Operator=type("Operator", (_Type,), {}),
        Panel=type("Panel", (_Type,), {}),
        Scene=type("Scene", (_Type,), {}),
        Object=type("Object", (_Type,), {}),
        SpaceView3D=types.SimpleNamespace(draw_handler_add=lambda *a, **k: object(),
                                          draw_handler_remove=lambda *a, **k: None),
        Mesh=type("Mesh", (_Type,), {}),
    )
    bpy.props = types.SimpleNamespace(**{
        n: _prop for n in ("BoolProperty", "EnumProperty", "FloatProperty", "IntProperty",
                           "PointerProperty", "StringProperty", "FloatVectorProperty",
                           "CollectionProperty")})
    bpy.utils = types.SimpleNamespace(
        register_class=lambda c: registered.append(c),
        unregister_class=lambda c: registered.remove(c) if c in registered else None)
    bpy.app = types.SimpleNamespace(
        timers=types.SimpleNamespace(register=lambda *a, **k: None,
                                     unregister=lambda *a, **k: None,
                                     is_registered=lambda *a, **k: False))
    bpy.context = None
    bpy.data = types.SimpleNamespace(objects={}, images={}, materials={})
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy.types
    sys.modules["bpy.props"] = bpy.props
    sys.modules["bpy.utils"] = bpy.utils

    for name in ("gpu", "gpu_extras", "gpu_extras.batch", "bmesh", "mathutils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    return bpy, registered


def main():
    print("\nTHE ADDON IMPORTS AND REGISTERS")
    bpy, registered = stub_bpy()
    sys.path.insert(0, os.path.dirname(ROOT))

    try:
        import autostroke
    except Exception as e:
        check("import autostroke", False, "%s: %s" % (type(e).__name__, e))
        print("\nFAILED: import\n")
        return 1
    check("import autostroke", True, "%d modules" % len(
        [m for m in sys.modules if m.startswith("autostroke")]))

    # this is where property annotations get evaluated and classes registered --
    # the step that failed in Blender with `name '_redraw' is not defined`
    try:
        autostroke.register()
    except Exception as e:
        check("register()", False, "%s: %s" % (type(e).__name__, e))
        print("\nFAILED: register\n")
        return 1
    check("register()", True, "%d classes" % len(registered))

    names = [getattr(c, "bl_idname", c.__name__) for c in registered]
    check("the operators are all there",
          sum(1 for n in names if n.startswith("autostroke.")) >= 8,
          ", ".join(sorted(n for n in names if n.startswith("autostroke."))))
    check("the settings group registered",
          any(c.__name__ == "AutoStrokeSettings" for c in registered))
    check("the scene got its pointer", hasattr(bpy.types.Scene, "autostroke"))

    try:
        autostroke.unregister()
    except Exception as e:
        check("unregister()", False, "%s: %s" % (type(e).__name__, e))
    else:
        check("unregister()", True, "%d classes left" % len(registered))
        check("and it unwinds completely", not registered, "%d left" % len(registered))

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
