"""bake() can select an object Blender's own bake operator needs selected.

A friend reported "Error: No valid selected object" on Bake -- a string this addon does
not write anywhere. It is Blender's own Cycles bake operator, leaking through uncaught.

ACTIVE and SELECTED are independent flags: the panel and setup.validate() check only
context.active_object, so an object can read as "ready" while hidden or Disable-Selection
locked -- an Outliner click sets active regardless of either, and a reopened .blend
restores whichever object was active when saved. bridge/position.py's bake() then does
select_all(DESELECT) + select_set(True) right before calling Cycles; select_set() does
NOT raise when it fails, it silently no-ops, so the object stays deselected and Cycles
fails with its own opaque message.

Real Cycles baking needs a running Blender, so this cannot be proven with a numpy test the
way the rest of the suite is. What CAN be tested without bpy is the selection/error-
handling logic itself, against a stub bpy built the way test_import.py builds one.

Run directly: python3 autostroke/tests/test_selectable.py
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


class FakeObj:
    """Just enough of bpy.types.Object for bake()'s selection dance."""

    def __init__(self, name="Cube", hide_select=False, hidden=False, in_view_layer=True,
                 has_uv=True):
        self.name = name
        self.hide_select = hide_select
        self._hidden = hidden
        self._selected = False
        self.in_view_layer = in_view_layer     # False simulates view-layer exclusion
        self.material_slots = []
        self.modifiers = []

        class _Materials(list):
            """append()/pop(index=...) -- matches the two calls swapped_material()
            makes, including its keyword-argument pop()."""

            def pop(self, index=None, **kw):
                i = kw.get("index", index)
                return list.pop(self, i)

        class _Data:
            class _UV:
                active = object() if has_uv else None
            uv_layers = _UV()
            materials = _Materials()
        self.data = _Data()

    def hide_get(self):
        return self._hidden

    def hide_set(self, v):
        self._hidden = v

    def select_get(self):
        return self._selected

    def select_set(self, v):
        # The real bpy.types.Object.select_set(): silently does nothing when the
        # object is hidden, selection-locked, or not in the active view layer --
        # never raises. That silence is the entire bug this file exists to catch.
        if self._hidden or self.hide_select or not self.in_view_layer:
            return
        self._selected = v


class FakeActiveSlot:
    """view_layer.objects.active: a plain attribute in real bpy, but assigning an
    object outside the view layer is refused there too."""

    def __init__(self):
        self._obj = None

    def get(self):
        return self._obj

    def set(self, obj):
        if not obj.in_view_layer:
            return                    # matches Object.select_set()'s own silence
        self._obj = obj


def configure(bpy_mod, bake_should_fail=None):
    """(Re)configure an already-imported bpy stub in place.

    bridge/position.py does `import bpy` once, at import time -- so position.bpy is a
    persistent reference to ONE module object forever. Swapping sys.modules["bpy"] for
    a fresh module after that does nothing to position.bpy; only mutating the SAME
    object's attributes reaches it, which is why every scenario below reconfigures the
    one bpy_mod created in main() rather than building a new module each time.

    bake_should_fail: None (bake succeeds once something is selected), or a callable
    obj -> bool deciding whether bpy.ops.object.bake() raises this call.
    """
    active_slot = FakeActiveSlot()
    calls = {"bake": 0}

    class _ActiveProxy:
        def __get__(self, obj, owner):
            return active_slot.get()

        def __set__(self, obj, value):
            active_slot.set(value)

    class _Objects:
        active = _ActiveProxy()

    class _ViewLayer:
        objects = _Objects()

    class _Ops:
        class object:
            @staticmethod
            def select_all(action):
                pass                  # the fixtures track selection per-object directly

            @staticmethod
            def bake(type):
                calls["bake"] += 1
                obj = active_slot.get()
                selected = obj is not None and obj.select_get()
                if not selected:
                    raise RuntimeError("Error: No valid selected objects")
                # Real Cycles: Selected to Active bakes FROM every OTHER selected
                # object ONTO the active one. This fixture only ever selects the
                # active object itself, so with the setting on there is never a
                # source -- matching what actually produced the reported bug.
                if bpy_mod.context.scene.render.bake.use_selected_to_active:
                    raise RuntimeError("Error: No valid selected objects")
                if bake_should_fail is not None and bake_should_fail(obj):
                    raise RuntimeError("Error: some other Cycles failure")

    bpy_mod.context = types.SimpleNamespace(scene=types.SimpleNamespace(
        render=types.SimpleNamespace(engine='CYCLES',
                                     bake=types.SimpleNamespace(margin=16, use_clear=True,
                                                                use_selected_to_active=True)),
        cycles=types.SimpleNamespace(device='CPU', samples=1)),
        view_layer=_ViewLayer())
    bpy_mod.ops = _Ops()
    bpy_mod.data = types.SimpleNamespace(
        materials=types.SimpleNamespace(get=lambda n: None, new=lambda n: FakeMat()),
        images=types.SimpleNamespace(get=lambda n: None,
                                     new=lambda n, w, h, float_buffer, is_data: FakeImg(w, h)))
    return calls


class _AnySocket(dict):
    """Real node sockets are indexable by any name the node type defines; links.new()
    is a no-op in this stub, so the values themselves are never inspected -- only that
    indexing by an arbitrary socket name (e.g. "Position") does not raise."""

    def __missing__(self, key):
        return object()


class FakeNode:
    def __init__(self, bl_idname):
        self.bl_idname = bl_idname
        self.inputs = _AnySocket()
        self.outputs = _AnySocket()
        self.vector_type = self.convert_from = self.convert_to = None


class FakeNodeTree:
    """Just enough of a Blender node tree for build_material()."""

    def __init__(self):
        self._nodes = []
        self.links = types.SimpleNamespace(new=lambda a, b: None)

        class _Nodes(list):
            def clear(inner_self):
                del inner_self[:]

            def new(inner_self, bl_idname):
                n = FakeNode(bl_idname)
                inner_self.append(n)
                return n
        self.nodes = _Nodes()
        self.nodes.active = None


class FakeMat:
    """use_nodes=True auto-creates a node_tree in real Blender -- matched here so
    build_material()'s nt.nodes.clear()/new()/links.new() calls have somewhere to land."""

    def __init__(self):
        self._use_nodes = False
        self.node_tree = None

    @property
    def use_nodes(self):
        return self._use_nodes

    @use_nodes.setter
    def use_nodes(self, v):
        self._use_nodes = v
        if v and self.node_tree is None:
            self.node_tree = FakeNodeTree()


class FakeImg:
    def __init__(self, w, h):
        self.size = (w, h)
        self.is_float = True
        self.channels = 4
        self.colorspace_settings = types.SimpleNamespace(name="")
        # foreach_get(buf) fills buf in place in real bpy; content is never checked
        # here, only that bridge/images.image_to_numpy() can read this image at all.
        self.pixels = types.SimpleNamespace(foreach_get=lambda buf: None)


def load_bake(bpy_mod):
    """bridge/position.py needs bpy at import time -- stub it first, the way
    test_import.py does for the whole addon."""
    sys.modules["bpy"] = bpy_mod
    sys.path.insert(0, ROOT)
    import bridge.position as position   # noqa
    return position


def main():
    bpy_mod = types.ModuleType("bpy")
    calls = configure(bpy_mod)
    position = load_bake(bpy_mod)

    print("\nSELECTABLE() FORCES HIDE_SELECT/HIDDEN OFF, RESTORES EXACTLY")
    obj = FakeObj(hide_select=True, hidden=True)
    with position.selectable(obj):
        check("hide_select cleared inside the block", obj.hide_select is False)
        check("hidden cleared inside the block", obj.hide_get() is False)
    check("hide_select restored to True after", obj.hide_select is True)
    check("hidden restored to True after", obj.hide_get() is True)

    obj2 = FakeObj(hide_select=False, hidden=False)
    with position.selectable(obj2):
        pass
    check("an already-unlocked object is left alone", obj2.hide_select is False
          and obj2.hide_get() is False, "no spurious toggling")

    print("\nSELECTABLE() RESTORES EVEN WHEN THE BLOCK RAISES")
    obj3 = FakeObj(hide_select=True, hidden=True)
    try:
        with position.selectable(obj3):
            raise ValueError("boom")
    except ValueError:
        pass
    check("hide_select restored after an exception", obj3.hide_select is True)
    check("hidden restored after an exception", obj3.hide_get() is True)

    print("\nBAKE() TURNS OFF SELECTED-TO-ACTIVE, RESTORES IT AFTER")
    # Cycles' "Selected to Active" bakes FROM every other selected object ONTO the
    # active one -- a high-poly-to-low-poly workflow. This function selects exactly
    # one object, so with that setting left on (as configure() defaults it, matching
    # a .blend that carries it from unrelated prior work) Cycles finds no source
    # objects and refuses with "No valid selected objects" -- even though the object
    # IS genuinely selected, active, unlocked and visible. This was the actual
    # reported bug; hide_select/hidden (above) is a separate, previously-fixed one.
    calls = configure(bpy_mod)
    bake_render = bpy_mod.context.scene.render.bake
    obj7 = FakeObj(name="Stone Ground")
    check("use_selected_to_active starts ON (simulating a carried-over .blend)",
          bake_render.use_selected_to_active is True)
    try:
        position.bake(obj7, 64, channel="position")
        ok, detail = True, "%d bake call(s)" % calls["bake"]
    except Exception as e:
        ok, detail = False, "%s: %s" % (type(e).__name__, e)
    check("bake() succeeds instead of reporting a phantom selection problem",
          ok, detail)
    check("use_selected_to_active is restored to True afterward",
          bake_render.use_selected_to_active is True,
          "must not leave the artist's own setting changed")

    print("\nBAKE() NOW SUCCEEDS ON A LOCKED/HIDDEN OBJECT")
    obj4 = FakeObj(hide_select=True, hidden=True)
    try:
        position.bake(obj4, 64, channel="position")
        ok, detail = True, "%d bake call(s)" % calls["bake"]
    except Exception as e:
        ok, detail = False, "%s: %s" % (type(e).__name__, e)
    check("bake() completes instead of raising Blender's raw error", ok, detail)
    check("the lock state is restored afterward", obj4.hide_select is True)
    check("the hidden state is restored afterward", obj4.hide_get() is True)

    print("\nA STILL-UNSELECTABLE OBJECT GETS A NAMED, ACTIONABLE ERROR")
    # standing in for view-layer exclusion / a library-linked object: select_set()
    # keeps failing even after selectable() clears hide_select/hidden
    configure(bpy_mod)
    obj5 = FakeObj(name="LockedThing", in_view_layer=False)
    try:
        position.bake(obj5, 64, channel="position")
        ok, msg = False, "did not raise"
    except RuntimeError as e:
        ok, msg = True, str(e)
    check("bake() raises rather than letting Cycles fail silently-wrong", ok, msg)
    check("the message names the object", "LockedThing" in msg, msg)
    check("and it does NOT just repeat Blender's bare string",
          "No valid selected" not in msg, msg)

    print("\nAN UNRELATED CYCLES FAILURE IS WRAPPED WITH CONTEXT, NOT SWALLOWED BARE")
    configure(bpy_mod, bake_should_fail=lambda obj: True)
    obj6 = FakeObj(name="Suzanne")
    try:
        position.bake(obj6, 64, channel="normal")
        ok, msg = False, "did not raise"
    except RuntimeError as e:
        ok, msg = True, str(e)
    check("still raises", ok, msg)
    check("names the object", "Suzanne" in msg, msg)
    check("names the channel", "normal" in msg, msg)
    check("keeps Blender's own text too, for anyone actually debugging it",
          "some other Cycles failure" in msg, msg)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
