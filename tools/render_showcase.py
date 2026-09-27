"""Before/after renders for the README and portfolio, from a fresh bake with the repo code.

Runs in a real Blender window (the GPU bake and EEVEE both want a GPU context), bakes each
object with the scene's saved AutoStroke settings, renders, writes a report and exits. It
never saves the .blend: every change (visibility, modifiers, materials, cameras) is made in
memory only, and the bake writes its maps into a temporary folder, not Output/.

    Blender --factory-startup PainterlyTexture.blend --python tools/render_showcase.py \
            -- <out_dir> [object ...]

For each object (default: Body) it writes, with the scene's lights:
  <name>_before.png  the mesh with a neutral grey material (what you start from)
  <name>_after.png   the same mesh with the <name>_AutoStroke material from that bake

Every object is shot with a temporary camera looking the SAME way as the scene camera,
pulled back to fit the object's bounding sphere, so shots are framed alike. Geometry-nodes
modifiers are turned off for the render, as the bake itself does.

--factory-startup keeps any INSTALLED copy of the add-on from loading, so the repo copy can
register without its classes colliding with the installed ones.
"""

import math
import os
import sys
import tempfile
import time
import traceback

import bpy
from mathutils import Vector

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(ARGV[0]) if ARGV else os.path.join(REPO, "docs", "media")
NAMES = ARGV[1:] or ["Body"]
RES = (1600, 900)
WORK = tempfile.mkdtemp(prefix="autostroke_showcase_")
lines = []
state = {"queue": list(NAMES), "current": None, "t": 0.0}


def log(s=""):
    lines.append(str(s))
    print(s)


def finish():
    with open(os.path.join(OUT, "render_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os._exit(0)


def view3d():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                return win, area, next(r for r in area.regions if r.type == 'WINDOW')
    raise RuntimeError("no 3D viewport in this window")


def start_bake(name):
    obj = bpy.data.objects[name]
    obj.hide_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    st = bpy.context.scene.autostroke
    st.working_dir = WORK + os.sep
    st.bake_device = 'GPU'
    st.last_report = ""
    win, area, region = view3d()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.autostroke.bake('INVOKE_DEFAULT')
    state["current"], state["t"] = name, time.time()


def grey_material():
    mat = bpy.data.materials.new("showcase_grey")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def camera_for(obj, base):
    """A copy of the scene camera, same view direction, pulled back to fit obj."""
    cam = base.copy()
    cam.data = base.data.copy()
    bpy.context.scene.collection.objects.link(cam)
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    centre = sum(corners, Vector()) / 8.0
    radius = max((c - centre).length for c in corners)
    fwd = (base.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()
    fov = min(cam.data.angle_x, cam.data.angle_y)
    cam.location = centre - fwd * (radius / math.sin(fov / 2.0) * 1.02)
    return cam


def shoot(cam, path):
    sc = bpy.context.scene
    sc.camera = cam
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    log("  wrote %s (%d KB)" % (os.path.basename(path), os.path.getsize(path) // 1024))


def render_all():
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    base, grey = sc.camera, grey_material()
    log("render: %s, %dx%d" % (sc.render.engine, *RES))
    for name in NAMES:
        obj = bpy.data.objects[name]
        mat = bpy.data.materials.get(name + "_AutoStroke")
        if mat is None:
            log("%s: no material after the bake -- skipped" % name)
            continue
        for o in bpy.data.objects:
            if o.type == 'MESH':
                o.hide_render = (o is not obj)
        for m in obj.modifiers:
            if m.type == 'NODES':
                m.show_render = False
        log("%s:" % name)
        cam = camera_for(obj, base)
        for s in obj.material_slots:
            s.material = grey
        shoot(cam, os.path.join(OUT, "%s_before.png" % name.lower()))
        for s in obj.material_slots:
            s.material = mat
        shoot(cam, os.path.join(OUT, "%s_after.png" % name.lower()))


def tick():
    try:
        st = getattr(bpy.context.scene, "autostroke", None)
        if st is None:
            sys.path.insert(0, REPO)
            import autostroke
            autostroke.register()
            os.makedirs(OUT, exist_ok=True)
            log("bake work dir (temporary): %s" % WORK)
            return 0.5
        if state["current"] is not None:
            if not st.last_report:
                if time.time() - state["t"] > 600:
                    log("TIMEOUT baking %s" % state["current"])
                    finish()
                return 0.25
            log("%s baked in %.1fs: %s" % (state["current"], time.time() - state["t"],
                                            st.last_report.replace("\n", " | ")))
            state["current"] = None
        if state["queue"]:
            start_bake(state["queue"].pop(0))
            return 0.25
        render_all()
    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc())
    finish()


bpy.app.timers.register(tick, first_interval=1.0)
