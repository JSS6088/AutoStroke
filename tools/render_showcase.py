"""Before/after renders for the README and portfolio, from a fresh bake with the repo code.

Runs in a real Blender window (the GPU bake and EEVEE both want a GPU context), bakes each
object with the scene's saved AutoStroke settings, renders, writes a report and exits. It
never saves the .blend: every change (visibility, modifiers, materials, cameras) is made in
memory only, and the bake writes its maps into a temporary folder, not Output/.

    Blender --factory-startup PainterlyTexture.blend --python tools/render_showcase.py \
            -- <out_dir> [object[@dx,dy,dz[,roll]] ...] [sun=K] [world=K]

For each object (default: Body) it writes, with the scene's lights:
  <name>_before.png  the mesh with a neutral grey material (what you start from)
  <name>_after.png   the same mesh with the <name>_AutoStroke material from that bake

Every object is shot with a temporary camera, framed tightly on the object's projected
vertices. By default it looks the same way as the scene camera. `@dx,dy,dz` instead places the camera
in that world-space direction from the object's centre, looking back at it (e.g.
`Suzanne@-0.4,-1,-0.7` is front, a little left and below). An optional 4th value rolls the
camera, in degrees. Geometry-nodes modifiers are turned off for the render, as the bake
itself does. `sun=K` and `world=K` scale the sun's energy and the world background's
strength (in memory), for more light/shadow contrast than the scene's defaults.

The README's images come from:

    ... -- docs/media "Suzanne@-0.7,-1,-0.2" sun=2 world=0.7

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
from mathutils import Quaternion, Vector

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(ARGV[0]) if ARGV else os.path.join(REPO, "docs", "media")
OPTS = dict(a.split("=", 1) for a in ARGV[1:] if "=" in a)
SPECS = [a for a in ARGV[1:] if "=" not in a] or ["Body"]
NAMES = [a.split("@")[0] for a in SPECS]
VIEWS = {a.split("@")[0]: [float(v) for v in a.split("@")[1].split(",")]
         for a in SPECS if "@" in a}
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


def camera_for(obj, base, view=None):
    """A copy of the scene camera looking along the scene camera's direction, or from
    `view` = (dx, dy, dz[, roll_deg]) towards the object, framed tightly on the object's
    vertices as projected onto the image plane (5% margin)."""
    cam = base.copy()
    cam.data = base.data.copy()
    bpy.context.scene.collection.objects.link(cam)
    if view:
        rot = (-Vector(view[:3])).normalized().to_track_quat('-Z', 'Y')
        if len(view) > 3:
            rot = rot @ Quaternion((0.0, 0.0, 1.0), math.radians(view[3]))
    else:
        rot = base.matrix_world.to_quaternion()
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = rot
    sc = bpy.context.scene
    cam.data.sensor_fit = 'HORIZONTAL'
    tx = math.tan(cam.data.angle_x / 2.0)
    ty = tx * sc.render.resolution_y / sc.render.resolution_x
    # vertices in the camera's frame (camera looks down -Z); the camera sits at (cx, cy, d)
    inv, mw = rot.inverted(), obj.matrix_world
    pts = [inv @ (mw @ v.co) for v in obj.data.vertices]
    cx = (max(p.x for p in pts) + min(p.x for p in pts)) / 2.0
    cy = (max(p.y for p in pts) + min(p.y for p in pts)) / 2.0
    d = max(max(abs(p.x - cx) / tx, abs(p.y - cy) / ty) + p.z for p in pts) * 1.05
    cam.location = rot @ Vector((cx, cy, d))
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
    k_sun, k_world = float(OPTS.get("sun", 1)), float(OPTS.get("world", 1))
    for o in bpy.data.objects:
        if o.type == 'LIGHT':
            o.data.energy *= k_sun
    bg = sc.world.node_tree.nodes.get("Background") if sc.world and sc.world.use_nodes else None
    if bg is not None:
        bg.inputs["Strength"].default_value *= k_world
    log("render: %s, %dx%d, sun x%g, world x%g" % (sc.render.engine, *RES, k_sun, k_world))
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
        cam = camera_for(obj, base, VIEWS.get(name))
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
