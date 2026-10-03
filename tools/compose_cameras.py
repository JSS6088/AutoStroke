"""Place a starting camera per model, for you to compose the hero shots by hand.

    Blender ~/Desktop/Showcase.blend --python tools/compose_cameras.py [-- --only a,b]

Opens your file in a normal Blender window and adds, for every renderable mesh that
doesn't have one yet, a camera named CAM_<object> in a "Hero Cameras" collection. Each
is framed the way render_scene.py would auto-frame it: a raised three-quarter view that
keeps the whole model in shot through a full 360 deg turn about its centre, at 1920x1080.

Nothing is saved for you. Compose each camera, then save the file yourself (Ctrl+S);
render_scene.py uses CAM_<object> when it renders that object. Cameras that already
exist are never moved, so re-running this only fills in models you added since.

Composing a camera:
  1. Select the model, press Numpad / (Local View) to see it on its own.
  2. Select its CAM_ camera, Ctrl+Numpad 0 to look through it.
  3. N panel > View > Lock Camera to View, then orbit/pan/zoom to compose.
  4. The model turns about its centre in the render, so leave room for its widest side:
     check with a quick  render_scene.py -- <out> <name> --frames 12  and look at the
     contact sheet (tools/encode_hero.sh).
"""

import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import showcase_lib as lib                      # noqa: E402

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ONLY = set(filter(None, (ARGV[ARGV.index("--only") + 1] if "--only" in ARGV else "").split(",")))
RESOLUTION = (1920, 1080)
VIEW = (-0.7, -1.0, 0.3)


def main():
    sc = bpy.context.scene
    coll = bpy.data.collections.get("Hero Cameras")
    if coll is None:
        coll = bpy.data.collections.new("Hero Cameras")
        sc.collection.children.link(coll)
    base_data = bpy.data.cameras.new("compose_base")
    base_data.lens = 50.0
    base = bpy.data.objects.new("compose_base", base_data)
    sc.collection.objects.link(base)
    res = (sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage)
    made, kept = [], []
    for obj in [o for o in sc.objects if o.type == 'MESH' and not o.hide_render
                and (not ONLY or o.name in ONLY)]:
        name = "CAM_%s" % obj.name
        if bpy.data.objects.get(name) is not None:
            kept.append(name)
            continue
        sc.render.resolution_x, sc.render.resolution_y = RESOLUTION
        sc.render.resolution_percentage = 100
        cam = lib.camera_for(obj, base, VIEW, spin=True, pivot=lib.bbox_center(obj))
        cam.name = cam.data.name = name
        for c in list(cam.users_collection):
            c.objects.unlink(cam)
        coll.objects.link(cam)
        made.append(name)
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = res
    bpy.data.objects.remove(base, do_unlink=True)
    bpy.data.cameras.remove(base_data)
    print("\nAutoStroke compose_cameras: added %d camera(s)%s; kept %d existing."
          % (len(made), (": " + ", ".join(made)) if made else "", len(kept)))
    print("Compose them, then save the file yourself (Ctrl+S). Nothing was saved.\n")


def tick():
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    return None


bpy.app.timers.register(tick, first_interval=0.5)
