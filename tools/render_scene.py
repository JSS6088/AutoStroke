"""Hero-video turntable of ONE model you have already baked and look-dev'd, in your scene.

Open your showcase .blend with this script and name the object to render. It changes
nothing about the look: your lights, world, render engine, colour management and
materials are used as they are. It only (in memory, never saved):
  - hides every other mesh from the render;
  - turns the model 360 deg about its own bounding-box centre (the lights stay put, so
    they sweep across the strokes);
  - frames it with YOUR camera if the file has one named CAM_<object> -- tools/
    compose_cameras.py places starting ones -- otherwise with a camera fitted to the turn;
  - switches off the pre-0.10 AutoStroke geometry-nodes modifier (ScatterSeeds...) if the
    model still has one, since it replaces the surface with seed geometry;
  - relinks textures whose file is missing to a same-named file under --search.

    Blender Showcase.blend --python tools/render_scene.py -- <out_dir> <object>
            [--wipe] [--view dx,dy,dz] [--frames 150] [--search ~/Downloads]
            [--transparent] [--no-before] [--check]

Run it once per model. With no object named, it lists the models it can render and
renders nothing.

It writes 16:9 (1920x1080) frames to <out>/<object>/16x9/after/####.png (your materials)
and before/####.png: each material with AutoStroke's influence undone (your look-dev
without the strokes). --wipe marks the object in <out>/report.json -- which each run adds
to rather than replaces -- so tools/encode_hero.sh makes its before->after wipe.

--check shows what would be rendered (before availability, relinked and still missing
textures) without rendering.
"""

import json
import os
import sys
import traceback

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import showcase_lib as lib                      # noqa: E402

RESOLUTION = (1920, 1080)
ASPECT = "16x9"                                 # the folder encode_hero.sh expects

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _opt(flag, default=None):
    return ARGV[ARGV.index(flag) + 1] if flag in ARGV else default


def _opts(flag):
    return [ARGV[i + 1] for i, a in enumerate(ARGV) if a == flag and i + 1 < len(ARGV)]


_VALUED = {"--view", "--frames", "--search"}
_POS = [a for i, a in enumerate(ARGV)
        if not a.startswith("--") and (i == 0 or ARGV[i - 1] not in _VALUED)]
OUT = os.path.abspath(os.path.expanduser(_POS[0])) if _POS else os.path.expanduser("~/HeroShots")
OBJECT = _POS[1] if len(_POS) > 1 else None
WIPE = "--wipe" in ARGV
VIEW = [float(x) for x in _opt("--view").split(",")] if _opt("--view") else [-0.7, -1.0, 0.3]
FRAMES = int(_opt("--frames", "150"))
SEARCH = _opts("--search") or ["~/Downloads"]
CHECK = "--check" in ARGV
lines = []


def log(s=""):
    lines.append(str(s))
    print(s)


def finish():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "scene_log.txt"), "a") as f:
        f.write("\n".join(lines) + "\n")
    os._exit(0)


def legacy_gn(obj):
    return [m for m in obj.modifiers if m.type == 'NODES' and m.show_render and (
        m.name == "AutoStroke" or (m.node_group and m.node_group.name.startswith("ScatterSeeds")))]


def before_materials(obj, grey):
    """Per slot, the look without AutoStroke: each slot's current material with the maps'
    influence undone (showcase_lib.without_strokes) -- your look-dev, minus the strokes.
    None when no slot reads AutoStroke's maps at all, since there is nothing to compare."""
    out = [lib.without_strokes(s.material, grey) for s in obj.material_slots]
    return out if any(b is not s.material for b, s in zip(out, obj.material_slots)) else None


def update_report(name, entry):
    path = os.path.join(OUT, "report.json")
    try:
        with open(path) as f:
            report = json.load(f)
    except (OSError, ValueError):
        report = {}
    report[name] = entry
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def job():
    sc = bpy.context.scene
    meshes = [o for o in sc.objects if o.type == 'MESH' and not o.hide_render]
    if OBJECT is None or OBJECT not in {o.name for o in meshes}:
        log(("No object named %r in this scene. " % OBJECT if OBJECT else
             "Name one object to render: -- <out_dir> <object>. ") +
            "Renderable models:\n  " + "\n  ".join(sorted(o.name for o in meshes)))
        return
    obj = bpy.data.objects[OBJECT]

    found, missing = lib.relink_missing(SEARCH)
    for name, path in found:
        log("relinked texture %s -> %s" % (name, path))
    for name in missing:
        log("WARNING: texture %s is still missing (renders pink)" % name)

    grey = lib.grey_material()
    gn = legacy_gn(obj)
    before = None if "--no-before" in ARGV else before_materials(obj, grey)
    mine = bpy.data.objects.get("CAM_%s" % obj.name)
    mine = mine if mine is not None and mine.type == 'CAMERA' else None
    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    log("%s %s: %s triangles, %d slot(s), camera: %s, before: %s%s" % (
        "checking" if CHECK else "rendering", obj.name, "{:,}".format(tris),
        len(obj.material_slots), mine.name if mine else "automatic",
        "yes" if before else "not available",
        "; legacy GN modifier '%s' switched off" % gn[0].name if gn else ""))
    if CHECK:
        return

    for m in gn:
        m.show_render = False
    for o in sc.objects:
        if o.type == 'MESH':
            o.hide_render = o is not obj
    if "--transparent" in ARGV:
        sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y = RESOLUTION
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA' if sc.render.film_transparent else 'RGB'

    pivot = lib.turntable_pivot(obj, FRAMES)
    if mine is not None:
        sc.camera = mine
    else:
        base_data = bpy.data.cameras.new("showcase_cam")
        base_data.lens = 50.0
        base = bpy.data.objects.new("showcase_cam", base_data)
        sc.collection.objects.link(base)
        sc.camera = lib.camera_for(obj, base, VIEW, spin=True, pivot=pivot)

    folder = os.path.join(OUT, obj.name, ASPECT)
    after = [s.material for s in obj.material_slots]
    lib.render_sequence(os.path.join(folder, "after"))
    yield
    if before:
        for s, m in zip(obj.material_slots, before):
            s.material = m
        lib.render_sequence(os.path.join(folder, "before"))
        for s, m in zip(obj.material_slots, after):
            s.material = m
    log("  rendered after%s (%d frames) -> %s" % (" + before" if before else "", FRAMES, folder))
    update_report(obj.name, {"hero": WIPE and bool(before)})
    if WIPE and not before:
        log("  no wipe: this model has no 'before' to wipe from")
    log("done")


_gen = None


def tick():
    global _gen
    try:
        if _gen is None:
            _gen = job()
        next(_gen)
        return 0.25
    except StopIteration:
        pass
    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc())
    finish()


bpy.app.timers.register(tick, first_interval=1.0)
