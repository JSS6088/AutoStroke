"""Raw shots for the hero showcase video: turntables of each model, plain and painterly.

For every model in tools/showcase/shots.json it imports the file, bakes it with the real
AutoStroke Bake (GPU), and renders 5 s looping turntables with transparent backgrounds:

  <out>/<name>/<aspect>/before/####.png     the model as it came (or grey, if untextured)
  <out>/<name>/<aspect>/after/####.png      painted: the model's own albedo broken into
                                            flat per-stroke colour, or the grey stroke
                                            normal material when it has no albedo
  <out>/<name>/<aspect>/brush_<set>/####.png  the same, re-baked with each extra brush set
  <out>/report.json                         bake times, strokes and coverage per bake

tools/encode_hero.sh then turns the frame folders into ProRes 4444 (alpha) for editing,
H.264 previews, the before->after wipe for hero models, and a contact sheet.

Runs in a real Blender window (the GPU bake and EEVEE want a GPU context), in a new empty
scene. It never saves any .blend, and bakes into a temporary folder.

    Blender --factory-startup --python tools/render_hero.py -- <out_dir>
            [--shots tools/showcase/shots.json] [--only name,name] [--aspect 16x9|4x5|both]
            [--frames 150] [--check]

--check imports every model and validates the entries -- file, UVs, albedo, brush sets --
then prints the plan and quits without baking or rendering.
"""

import json
import os
import sys
import tempfile
import time
import traceback

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import showcase_lib as lib                      # noqa: E402

ASPECTS = {"16x9": (1920, 1080), "4x5": (1080, 1350)}

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _opt(flag, default=None):
    return ARGV[ARGV.index(flag) + 1] if flag in ARGV else default


OUT = os.path.abspath(os.path.expanduser(ARGV[0])) if ARGV and not ARGV[0].startswith("--") \
    else os.path.join(tempfile.gettempdir(), "autostroke_hero")
SHOTS = os.path.expanduser(_opt("--shots", os.path.join(HERE, "showcase", "shots.json")))
ONLY = set(filter(None, (_opt("--only", "") or "").split(",")))
ASPECT_SEL = _opt("--aspect", "both")
FRAMES = int(_opt("--frames", "150"))
CHECK = "--check" in ARGV
lines = []


def log(s=""):
    lines.append(str(s))
    print(s)


def finish():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "hero_log.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os._exit(0)


# ---------------------------------------------------------------------------
# shots.json
# ---------------------------------------------------------------------------

DEFAULTS = dict(object=None, kind="", strokes=5000, stroke_size=None, brush="standard",
                view=[-0.7, -1.0, -0.2], albedo="auto", hero=False, brushes=[])


def brush_spec(spec):
    """(brush_dir, brush_set) for a shipped set name ("rough") or a folder of images."""
    from autostroke.bridge import brushes as bb
    if os.sep in spec or spec.startswith("~"):
        d = os.path.abspath(os.path.expanduser(spec))
        imgs = [f for f in os.listdir(d)] if os.path.isdir(d) else []
        if not any(f.lower().endswith(bb.EXTS) for f in imgs):
            raise RuntimeError("brush folder %s holds no images" % d)
        return os.path.dirname(d), os.path.basename(d)
    shipped = bb.sets(bb.default_root())
    if spec not in shipped:
        raise RuntimeError("unknown brush set %r (shipped: %s; or give a folder path)"
                           % (spec, ", ".join(shipped)))
    return "", spec


def load_shots():
    if not os.path.exists(SHOTS):
        raise RuntimeError("no shot list at %s -- copy tools/showcase/shots.example.json "
                           "there and point it at your models" % SHOTS)
    with open(SHOTS) as f:
        data = json.load(f)
    out = []
    for i, raw in enumerate(data.get("models", [])):
        e = dict(DEFAULTS, **raw)
        if "name" not in e or "file" not in e:
            raise RuntimeError("model #%d needs a 'name' and a 'file'" % (i + 1))
        if ONLY and e["name"] not in ONLY:
            continue
        for b in [e["brush"]] + list(e["brushes"]):
            brush_spec(b)
        out.append(e)
    if not out:
        raise RuntimeError("no models to render%s" % (" matching --only" if ONLY else ""))
    return out


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------

def set_look(e, brush):
    from autostroke.bridge import brushes as bb
    st = bpy.context.scene.autostroke
    st.target_strokes = int(e["strokes"])
    if e["stroke_size"] is not None:
        st.stroke_size = float(e["stroke_size"])
    st.resolution = '2048'
    d, name = brush_spec(brush)
    st.brush_dir = d
    bb.refresh()
    st.brush_set = name


def albedos_for(e, obj, orig):
    """Per material slot, the image to paint with (or None for grey)."""
    if not e["albedo"]:
        return [None] * max(len(orig), 1)
    if e["albedo"] != "auto":
        img = bpy.data.images.load(os.path.expanduser(e["albedo"]), check_existing=True)
        return [img] * max(len(orig), 1)
    return [lib.albedo_of(m) for m in orig] or [None]


def assign(obj, mats):
    if not obj.material_slots:
        obj.data.materials.append(mats[0])
    for s, m in zip(obj.material_slots, mats):
        s.material = m


def bake(obj, workdir):
    """Generator: run the real Bake and wait for it. Returns (seconds, report text)."""
    lib.start_bake(obj, workdir)
    st = bpy.context.scene.autostroke
    t0 = time.time()
    while not st.last_report:
        if time.time() - t0 > 900:
            raise RuntimeError("bake timed out")
        yield
    report = st.last_report.replace("\n", " | ")
    if None in lib.baked_images(obj):
        raise RuntimeError("bake produced no maps: %s" % report)
    return time.time() - t0, report


def model(e, report):
    name = e["name"]
    lib.clear_scene()
    obj = lib.import_model(e["file"], e["object"], name)
    orig = [s.material for s in obj.material_slots]
    albs = albedos_for(e, obj, orig)
    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    log("%s: %s triangles, %d material slot(s), albedo: %s, brushes: %s"
        % (name, "{:,}".format(tris), len(orig),
           ", ".join(sorted({a.name for a in albs if a})) or "none (grey)",
           ", ".join([e["brush"]] + list(e["brushes"]))))
    if CHECK:
        return

    aspects = list(ASPECTS) if ASPECT_SEL == "both" else [ASPECT_SEL]
    base = lib.studio(ASPECTS[aspects[0]], e["view"])      # lights once per model
    cam = None
    grey = lib.grey_material()
    # the turn is carried by an empty at the model's centre, so the model keeps whatever
    # rotation it has; the camera is fitted around that same centre
    pivot = lib.turntable_pivot(obj, FRAMES)
    rec = report.setdefault(name, {"bakes": []})
    for bi, brush in enumerate([e["brush"]] + list(e["brushes"])):
        set_look(e, brush)
        work = tempfile.mkdtemp(prefix="autostroke_hero_")
        secs, text = yield from bake(obj, work)
        rec["bakes"].append(dict(brush=brush, seconds=round(secs, 2), report=text))
        log("  baked with %s in %.1fs: %s" % (brush, secs, text))
        nrm, ind = lib.baked_images(obj)
        stroke_mat = bpy.data.materials.get(obj.name + "_AutoStroke")
        after = [lib.painterly_colour(m, nrm, ind, a) if a is not None else stroke_mat
                 for m, a in zip(orig or [None], albs)]
        if e["albedo"] == "auto" and any(albs):
            before = orig                      # the model's own materials, as imported
        else:
            before = [lib.plain_textured(a) if a is not None else grey
                      for a in albs]
        shot = "after" if bi == 0 else "brush_%s" % os.path.basename(brush.rstrip(os.sep))
        for asp in aspects:
            sc = bpy.context.scene
            sc.render.resolution_x, sc.render.resolution_y = ASPECTS[asp]
            if cam is not None:
                bpy.data.objects.remove(cam, do_unlink=True)
            sc.frame_set(1)
            cam = lib.camera_for(obj, base, e["view"], spin=True, pivot=pivot)
            sc.camera = cam
            folder = os.path.join(OUT, name, asp)
            if bi == 0:
                assign(obj, before)
                lib.render_sequence(os.path.join(folder, "before"))
            assign(obj, after)
            lib.render_sequence(os.path.join(folder, shot))
            log("  rendered %s %s (%d frames)" % (asp, "before + " + shot if bi == 0 else shot,
                                                 FRAMES))
            yield
    rec["hero"] = bool(e["hero"])


def job():
    sys.path.insert(0, REPO)
    import autostroke
    autostroke.register()
    if ASPECT_SEL != "both" and ASPECT_SEL not in ASPECTS:
        raise RuntimeError("--aspect must be both, %s" % " or ".join(ASPECTS))
    entries = load_shots()
    log("%s %d model(s) from %s -> %s"
        % ("checking" if CHECK else "rendering", len(entries), SHOTS, OUT))
    report = {}
    for e in entries:
        try:
            yield from model(e, report)
        except Exception as ex:                       # one bad model must not stop the rest
            log("%s: FAILED -- %s" % (e["name"], ex))
            if not isinstance(ex, RuntimeError):
                log(traceback.format_exc())
            report.setdefault(e["name"], {})["error"] = str(ex)
    if not CHECK:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
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
