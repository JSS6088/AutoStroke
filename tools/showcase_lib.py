"""Shared helpers for the showcase scripts (render_showcase.py, render_hero.py).

Importing this has no side effects: it defines functions only. Everything that changes a
scene does so in memory -- none of the showcase scripts ever saves a .blend.
"""

import math
import os

import bpy
from mathutils import Matrix, Quaternion, Vector

TAG = "autostroke_map"        # ops/material.py marks the image nodes it owns with this


# ---------------------------------------------------------------------------
# Viewport context and the bake
# ---------------------------------------------------------------------------

def view3d():
    """(window, area, region) of a 3D viewport: the Bake operator needs one to invoke."""
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                return win, area, next(r for r in area.regions if r.type == 'WINDOW')
    raise RuntimeError("no 3D viewport in this window")


def start_bake(obj, workdir):
    """Select only `obj` and invoke the real AutoStroke Bake (GPU) into `workdir`. The
    caller waits for scene.autostroke.last_report to become non-empty."""
    obj.hide_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    st = bpy.context.scene.autostroke
    st.working_dir = workdir.rstrip(os.sep) + os.sep
    st.bake_device = 'GPU'
    st.last_report = ""
    win, area, region = view3d()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.autostroke.bake('INVOKE_DEFAULT')


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def grey_material():
    mat = bpy.data.materials.new("showcase_grey")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def baked_images(obj):
    """(stroke_normal image, stroke_indirection image) from obj's AutoStroke material."""
    mat = bpy.data.materials.get(obj.name + "_AutoStroke")
    if mat is None:
        return None, None
    found = {n.get(TAG): n.image for n in mat.node_tree.nodes
             if n.bl_idname == "ShaderNodeTexImage" and n.get(TAG)}
    return found.get("stroke_normal"), found.get("stroke_indirection")


def albedo_of(mat):
    """The image feeding a material's Principled Base Color directly, or None."""
    if mat is None or not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            links = n.inputs["Base Color"].links
            if links and links[0].from_node.bl_idname == "ShaderNodeTexImage":
                return links[0].from_node.image
    return None


def _stroke_nodes(nt, suffix):
    """Image nodes reading one of AutoStroke's baked maps, found by IMAGE NAME -- not by
    the node tag, which Blender copies onto any node an artist duplicates from ours."""
    return [n for n in nt.nodes if n.bl_idname == "ShaderNodeTexImage" and n.image
            is not None and n.image.name.endswith(suffix)]


def without_strokes(mat, grey):
    """`mat` without AutoStroke, as a temporary copy -- or `mat` itself if it never read
    AutoStroke's maps. Every node the artist made is kept; only the maps' influence is
    undone:
      - textures read THROUGH the indirection pointer (colour, AO, roughness...) go back
        to the mesh's own UVs: the pointer link into their Vector input is removed;
      - anything else fed by the indirection map (the per-stroke tone, ramps driven by
        it) gets a neutral mid value, 0.5, instead of a random per stroke;
      - the stroke normal map is unplugged. If an artist's own Normal Map / Bump node is
        left with a live input but no output, it goes back into the BSDF's Normal (that
        is where wire_strokes found it); otherwise the mesh's own normals show."""
    if mat is None:
        return grey
    if not mat.use_nodes or not (_stroke_nodes(mat.node_tree, "_stroke_indirection")
                                 or _stroke_nodes(mat.node_tree, "_stroke_normal")):
        return mat
    c = mat.copy()
    nt = c.node_tree
    for n in _stroke_nodes(nt, "_stroke_indirection"):
        for link in list(n.outputs["Color"].links):
            to, to_node = link.to_socket, link.to_node
            nt.links.remove(link)
            if not (to_node.bl_idname == "ShaderNodeTexImage" and to.identifier == "Vector"):
                try:
                    to.default_value = (0.5,) * len(to.default_value)
                except (AttributeError, TypeError):
                    try:
                        to.default_value = 0.5
                    except (AttributeError, TypeError):
                        pass
    for n in _stroke_nodes(nt, "_stroke_normal"):
        for link in list(n.outputs["Color"].links):
            nm = link.to_node
            nt.links.remove(link)
            # an object-space Normal Map with nothing plugged in outputs a fixed normal,
            # which would be wrong everywhere: take it out of the shader entirely
            if nm.bl_idname == "ShaderNodeNormalMap":
                for out in list(nm.outputs["Normal"].links):
                    nt.links.remove(out)
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is not None and not bsdf.inputs["Normal"].links:
        for n in nt.nodes:
            live = any(i.links for i in n.inputs)
            if n.bl_idname in ("ShaderNodeNormalMap", "ShaderNodeBump") and live \
                    and not n.outputs["Normal"].links:
                nt.links.new(n.outputs["Normal"], bsdf.inputs["Normal"])
                break
    return c


def plain_textured(albedo_img):
    """The model as it would look WITHOUT AutoStroke: its albedo on its own UVs."""
    mat = bpy.data.materials.new("showcase_plain_%s" % albedo_img.name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    alb = nt.nodes.new("ShaderNodeTexImage")
    alb.image = albedo_img
    nt.links.new(alb.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def painterly_colour(src_mat, nrm_img, ind_img, albedo_img):
    """The "any texture through the pointer" feature, built by hand for the video.

    Base Color = the model's own albedo image read at the indirection map's (R, G) -- one
    flat colour per stroke -- times the stroke tone (B), remapped the way ops/material.py
    does. Normal = the stroke normal map, object space. Roughness and metallic copy the
    source material's plain values where it has them."""
    mat = bpy.data.materials.new("showcase_colour_%s" % (src_mat.name if src_mat else "grey"))
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")

    ind = nt.nodes.new("ShaderNodeTexImage")
    ind.image, ind.interpolation = ind_img, 'Closest'     # pointers must never be blended
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(ind.outputs["Color"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(sep.outputs["Y"], comb.inputs["Y"])

    alb = nt.nodes.new("ShaderNodeTexImage")
    alb.image = albedo_img
    nt.links.new(comb.outputs["Vector"], alb.inputs["Vector"])

    tone = nt.nodes.new("ShaderNodeMix")
    tone.data_type = 'RGBA'
    tone.inputs[6].default_value = (0.85, 0.85, 0.85, 1.0)
    tone.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(sep.outputs["Z"], tone.inputs["Factor"])
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type, mul.blend_type = 'RGBA', 'MULTIPLY'
    mul.inputs["Factor"].default_value = 1.0
    nt.links.new(alb.outputs["Color"], mul.inputs[6])
    nt.links.new(tone.outputs[2], mul.inputs[7])
    nt.links.new(mul.outputs[2], bsdf.inputs["Base Color"])

    nrm = nt.nodes.new("ShaderNodeTexImage")
    nrm.image, nrm.interpolation = nrm_img, 'Linear'
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.space = 'OBJECT'
    nt.links.new(nrm.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    src = next((n for n in src_mat.node_tree.nodes
                if n.bl_idname == "ShaderNodeBsdfPrincipled"), None) \
        if src_mat is not None and src_mat.use_nodes else None
    for name in ("Roughness", "Metallic"):
        if src is not None and not src.inputs[name].links:
            bsdf.inputs[name].default_value = src.inputs[name].default_value
    return mat


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def camera_for(obj, base, view=None, spin=False, pivot=None):
    """A copy of `base` looking along base's own direction, or from `view` =
    (dx, dy, dz[, roll_deg]) towards the object, framed tightly on the object's vertices as
    projected onto the image plane (5% margin), for the scene's current resolution.

    With `spin`, the framing holds for the object turned to ANY angle about the world Z
    axis through `pivot` (default: its origin) -- what a turntable needs, so nothing
    leaves frame mid-turn."""
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
    if spin:
        cx, cy, d = _spin_fit(obj, inv, tx, ty, pivot)
    else:
        world = [mw @ v.co for v in obj.data.vertices]
        pts = [inv @ p for p in world]
        cx = (max(p.x for p in pts) + min(p.x for p in pts)) / 2.0
        cy = (max(p.y for p in pts) + min(p.y for p in pts)) / 2.0
        d = max(max(abs(p.x - cx) / tx, abs(p.y - cy) / ty) + p.z for p in pts) * 1.05
    cam.location = rot @ Vector((cx, cy, d))
    return cam


def _spin_fit(obj, inv, tx, ty, pivot=None):
    """camera_for's fit over every turn of 10 deg about world Z through `pivot` (default
    the object's origin) -- in numpy, since that is 36 copies of every vertex."""
    import numpy as np
    n = len(obj.data.vertices)
    co = np.empty(n * 3, np.float64)
    obj.data.vertices.foreach_get("co", co)
    mw = np.array(obj.matrix_world)
    # Blender's numpy 1.26 on Apple's Accelerate raises spurious divide/overflow flags
    # from matmul even when every result is finite; hush them rather than alarm the user.
    with np.errstate(all="ignore"):
        world = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        o = np.array(pivot, np.float64) if pivot is not None else mw[:3, 3]
        a = np.radians(np.arange(0, 360, 10))[:, None]
        x, y = world[:, 0] - o[0], world[:, 1] - o[1]
        turned = np.stack([o[0] + np.cos(a) * x - np.sin(a) * y,
                           o[1] + np.sin(a) * x + np.cos(a) * y,
                           np.broadcast_to(world[:, 2], (len(a), n))], -1).reshape(-1, 3)
        pts = turned @ np.array(inv.to_matrix()).T
    if not np.isfinite(pts).all():
        raise ValueError("%s: non-finite vertex positions, can't frame it" % obj.name)
    cx = (pts[:, 0].max() + pts[:, 0].min()) / 2.0
    cy = (pts[:, 1].max() + pts[:, 1].min()) / 2.0
    d = float((np.maximum(np.abs(pts[:, 0] - cx) / tx,
                          np.abs(pts[:, 1] - cy) / ty) + pts[:, 2]).max()) * 1.05
    return float(cx), float(cy), d


# ---------------------------------------------------------------------------
# Scene: models, studio, turntable
# ---------------------------------------------------------------------------

IMPORTERS = {
    ".glb": lambda f: bpy.ops.import_scene.gltf(filepath=f),
    ".gltf": lambda f: bpy.ops.import_scene.gltf(filepath=f),
    ".fbx": lambda f: bpy.ops.import_scene.fbx(filepath=f),
    ".obj": lambda f: bpy.ops.wm.obj_import(filepath=f),
}


def import_model(path, object_name=None, name="model"):
    """Import a model file as ONE mesh object: centred on the origin, scaled to a 2 m
    bounding sphere, transforms applied, modifiers (armatures included) baked in, and
    pieces joined so the reach rule sees a single mesh. Raises with a clear message on
    anything the bake cannot use."""
    path = os.path.expanduser(path)
    ext = os.path.splitext(path)[1].lower()
    if not os.path.exists(path):
        raise RuntimeError("%s: file not found: %s" % (name, path))
    before = set(bpy.data.objects)
    if ext == ".blend":
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = [n for n in src.objects if object_name in (None, n)]
        for o in dst.objects:
            if o is not None:
                bpy.context.scene.collection.objects.link(o)
    elif ext in IMPORTERS:
        IMPORTERS[ext](path)
    else:
        raise RuntimeError("%s: unsupported file type %s (use .glb, .gltf, .fbx, .obj "
                           "or .blend)" % (name, ext))
    new = [o for o in bpy.data.objects if o not in before]
    picked = [o for o in new if object_name is None or o.name == object_name
              or o.name.startswith(object_name + ".")]
    meshes = [o for o in picked if o.type == 'MESH']
    if not meshes:
        raise RuntimeError("%s: no mesh objects in %s%s" % (
            name, path, " named %r" % object_name if object_name else ""))

    # bake every modifier and parent transform into plain world-space meshes
    deps = bpy.context.evaluated_depsgraph_get()
    flat = []
    for o in meshes:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(deps), depsgraph=deps)
        me.transform(o.matrix_world)
        flat_o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(flat_o)
        flat.append(flat_o)
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    if len(flat) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in flat:
            o.select_set(True)
        bpy.context.view_layer.objects.active = flat[0]
        win, area, region = view3d()
        with bpy.context.temp_override(window=win, area=area, region=region,
                                       active_object=flat[0], selected_editable_objects=flat):
            bpy.ops.object.join()
    obj = flat[0]
    obj.name = name
    if not obj.data.uv_layers:
        raise RuntimeError("%s: the mesh has no UV map -- unwrap it first" % name)

    # centre on the origin and scale to a 2 m bounding sphere, applied to the mesh
    vs = [v.co for v in obj.data.vertices]
    lo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    hi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    c = (lo + hi) / 2.0
    radius = max((v - c).length for v in vs) or 1.0
    obj.data.transform(Matrix.Scale(1.0 / radius, 4) @ Matrix.Translation(-c))
    obj.matrix_world = Matrix.Identity(4)
    obj.data.update()
    return obj


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def studio(resolution, view=(-0.7, -1.0, -0.2)):
    """The same look for every shot: EEVEE, AgX, a transparent background (drop any
    backdrop in the editor), and three sun lights fixed relative to the VIEW -- a raking
    key ~60 deg off it (strokes are relief, so they read where light grazes), a soft fill
    opposite, and a rim from behind. Returns the base camera."""
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x, sc.render.resolution_y = resolution
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'AgX'
    if sc.world is None:
        sc.world = bpy.data.worlds.new("showcase_world")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.05, 0.05, 0.055, 1.0)
        bg.inputs["Strength"].default_value = 0.6

    cam_data = bpy.data.cameras.new("showcase_cam")
    cam_data.lens = 50.0
    base = bpy.data.objects.new("showcase_cam", cam_data)
    sc.collection.objects.link(base)
    fwd = (-Vector(view[:3])).normalized()
    base.rotation_mode = 'QUATERNION'
    base.rotation_quaternion = fwd.to_track_quat('-Z', 'Y')
    sc.camera = base

    right = fwd.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(fwd).normalized()
    for label, direction, energy in (
            ("key", (fwd * 0.5 - right * 0.8 + up * 0.6), 4.0),     # raking, from the left
            ("fill", (fwd * 0.6 + right * 0.7 + up * 0.1), 0.8),
            ("rim", (-fwd * 0.9 + up * 0.5 + right * 0.3), 2.5)):
        light = bpy.data.lights.new("showcase_%s" % label, 'SUN')
        light.energy = energy
        lo = bpy.data.objects.new("showcase_%s" % label, light)
        sc.collection.objects.link(lo)
        # a sun shines along its local -Z: point it along `direction` (light travel)
        lo.rotation_mode = 'QUATERNION'
        lo.rotation_quaternion = Vector(direction).normalized().to_track_quat('-Z', 'Y')
    return base


def bbox_center(obj):
    """World-space centre of an object's bounding box."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return sum(corners, Vector()) / 8.0


def turntable_pivot(obj, frames):
    """Turn an object that is already placed and rotated in a scene about world Z through
    its bounding-box centre, without touching its own transform: it is parented (keeping
    its world matrix) to a new empty at that centre, and the EMPTY turns 360 deg over
    `frames`, linear, looping. Returns the pivot location."""
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, frames
    sc.render.fps = 30
    c = bbox_center(obj)
    pivot = bpy.data.objects.new("showcase_pivot_%s" % obj.name, None)
    sc.collection.objects.link(pivot)
    pivot.location = c
    bpy.context.view_layer.update()
    mw = obj.matrix_world.copy()
    obj.parent = pivot
    obj.matrix_parent_inverse = pivot.matrix_world.inverted()
    obj.matrix_world = mw
    pivot.rotation_mode = 'XYZ'
    pivot.keyframe_insert("rotation_euler", index=2, frame=1)
    pivot.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    pivot.keyframe_insert("rotation_euler", index=2, frame=frames + 1)
    _linear_everywhere(pivot)
    sc.frame_set(1)
    return c


def relink_missing(search_dirs):
    """Point image datablocks whose file is missing at a same-named file found under any of
    `search_dirs` (in memory only). Returns [(image name, new path)] and the still-missing
    list."""
    found, missing = [], []
    index = {}
    for root in search_dirs:
        for dirpath, _dirs, files in os.walk(os.path.expanduser(root)):
            for f in files:
                index.setdefault(f, os.path.join(dirpath, f))
    for im in bpy.data.images:
        if im.source != 'FILE' or im.packed_file:
            continue
        if os.path.exists(bpy.path.abspath(im.filepath)):
            continue
        hit = index.get(os.path.basename(bpy.path.abspath(im.filepath)))
        if hit:
            im.filepath = hit
            im.reload()
            found.append((im.name, hit))
        else:
            missing.append(im.name)
    return found, missing


def _linear_everywhere(obj):
    """Blender 5 actions keep fcurves in layered channelbags; set LINEAR on all of them."""
    act = obj.animation_data.action if obj.animation_data else None
    if act is None:
        return
    curves = list(getattr(act, "fcurves", []) or [])
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                curves.extend(bag.fcurves)
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'


def render_sequence(folder):
    """Render the scene's frame range as PNG frames folder/0001.png ..."""
    os.makedirs(folder, exist_ok=True)
    bpy.context.scene.render.filepath = os.path.join(folder, "####")
    bpy.ops.render.render(animation=True)
