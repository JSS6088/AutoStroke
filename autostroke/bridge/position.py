"""Bake the object-space position map via Cycles.

Ported from bake_position.py with the two bugs that were tolerable in a script and
are not in an addon fixed: render settings are saved/restored, and the material swap
is exception-safe (including objects that had no material slots at all).
"""

from contextlib import contextmanager

import bpy

from . import images as bi

MAT_NAME = "AUTOSTROKE_bake_%s"
IMG_NAME = "AUTOSTROKE_%s"

# Geometry socket + Vector Transform mode per channel. NORMAL (not POINT/VECTOR) applies
# the inverse-transpose, which is what keeps normals correct under non-uniform scale.
CHANNELS = {
    "position": ("Position", 'POINT'),
    "normal":   ("Normal",   'NORMAL'),
}


@contextmanager
def preserved_bake_settings(scene):
    """Restore every render setting the bake touches, however it exits.

    Forces `use_selected_to_active` off here rather than merely saving it, because it
    is not something bake() ever wants on: that mode bakes FROM every other selected
    object ONTO the active one (a high-poly-to-low-poly workflow), and this function
    selects exactly one object. With it left on -- a scene setting that persists in
    the .blend and is easy to have turned on from unrelated prior work -- Cycles finds
    no source objects and refuses with its own "No valid selected objects", which was
    genuinely confusing to debug: the object IS selected and active, just not in the
    mode this setting demands.
    """
    bk = scene.render.bake
    saved = {
        "engine": scene.render.engine,
        "margin": bk.margin,
        "use_clear": bk.use_clear,
        "use_selected_to_active": bk.use_selected_to_active,
        "normal_space": bk.normal_space,
        "normal_swizzle": (bk.normal_r, bk.normal_g, bk.normal_b),
    }
    has_cycles = hasattr(scene, "cycles")
    if has_cycles:
        saved["device"] = scene.cycles.device
        saved["samples"] = scene.cycles.samples
    scene.render.bake.use_selected_to_active = False
    try:
        yield
    finally:
        scene.render.engine = saved["engine"]
        scene.render.bake.margin = saved["margin"]
        scene.render.bake.use_clear = saved["use_clear"]
        scene.render.bake.use_selected_to_active = saved["use_selected_to_active"]
        bk.normal_space = saved["normal_space"]
        bk.normal_r, bk.normal_g, bk.normal_b = saved["normal_swizzle"]
        if has_cycles:
            scene.cycles.device = saved["device"]
            scene.cycles.samples = saved["samples"]


@contextmanager
def swapped_material(obj, mat):
    """Assign `mat` to every slot, restoring exactly on exit.

    The script version used `if i < len(orig_mats)`, which permanently kept the bake
    material on an object that started with NO slots. Here an added slot is removed.
    """
    orig = [s.material for s in obj.material_slots]
    added_slot = False
    if obj.material_slots:
        for s in obj.material_slots:
            s.material = mat
    else:
        obj.data.materials.append(mat)
        added_slot = True
    try:
        yield
    finally:
        if added_slot:
            obj.data.materials.pop(index=len(obj.data.materials) - 1)
        else:
            for i, s in enumerate(obj.material_slots):
                if i < len(orig):
                    s.material = orig[i]


@contextmanager
def preserved_selection(view_layer):
    """Put the artist's selection and active object back, however the bake exits.

    Cycles bakes the SELECTED object, so bake() deselects everything and selects the one
    object it is baking. Left unrestored, a multi-object bake collapsed the artist's
    selection to whichever object baked last -- and pressing Bake again then silently
    baked that one object instead of the group they had selected.
    """
    selected = [o for o in view_layer.objects if o.select_get()]
    active = view_layer.objects.active
    try:
        yield
    finally:
        for o in view_layer.objects:
            if o.select_get():
                o.select_set(False)
        for o in selected:
            o.select_set(True)
        view_layer.objects.active = active


@contextmanager
def selectable(obj):
    """Guarantee obj can be selected for the bake, whatever its Outliner state.

    ACTIVE and SELECTED are independent Blender flags; validate()/the panel check only
    the former. An object can be active -- and read as "ready" -- while hidden or
    Disable-Selection locked, with no deliberate lock needed: an Outliner click sets
    active regardless of either, and a reopened .blend restores whichever object was
    active when saved, hidden or not.

    Cycles' bake operator filters selected_editable_objects, which silently excludes
    such an object even after select_set(True) appears to succeed -- so without this,
    the bake proceeds with nothing actually selected and fails with Blender's own
    opaque "No valid selected objects", with nothing to tell the artist it is about a
    lock/hide state rather than the selection they can see.
    """
    saved_select = obj.hide_select
    saved_hide = obj.hide_get()
    obj.hide_select = False
    if saved_hide:
        obj.hide_set(False)
    try:
        yield
    finally:
        obj.hide_select = saved_select
        if saved_hide:
            obj.hide_set(True)


@contextmanager
def disabled_geometry_nodes(obj):
    """Bake the surface mesh, not the GN point cloud (which carries no UVs)."""
    disabled = []
    for m in obj.modifiers:
        if m.type == 'NODES' and m.show_render:
            m.show_render = False
            disabled.append(m)
    try:
        yield
    finally:
        for m in disabled:
            m.show_render = True


def _find(nodes, bl_idname):
    return next((n for n in nodes if n.bl_idname == bl_idname), None)


def _linked(nt, a, a_sock, b, b_sock):
    # `==` not `is`: bpy returns a fresh Python wrapper per RNA access, so identity
    # never matches even for the same node. bpy_struct.__eq__ compares the pointer.
    return any(l.from_node == a and l.from_socket.name == a_sock and
               l.to_node == b and l.to_socket.name == b_sock for l in nt.links)


def material_is_current(mat, channel="position"):
    """Structural check, not a name check.

    A material left by an older addon version has the right NAME and the wrong GRAPH;
    reusing it on name alone would silently bake stale data -- which is exactly how the
    world/object-space mix-up stayed hidden for weeks.
    """
    if not mat.use_nodes or mat.node_tree is None:
        return False
    nt = mat.node_tree
    geo = _find(nt.nodes, "ShaderNodeNewGeometry")
    xf = _find(nt.nodes, "ShaderNodeVectorTransform")
    emit = _find(nt.nodes, "ShaderNodeEmission")
    out = _find(nt.nodes, "ShaderNodeOutputMaterial")
    tex = _find(nt.nodes, "ShaderNodeTexImage")
    if not all((geo, xf, emit, out, tex)):
        return False
    sock, vtype = CHANNELS[channel]
    if (xf.vector_type, xf.convert_from, xf.convert_to) != (vtype, 'WORLD', 'OBJECT'):
        return False
    return (_linked(nt, geo, sock, xf, "Vector") and
            _linked(nt, xf, "Vector", emit, "Color") and
            _linked(nt, emit, "Emission", out, "Surface"))


def build_material(mat, channel="position"):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    # Geometry.Position is WORLD space; the GN `position` attribute is OBJECT space.
    # Any object transform then puts seeds and position map in different frames and the
    # baker matches almost nothing. POINT, not VECTOR: we need the translation too.
    sock, vtype = CHANNELS[channel]
    xf = nt.nodes.new("ShaderNodeVectorTransform")
    xf.vector_type, xf.convert_from, xf.convert_to = vtype, 'WORLD', 'OBJECT'
    nt.nodes.new("ShaderNodeTexImage")
    nt.links.new(geo.outputs[sock], xf.inputs["Vector"])
    nt.links.new(xf.outputs["Vector"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def ensure_material(channel="position"):
    name = MAT_NAME % channel
    mat = bpy.data.materials.get(name)
    if mat is None:
        return build_material(bpy.data.materials.new(name), channel)
    if not material_is_current(mat, channel):
        build_material(mat, channel)
    return mat


def ensure_image(res, channel="position"):
    name = IMG_NAME % channel
    img = bpy.data.images.get(name)
    if img is not None and (tuple(img.size) != (res, res) or not img.is_float):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, res, res, float_buffer=True, is_data=True)
    img.colorspace_settings.name = 'Non-Color'
    return img


BAKE_MARGIN = 16
"""Pixels Cycles extends each UV island by. Those margin texels read as valid in the
position map but lie outside every triangle, so core/reach.raster_tri_ids grows triangle
ids over the same distance."""


def bake(obj, res, margin=BAKE_MARGIN, channel="position"):
    """Bake an object-space channel, returned as a TOP-DOWN (res,res,3) array.

    channel="position" gives the UV->3D bridge the baker needs; "normal" gives the true
    surface normal, used so texels no stamp covered fall back to real shading instead of
    black.
    """
    if not obj.data.uv_layers.active:
        raise RuntimeError("%s has no active UV map" % obj.name)

    mat = ensure_material(channel)
    img = ensure_image(res, channel)
    tex = _find(mat.node_tree.nodes, "ShaderNodeTexImage")
    tex.image = img
    mat.node_tree.nodes.active = tex
    _run(obj, mat, margin, channel, 'EMIT')
    return bi.image_to_numpy(img, 3), img


TANGENT_SRC = "AUTOSTROKE_object_space_normal"

NORMAL_SPACE = "autostroke_normal_space"
"""On the stroke normal map's image: the space it was baked in, 'TANGENT' since 0.14.0.
A map without it predates that and is OBJECT space; the material reads this to set its
Normal Map node, so an old bake keeps rendering correctly until it is re-baked."""
NORMAL_UV = "autostroke_normal_uv"
"""On the stroke normal map's image: the UV map whose tangents it was encoded against."""


def to_tangent_space(obj, nrm_obj, margin=BAKE_MARGIN):
    """Re-express an encoded OBJECT-space normal map in TANGENT space, via Cycles.

    `nrm_obj` is TOP-DOWN (H,W,3) holding n*0.5+0.5. Returns (tangent, uv_name): the map
    in Blender's tangent-space encoding (OpenGL, +Y), TOP-DOWN, with Cycles' own margin
    around each UV island, and the UV map it was encoded against.

    Cycles does the conversion rather than numpy because the encoding has to invert
    exactly what the Normal Map node decodes at render time: the MikkTSpace tangent and
    bitangent sign of this UV map, interpolated the way the renderer interpolates them,
    on the render mesh, n-gons and custom normals included. A Normal bake in tangent
    space is that inverse by construction. The UV map is named explicitly on both sides
    (the source lookup here, the Normal Map node in the material), because an Image node
    left unconnected reads the RENDER-active UV map while baking uses the active one.
    """
    uv = obj.data.uv_layers.active
    if uv is None:
        raise RuntimeError("%s has no active UV map" % obj.name)
    h, w = nrm_obj.shape[:2]
    src = bi.numpy_to_image(nrm_obj, TANGENT_SRC)
    out = ensure_image(w, "tangent_normal")

    mat = bpy.data.materials.get(MAT_NAME % "tangent") or \
        bpy.data.materials.new(MAT_NAME % "tangent")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = uv.name
    read = nt.nodes.new("ShaderNodeTexImage")
    # Closest: each texel reads back exactly the value written for it, unblurred
    read.image, read.interpolation = src, 'Closest'
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.space = 'OBJECT'
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    target = nt.nodes.new("ShaderNodeTexImage")
    target.image = out
    nt.links.new(uvn.outputs["UV"], read.inputs["Vector"])
    nt.links.new(read.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], outn.inputs["Surface"])
    nt.nodes.active = target

    _run(obj, mat, margin, "tangent-space normal", 'NORMAL', normal_space='TANGENT')
    return bi.image_to_numpy(out, 3), uv.name


def _run(obj, mat, margin, channel, bake_type, normal_space=None):
    """One Cycles bake of `mat` on `obj` alone, with every setting it touches restored --
    the normal space and swizzle included, which is why they are set in here, inside
    preserved_bake_settings, and not by the caller."""
    scene = bpy.context.scene
    # preserved_selection is outermost so it restores LAST, after selectable() has put
    # the object's own hide/lock state back.
    with preserved_selection(bpy.context.view_layer), preserved_bake_settings(scene), \
            disabled_geometry_nodes(obj), swapped_material(obj, mat), selectable(obj):
        scene.render.engine = 'CYCLES'
        if hasattr(scene, "cycles"):
            scene.cycles.device = 'CPU'
            scene.cycles.samples = 1      # emit and normal bakes need no sampling
        scene.render.bake.margin = margin
        scene.render.bake.use_clear = True
        if normal_space is not None:
            bk = scene.render.bake
            bk.normal_space = normal_space
            bk.normal_r, bk.normal_g, bk.normal_b = 'POS_X', 'POS_Y', 'POS_Z'  # OpenGL (+Y)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # select_set() does not raise on failure -- it silently no-ops -- so it is
        # checked here rather than trusted. Anything past selectable() unlocking
        # hide_select/hidden is something that cannot be fixed transparently (the
        # object excluded from the view layer, or a library-linked object), and
        # deserves a clear reason rather than letting Cycles fail unexplained below.
        if not obj.select_get() or bpy.context.view_layer.objects.active != obj:
            raise RuntimeError(
                "AutoStroke could not select %s to bake it -- it may be excluded "
                "from the active view layer, or linked from a library. Check the "
                "Outliner." % obj.name)
        try:
            bpy.ops.object.bake(type=bake_type)
        except RuntimeError as e:
            raise RuntimeError(
                "AutoStroke's Cycles bake failed baking the %s map for %s: %s"
                % (channel, obj.name, e)) from e
