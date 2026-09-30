"""Build the showcase material from the baked maps.

Deliberately NOT the same material as the position bake -- they are opposites. The bake
material is a transient position emitter; this one samples THROUGH the indirection map.

The normal channel needs no indirection lookup at all: <stem>_debug_normal is already
flat-per-cell OBJECT-space normals, so it feeds a Normal Map node set to OBJECT and the
painterly shading appears with no artist-supplied art. Indirection sampling earns its
keep for the channels the artist does supply.
"""

import bpy

from .bake import safe_name

MAT_SUFFIX = "_AutoStroke"


TAG = "autostroke_map"          # marks the image nodes this addon owns

LEGACY_KEYS = {"debug_normal": "stroke_normal", "indirection": "stroke_indirection"}
"""Maps renamed in 0.7.0. A material built before then carries the old tag, and without
this its nodes would look untagged: build() would take that as "nothing of ours here" and
call _build_fresh, which clears the node tree and takes any hand-wiring with it. Accepting
the old tag turns the rename into a re-point instead."""

INTERPOLATION = {
    # Closest is NON-NEGOTIABLE on the indirection map: it stores UV pointers, and
    # interpolating between two pointers yields a third that points nowhere.
    "stroke_indirection": 'Closest',
    # The normal map is NOT read through the indirection -- it is sampled with the
    # mesh's own UVs (see the module docstring), so nothing in it is a pointer and
    # Closest only buys visible texel stair-stepping along every cell edge. Linear
    # softens each stroke edge over ~2 texels; Cubic (a B-spline in Blender) spreads it
    # over ~4 and read as blurry next to the per-pixel live preview. A map that IS read
    # through the indirection belongs above, not here.
    "stroke_normal": 'Linear',
}


def _img(stem, key):
    return bpy.data.images.get("%s_%s" % (stem, key))


def _tagged(nt, key):
    """The image node this addon owns for `key`, or None.

    Found by custom property, never by type or position -- otherwise a rebuild would
    hijack a texture node the artist added themselves.
    """
    legacy = [old for old, new in LEGACY_KEYS.items() if new == key]
    for n in nt.nodes:
        if n.bl_idname != "ShaderNodeTexImage":
            continue
        tag = n.get(TAG)
        if tag == key:
            return n
        if tag in legacy:
            n[TAG] = key            # migrate in place; the node and its links survive
            return n
    return None


def _configure(node, img, key):
    node[TAG] = key
    node.image = img
    # Default to Closest for anything unlisted: a map whose meaning we do not know is
    # safer unfiltered than silently blended.
    node.interpolation = INTERPOLATION.get(key, 'Closest')
    img.colorspace_settings.name = 'Non-Color'


def _build_fresh(mat, nrm_img, ind_img):
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial");   out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled");  bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    tex = nt.nodes.new("ShaderNodeTexImage"); tex.location = (-300, -100)
    _configure(tex, nrm_img, "stroke_normal")
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.space = 'OBJECT'             # the baked normals ARE object-space
    nmap.location = (0, -100)
    nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    if ind_img is not None:
        ind = nt.nodes.new("ShaderNodeTexImage"); ind.location = (-600, 250)
        _configure(ind, ind_img, "stroke_indirection")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-350, 250)
        nt.links.new(ind.outputs["Color"], sep.inputs["Vector"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = 'RGBA'; mix.location = (0, 250)
        # per-stroke luminance remapped into 0.85..1.0 rather than 0.5..1.0: the B
        # channel is a flat random per cell, so a wide range reads as noise instead of
        # as pigment variation
        mix.inputs["Factor"].default_value = 0.25
        mix.inputs[6].default_value = (0.85, 0.85, 0.85, 1.0)
        mix.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)
        nt.links.new(sep.outputs["Z"], mix.inputs["Factor"])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return mat


SOURCE = "autostroke_source"
"""On a per-object copy: the name of the artist's material it was copied from."""
OWNER = "autostroke_object"
"""On a per-object copy: the object it belongs to (its maps are that object's)."""
ORIGINALS = "autostroke_slot_materials"
"""On the object: the material each slot held before AutoStroke touched it ('' = empty),
so Remove Strokes can put every slot back."""


def _node(nt, tag, bl_idname):
    """The node this addon owns under `tag` (custom property), or None."""
    for n in nt.nodes:
        if n.bl_idname == bl_idname and n.get(TAG) == tag:
            return n
    return None


def _new(nt, tag, bl_idname, location):
    n = nt.nodes.new(bl_idname)
    n[TAG] = tag
    n.location = location
    return n


def _principled(nt):
    """The Principled BSDF feeding the material output, else the first one, else None."""
    for n in nt.nodes:
        if n.bl_idname == "ShaderNodeOutputMaterial" and n.inputs["Surface"].links:
            src = n.inputs["Surface"].links[0].from_node
            if src.bl_idname == "ShaderNodeBsdfPrincipled":
                return src
    return next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)


def wire_strokes(mat, nrm_img, ind_img):
    """Add the stroke maps to an artist's material, keeping everything else it does.

    Normal: the stroke normal map (object space) into the Principled BSDF's Normal input.
    That replaces whatever fed Normal before -- one input, one normal. Base Color: the
    material's own base colour (a linked node or its plain value) times the per-stroke
    tone, the same 0.85..1.0 remap the grey material uses. Every node added is tagged, so
    running this again re-points images instead of stacking a second copy of the wiring.

    Returns False, touching nothing, when the material has no Principled BSDF."""
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(nt)
    if bsdf is None:
        return False
    x, y = bsdf.location

    tex = _tagged(nt, "stroke_normal") or nt.nodes.new("ShaderNodeTexImage")
    tex.location = (x - 700, y - 450)
    _configure(tex, nrm_img, "stroke_normal")
    nmap = _node(nt, "stroke_normal_map", "ShaderNodeNormalMap") or \
        _new(nt, "stroke_normal_map", "ShaderNodeNormalMap", (x - 350, y - 450))
    nmap.space = 'OBJECT'             # the baked normals ARE object-space
    nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    if ind_img is not None:
        ind = _tagged(nt, "stroke_indirection") or nt.nodes.new("ShaderNodeTexImage")
        ind.location = (x - 1050, y + 350)
        _configure(ind, ind_img, "stroke_indirection")
        sep = _node(nt, "stroke_tone_split", "ShaderNodeSeparateXYZ") or \
            _new(nt, "stroke_tone_split", "ShaderNodeSeparateXYZ", (x - 750, y + 350))
        tone = _node(nt, "stroke_tone", "ShaderNodeMix")
        if tone is None:
            tone = _new(nt, "stroke_tone", "ShaderNodeMix", (x - 500, y + 350))
            tone.data_type = 'RGBA'
            tone.inputs[6].default_value = (0.85, 0.85, 0.85, 1.0)
            tone.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)
        nt.links.new(ind.outputs["Color"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["Z"], tone.inputs["Factor"])
        mul = _node(nt, "stroke_tone_multiply", "ShaderNodeMix")
        if mul is None:
            # splice in between the material's own base colour and the BSDF, once
            mul = _new(nt, "stroke_tone_multiply", "ShaderNodeMix", (x - 250, y + 150))
            mul.data_type, mul.blend_type = 'RGBA', 'MULTIPLY'
            mul.inputs["Factor"].default_value = 1.0
            base = bsdf.inputs["Base Color"]
            if base.links:
                nt.links.new(base.links[0].from_socket, mul.inputs[6])
            else:
                mul.inputs[6].default_value = base.default_value
            nt.links.new(mul.outputs[2], base)
        nt.links.new(tone.outputs[2], mul.inputs[7])
    return True


def _copy_for(obj, src):
    """This object's AutoStroke copy of `src`, made once and then reused."""
    for m in bpy.data.materials:
        if m.get(SOURCE) == src.name and m.get(OWNER) == obj.name:
            return m
    copy = src.copy()
    copy.name = "%s_%s%s" % (src.name, obj.name, MAT_SUFFIX)
    copy[SOURCE] = src.name
    copy[OWNER] = obj.name
    return copy


def build(obj, force_rebuild=False):
    """Apply the baked maps to every material on `obj`. Returns a one-line summary.

    Each slot's material gets a per-object COPY with the strokes wired in (wire_strokes),
    so the artist's colour, roughness and the rest survive, the original stays untouched,
    and other objects sharing it are unaffected. Slots that are empty, or hold a material
    with no Principled BSDF, get the grey <object>_AutoStroke material instead -- which is
    always built, since it is also the fallback for an object with no materials at all.
    What each slot held before is recorded on the object, for Remove Strokes.

    Re-running (every re-bake does) re-points the tagged image nodes and never discards
    anything the artist wired up around them."""
    stem = safe_name(obj.name)
    nrm_img = _img(stem, "stroke_normal")
    ind_img = _img(stem, "stroke_indirection")
    if nrm_img is None:
        raise RuntimeError("No baked maps for %s yet -- press Bake first." % obj.name)

    grey = _grey(obj, nrm_img, ind_img, force_rebuild)
    if not obj.material_slots:
        obj.data.materials.append(grey)
        return "%s: 1 material (%s)" % (obj.name, grey.name)

    originals = list(obj.get(ORIGINALS) or [])
    originals += [""] * (len(obj.material_slots) - len(originals))
    copied, fallback = 0, []
    for i, slot in enumerate(obj.material_slots):
        cur = slot.material
        ours = cur is not None and (cur == grey or cur.get(OWNER) == obj.name)
        if not ours:
            # another object's copy (this object was duplicated after a bake): trace it
            # back to the artist's material rather than copying a copy
            originals[i] = cur.get(SOURCE, cur.name) if cur is not None else ""
        src = bpy.data.materials.get(originals[i]) if originals[i] else None
        if src is None:
            slot.material = grey
            if originals[i]:
                fallback.append(originals[i])
            continue
        copy = _copy_for(obj, src)
        if not wire_strokes(copy, nrm_img, ind_img):
            bpy.data.materials.remove(copy)
            slot.material = grey
            fallback.append(src.name)
            continue
        slot.material = copy
        copied += 1
    obj[ORIGINALS] = originals
    msg = "%s: strokes added to %d material%s" % (obj.name, copied, "" if copied == 1 else "s")
    if fallback:
        msg += "; grey AutoStroke material on %s (no Principled BSDF)" % ", ".join(fallback)
    return msg


def restore(obj):
    """Put back the materials the slots held before AutoStroke. Returns how many slots."""
    originals = list(obj.get(ORIGINALS) or [])
    n = 0
    for slot, name in zip(obj.material_slots, originals):
        slot.material = bpy.data.materials.get(name) if name else None
        n += 1
    if ORIGINALS in obj:
        del obj[ORIGINALS]
    return n


def _grey(obj, nrm_img, ind_img, force_rebuild):
    """The grey <object>_AutoStroke material, created or refreshed WITHOUT discarding edits.

    An existing material keeps its node graph; only the image nodes this addon tagged
    are re-pointed at the current datablocks. That matters now that the maps are
    file-backed: a rebake rebinds the datablock, and the material must follow without
    the artist losing whatever they wired up around it.
    """
    name = obj.name + MAT_SUFFIX
    mat = bpy.data.materials.get(name)
    created = mat is None
    if created:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    if created or force_rebuild or _tagged(mat.node_tree, "stroke_normal") is None:
        # nothing of ours to re-point (or explicitly asked for a clean slate)
        _build_fresh(mat, nrm_img, ind_img)
    else:
        for key, img in (("stroke_normal", nrm_img), ("stroke_indirection", ind_img)):
            if img is None:
                continue
            node = _tagged(mat.node_tree, key)
            if node is not None:
                _configure(node, img, key)
    return mat


class AUTOSTROKE_OT_build_material(bpy.types.Operator):
    bl_idname = "autostroke.build_material"
    bl_label = "Build Material"
    bl_description = "Create/refresh the painterly material from the baked maps"
    bl_options = {'REGISTER', 'UNDO'}

    force_rebuild: bpy.props.BoolProperty(
        name="Rebuild from scratch", default=False,
        description="Discard the existing node graph instead of re-pointing its images")

    def execute(self, context):
        try:
            msg = build(context.active_object, force_rebuild=self.force_rebuild)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class AUTOSTROKE_OT_remove_material(bpy.types.Operator):
    bl_idname = "autostroke.remove_material"
    bl_label = "Remove Strokes"
    bl_description = ("Put back the materials this object had before AutoStroke. The "
                      "AutoStroke copies stay in the file until unused data is purged")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.get(ORIGINALS) is not None

    def execute(self, context):
        n = restore(context.active_object)
        self.report({'INFO'}, "Restored %d material slot%s" % (n, "" if n == 1 else "s"))
        return {'FINISHED'}


class AUTOSTROKE_OT_open_folder(bpy.types.Operator):
    bl_idname = "autostroke.open_folder"
    bl_label = "Open Folder"
    bl_description = "Reveal the working directory"

    def execute(self, context):
        from .bake import resolve_working_dir
        try:
            path = resolve_working_dir(context.scene.autostroke)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        bpy.ops.wm.path_open(filepath=path)
        return {'FINISHED'}


class AUTOSTROKE_OT_refresh_brushes(bpy.types.Operator):
    bl_idname = "autostroke.refresh_brushes"
    bl_label = "Refresh"
    bl_description = ("Re-scan the brush folder. The set list is cached because Blender "
                     "asks for it on every redraw; press this after adding files")

    def execute(self, context):
        from ..bridge import brushes as brush_bridge
        brush_bridge.refresh()
        found = brush_bridge.sets(brush_bridge.root_dir(context.scene.autostroke))
        self.report({'INFO'}, "AutoStroke: %d brush set%s"
                    % (len(found), "" if len(found) == 1 else "s"))
        return {'FINISHED'}


classes = (AUTOSTROKE_OT_build_material, AUTOSTROKE_OT_remove_material,
           AUTOSTROKE_OT_open_folder, AUTOSTROKE_OT_refresh_brushes)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
