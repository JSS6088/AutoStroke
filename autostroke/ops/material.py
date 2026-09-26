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


def build(obj, force_rebuild=False):
    """Create the material, or refresh an existing one WITHOUT discarding edits.

    An existing material keeps its node graph; only the image nodes this addon tagged
    are re-pointed at the current datablocks. That matters now that the maps are
    file-backed: a rebake rebinds the datablock, and the material must follow without
    the artist losing whatever they wired up around it.
    """
    stem = safe_name(obj.name)
    nrm_img = _img(stem, "stroke_normal")
    ind_img = _img(stem, "stroke_indirection")
    if nrm_img is None:
        raise RuntimeError("No baked maps for %s yet -- press Bake first." % obj.name)

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

    if obj.data.materials:
        if mat.name not in [m.name for m in obj.data.materials if m]:
            obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
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
            mat = build(context.active_object, force_rebuild=self.force_rebuild)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Built %s" % mat.name)
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


classes = (AUTOSTROKE_OT_build_material, AUTOSTROKE_OT_open_folder,
           AUTOSTROKE_OT_refresh_brushes)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
