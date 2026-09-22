"""Validation, and a one-click "ready" state.

Seeding moved into Python (core/sampling.py), so there is no node group to append, no
socket name->identifier mapping and no depsgraph point-cloud walk. What remains is
checking the object can actually be baked, and telling the artist plainly when it cannot.
"""

import bpy

LEGACY_NODE_GROUP = "ScatterSeeds_Face"


def legacy_modifier(obj):
    """The old Geometry Nodes seeder, if this object still carries one."""
    if obj is None:
        return None
    for m in obj.modifiers:
        if (m.type == 'NODES' and m.node_group
                and m.node_group.name.startswith(LEGACY_NODE_GROUP)):
            return m
    return None


def validate(obj):
    """Human-readable problems; empty means ready to bake."""
    if obj is None:
        return ["No active object"]
    if obj.type != 'MESH':
        return ["%s is not a mesh" % obj.name]
    problems = []
    if not obj.data.polygons:
        problems.append("%s has no faces" % obj.name)
    if not obj.data.uv_layers:
        problems.append("%s has no UV map -- unwrap it first (Smart UV Project is fine)"
                        % obj.name)
    elif obj.data.uv_layers.active is None:
        problems.append("%s has UV maps but none is active" % obj.name)
    return problems


class AUTOSTROKE_OT_estimate(bpy.types.Operator):
    bl_idname = "autostroke.estimate"
    bl_label = "Check Coverage"
    bl_description = ("Run the real stamp test against a few thousand points on the "
                      "surface. About a second, and lands within a point of a full bake")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return not validate(context.active_object)

    def execute(self, context):
        import time
        from ..core import coverage as CV, geometry
        from ..bridge import seeds as seed_bridge, mesh as mesh_bridge
        from .bake import config_from_settings, load_brush
        from ..core.config import Config

        obj = context.active_object
        st = context.scene.autostroke
        t0 = time.time()
        try:
            mask, _set = load_brush(st, Config())
            cfg = config_from_settings(st, mask)
            sd, ss = seed_bridge.build_seeds(
                obj, cfg.target_strokes, cfg.min_strokes, cfg.max_strokes,
                cfg.aspect_alpha)
            tan = None
            if cfg.direction_source == "curvature":
                tan, _ = geometry.compute_direction_field(sd["position"], sd["normal"], cfg)
            rot = cfg.stamp_rotate_deg      # per-brush alignment happens in resolve_uv
            m = mesh_bridge.read_triangles(obj)
            # the rate is a share of the texture the bake would write, so the check
            # scales with the resolution the artist actually picked
            n_pts = int(min(200000, max(500,
                        round(int(st.resolution) ** 2 * st.estimate_rate / 100.0))))
            pts, pnrm = CV.sample_surface(m["P"], m["Q"], m["R"], n_pts)
            cov, n = CV.estimate_coverage(pts, sd, mask, cfg, tan, rot, pt_nrm=pnrm)
        except (RuntimeError, mesh_bridge.MeshError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        short = ss["reachable"] < ss["target"]
        st.last_estimate = (
            "%.0f%% coverage · %s strokes%s · %.2f%% sampled in %.1fs"
            % (100 * cov, "{:,}".format(ss["strokes"]),
               " (capped: raise Max Strokes per Face)" if short else "",
               100.0 * n / max(int(st.resolution) ** 2, 1), time.time() - t0))
        self.report({'INFO'}, st.last_estimate)
        return {'FINISHED'}


class AUTOSTROKE_OT_disable_legacy(bpy.types.Operator):
    bl_idname = "autostroke.disable_legacy"
    bl_label = "Disable old seeder"
    bl_description = ("Strokes are placed by the add-on now. The old ScatterSeeds_Face "
                      "modifier only hides the mesh in the viewport")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        mod = legacy_modifier(context.active_object)
        if mod is None:
            self.report({'INFO'}, "No legacy modifier found")
            return {'CANCELLED'}
        mod.show_viewport = False
        self.report({'INFO'}, "Hid %s -- placement settings are in this panel now" % mod.name)
        return {'FINISHED'}


classes = (AUTOSTROKE_OT_estimate, AUTOSTROKE_OT_disable_legacy)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
