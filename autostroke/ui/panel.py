"""The AutoStroke N-panel."""

import numpy as np
import bpy

from .. import livepreview
from ..bridge import brushes as brush_bridge
from ..ops import setup as setup_ops


def _estimate(context, obj):
    """(stroke count, seconds) for the current settings, shown BEFORE baking.

    Counted exactly the way the sampler will count, so the number on screen is the number
    the artist gets -- including the min/max clamps, which quietly take over as density
    rises. The time constant is calibrated from real bakes.
    """
    from ..core import cost, sampling as S
    from ..bridge import seeds as seed_bridge
    st = context.scene.autostroke
    try:
        # The count itself comes from bridge/seeds, which is also what the Min/Max
        # clamp asks -- one implementation of "how many strokes will this produce", so
        # the readout and the guard can never disagree about it.
        strokes, _n_faces = seed_bridge.predicted_total(
            obj, st.target_strokes, st.min_strokes, st.max_strokes)
        me = obj.data
        areas = np.empty(len(me.polygons), np.float32)
        me.polygons.foreach_get("area", areas)
        scale = np.array(obj.matrix_world.to_3x3()).__abs__().sum(0).prod() ** (2.0 / 3.0)
        areas = areas.astype(np.float64) * scale
        density, _reachable = S.solve_density(areas, st.target_strokes,
                                              st.min_strokes, st.max_strokes)
        counts = S.stroke_counts(areas, density, st.min_strokes, st.max_strokes)
        at_min = float((counts == st.min_strokes).mean())
        at_max = float((counts == st.max_strokes).mean())
    except Exception:
        return 0, 0.0, 0.0, 0.0
    # See core/cost.py for the model and what it can and cannot know. Stroke Size is in
    # there now: bigger stamps overlap more, so more strokes survive each tile's rejection,
    # and leaving it out was most of why this read low.
    n = max(strokes, 1)
    secs = cost.bake_seconds(int(st.resolution), n, st.stroke_size, st.est_scale)
    return strokes, secs, at_min, at_max


def _fmt_time(s):
    if s < 90:
        return "%.0fs" % s
    if s < 3600:
        return "%.0f min" % (s / 60.0)
    return "%.1f hr" % (s / 3600.0)


class AUTOSTROKE_PT_main(bpy.types.Panel):
    bl_label = "AutoStroke"
    bl_idname = "AUTOSTROKE_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AutoStroke"

    def draw(self, context):
        layout = self.layout
        st = context.scene.autostroke
        obj = context.active_object

        problems = setup_ops.validate(obj)
        if problems:
            box = layout.box()
            for p in problems:
                box.label(text=p, icon='ERROR')
            return

        layout.label(text="%s  ready" % obj.name, icon='CHECKMARK')

        legacy = setup_ops.legacy_modifier(obj)
        if legacy is not None and legacy.show_viewport:
            warn = layout.box()
            warn.label(text="Old seeder modifier is still active", icon='ERROR')
            warn.label(text="Strokes are placed by this panel now.")
            warn.operator("autostroke.disable_legacy", icon='CANCEL')

        box = layout.box()
        box.label(text="Placement")
        box.prop(st, "target_strokes")
        # Locked while the preview is on. Both are per-FACE, so they multiply by the face
        # count and ignore Stroke Count: Min 64 on a 20,000-face mesh is 1,280,000
        # strokes. Rebuilding that means re-running the direction field, whose kNN is
        # O(N^2) and runs on Blender's main thread with no progress and no Esc -- the one
        # combination that takes Blender down rather than merely being slow.
        #
        # Locking rather than clamping: these two are the least likely dials to want
        # mid-preview, and a lock is a thing the artist can see, where a slider that
        # silently stops moving is not.
        live = livepreview.enabled()
        row = box.row(align=True)
        row.enabled = not live
        row.prop(st, "min_strokes")
        row.prop(st, "max_strokes")
        if live:
            box.label(text="Min/Max per Face locked while previewing", icon='LOCKED')
        box.prop(st, "crease_angle")
        strokes, secs, at_min, at_max = _estimate(context, obj)
        box.label(text="~ %s strokes  ·  bake ~ %s" % ("{:,}".format(strokes), _fmt_time(secs)),
                  icon='INFO' if secs < 300 else 'ERROR')
        if strokes:
            capped = at_max > 0.5
            box.label(text="faces at min %.0f%%   at max %.0f%%" % (100 * at_min, 100 * at_max),
                      icon='ERROR' if capped else 'NONE')
            if capped:
                box.label(text="Most faces are capped -- density has little effect.")
                box.label(text="Raise Max Strokes per Face.")
        row = box.row(align=True)
        row.operator("autostroke.estimate", icon='VIEWZOOM')
        row.prop(st, "estimate_rate", text="")
        if st.last_estimate:
            box.label(text=st.last_estimate, icon='INFO')

        box = layout.box()
        box.label(text="Brush")
        row = box.row(align=True)
        row.prop(st, "brush_set", text="")
        row.operator("autostroke.refresh_brushes", text="", icon='FILE_REFRESH')
        box.prop(st, "brush_dir")
        n_brushes = len(brush_bridge.files(st)[2])
        box.label(text="%d brush%s in this set" % (n_brushes, "" if n_brushes == 1 else "es"),
                  icon='INFO' if n_brushes else 'ERROR')
        box.prop(st, "stroke_size")
        box.prop(st, "size_random")
        box.prop(st, "flow_angle")
        box.prop(st, "rot_jitter")

        box = layout.box()
        box.label(text="Output")
        box.prop(st, "working_dir")
        box.prop(st, "resolution")
        box.prop(st, "force_position_bake")

        box = layout.box()
        on = livepreview.enabled()
        row = box.row(align=True)
        row.operator("autostroke.live", text="Live Preview: %s" % ("ON" if on else "off"),
                     icon='HIDE_OFF' if on else 'HIDE_ON', depress=on)
        if on:
            row.operator("autostroke.live_refresh", text="", icon='FILE_REFRESH')
            box.prop(st, "live_mode", text="")
            box.label(text="Updates as you tweak. Refresh after editing the mesh.",
                      icon='INFO')
            box.label(text="Shades the object directly -- not your material.", icon='INFO')
        if livepreview.last_error():
            box.label(text=livepreview.last_error()[:70], icon='ERROR')

        layout.separator()
        layout.operator("autostroke.bake", text="B A K E", icon='RENDER_STILL')
        if st.last_report:
            for line in st.last_report.split("|"):
                layout.label(text=line)
        row = layout.row(align=True)
        row.operator("autostroke.build_material", icon='MATERIAL')
        row.operator("autostroke.open_folder", icon='FILE_FOLDER')


classes = (AUTOSTROKE_PT_main,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
