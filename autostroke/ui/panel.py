"""The AutoStroke N-panel."""

import numpy as np
import bpy

from .. import livepreview
from ..bridge import brushes as brush_bridge
from ..ops import bake as bake_ops
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


def _estimate_batch(context, objs):
    """_estimate(), summed across every object Bake will actually process.

    Stays exactly as cheap as _estimate() itself: predicted_total() and the density
    solve below both read obj.data.polygons directly (no to_mesh(), no depsgraph), so
    looping this over a selection is still plain numpy work per object, not N depsgraph
    evaluations -- safe to call on every panel redraw the same way the single-object
    version always has been.

    at_min/at_max are OR'd across the selection rather than kept per-object: "is ANY
    selected object capped" is the one thing worth a warning at a glance; a per-object
    breakdown would just be noise until an artist actually suspects a specific object.
    """
    total_strokes, total_secs = 0, 0.0
    any_at_min = any_at_max = False
    for obj in objs:
        s, t, amin, amax = _estimate(context, obj)
        total_strokes += s
        total_secs += t          # objects bake sequentially, never in parallel -- sum is right
        any_at_min = any_at_min or amin > 0.5
        any_at_max = any_at_max or amax > 0.5
    return total_strokes, total_secs, any_at_min, any_at_max


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
        # The single source of truth for "which objects will Bake actually process" --
        # imported rather than reimplemented, so the readiness line below and what
        # pressing Bake does can never disagree about the answer.
        objs = bake_ops.targets(context)

        if not objs:
            problems = setup_ops.validate(context.active_object)
            box = layout.box()
            for p in problems:
                box.label(text=p, icon='ERROR')
            return

        if len(objs) == 1:
            layout.label(text="%s  ready" % objs[0].name, icon='CHECKMARK')
        else:
            layout.label(text="%d objects ready" % len(objs), icon='CHECKMARK')
            names = layout.box()
            for o in objs:
                names.label(text=o.name)

        legacy_names = []
        for o in objs:
            mod = setup_ops.legacy_modifier(o)
            if mod is not None and mod.show_viewport:
                legacy_names.append(o.name)
        if legacy_names:
            warn = layout.box()
            warn.label(text="Old seeder modifier is still active", icon='ERROR')
            if len(objs) > 1:
                warn.label(text="on: %s" % ", ".join(legacy_names))
                warn.label(text="The button below only disables it on the active object.")
            else:
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
        if len(objs) == 1:
            strokes, secs, at_min, at_max = _estimate(context, objs[0])
        else:
            strokes, secs, at_min, at_max = _estimate_batch(context, objs)
        # The time model is fitted to the CPU resolver; a GPU bake is dominated by the
        # (cached) Cycles position bake instead, so quoting a CPU time there would be wrong.
        on_gpu = st.bake_device == 'GPU'
        when = "bake on GPU" if on_gpu else "bake ~ %s" % _fmt_time(secs)
        label = ("~ %s strokes  ·  %s" % ("{:,}".format(strokes), when)
                 if len(objs) == 1 else
                 "~ %s strokes total  ·  %s across %d objects"
                 % ("{:,}".format(strokes), when, len(objs)))
        box.label(text=label, icon='INFO' if on_gpu or secs < 300 else 'ERROR')
        if strokes:
            capped = at_max > 0.5
            if len(objs) == 1:
                box.label(text="faces at min %.0f%%   at max %.0f%%" % (100 * at_min, 100 * at_max),
                          icon='ERROR' if capped else 'NONE')
            elif capped:
                box.label(text="At least one selected object is capped by Max Strokes per Face.",
                          icon='ERROR')
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
        row = box.row(align=True)
        row.prop(st, "bake_device", expand=True)
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
