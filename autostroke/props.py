"""Scene- and object-level settings backing the panel."""

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)

from . import livepreview
from .bridge import brushes as brush_bridge


def _relive(self, context):
    """Anything that changes WHAT the preview draws: coalesced into one rebuild."""
    livepreview.request_rebuild()


def _redraw(self, context):
    """Anything the shader reads live, so the viewport only needs to redraw."""
    livepreview.redraw_only()


class AutoStrokeSettings(bpy.types.PropertyGroup):
    # ---- placement --------------------------------------------------------
    target_strokes: IntProperty(
        name="Stroke Count", default=5000, min=1, soft_max=50000, update=_relive,
        description="How many strokes to place on this object. The per-square-metre "
                    "density is worked out from this, so it stays right when the model "
                    "is scaled or re-topologised")
    min_strokes: IntProperty(
        name="Min Strokes per Face", default=1, min=1, max=64, update=_relive,
        description="Every face gets at least this many strokes, so none is left empty. "
                    "Applies to EVERY face, so it multiplies by the face count and can "
                    "overrule Stroke Count entirely. Locked while the live preview is "
                    "on -- see the Placement panel")
    max_strokes: IntProperty(
        name="Max Strokes per Face", default=128, min=1, max=4096, soft_max=512,
        update=_relive,
        description="Safety cap on one face, not the main control. Low-poly meshes have "
                    "few, large faces, so a low cap swallows the density dial entirely -- "
                    "watch the 'at max' readout: if it says 100%, density is doing nothing")

    # ---- brush ------------------------------------------------------------
    brush_set: EnumProperty(
        name="Brush Set", items=brush_bridge.set_items, update=_relive,
        description="Which set of stroke shapes to paint with. Every stroke draws one at "
                    "random from the set, so no single silhouette repeats across the model")
    brush_dir: StringProperty(
        name="Brush Folder", subtype='DIR_PATH', default="", update=_relive,
        description="Use your own brushes: point this at a folder whose SUBFOLDERS are "
                    "the sets. Leave empty for the shipped sets. Press Refresh after "
                    "adding files while Blender is open")
    stroke_size: FloatProperty(
        name="Stroke Size", default=4.0, min=0.1, max=10.0, soft_max=10.0, update=_relive,
        description="How big each stamp is relative to the patch of surface it owns. "
                    "Around 4 is where marks start meeting across their WIDTH: a brush "
                    "fills most of its image lengthwise but only a fifth of it across, "
                    "so lower values leave gaps between strokes. Scales the whole brush "
                    "uniformly -- nothing stretches")
    size_random: FloatProperty(
        name="Size Variation", default=0.0, min=0.0, max=1.0, update=_relive,
        description="Random size per stroke. 0 = every stroke at Stroke Size; 1 = each "
                    "lands between half and double it, so Stroke Size 5 gives 2.5 to 10. "
                    "Halving is as likely as doubling")
    flow_angle: FloatProperty(
        # identifier stays `flow_angle`: renaming it would make Blender drop the value
        # every saved scene holds for it. Panel label only.
        name="Global Rotation Angle", default=0.0, min=-180.0, max=180.0,
        subtype='ANGLE', update=_relive,
        description="Rotate every stroke relative to the flow direction. 90 makes them "
                    "run ACROSS the form instead of along it")

    # ---- creases ----------------------------------------------------------
    rot_jitter: FloatProperty(
        name="Rotation Jitter", default=0.0, min=0.0, max=0.785398, update=_relive,
        subtype='ANGLE',
        description="Random rotation per stroke, plus or minus this much. Breaks up the "
                    "repetition of one brush shape. Capped at 45 degrees: beyond that "
                    "strokes stop following the flow they were aligned to")

    crease_angle: FloatProperty(
        # identifier stays `crease_angle` for the same reason as flow_angle above
        name="Stroke Cutoff", default=0.785398, min=0.174533, max=1.570796,
        update=_redraw,
        subtype='ANGLE',
        description="How far the surface may turn away from a stroke before that stroke "
                    "gives up. LOWER = strokes stop sooner, so more of the model keeps "
                    "its own smooth surface normal: accurate shading, but less visible "
                    "brushwork. HIGHER = strokes wrap further and the brush pattern reads "
                    "more strongly, at the cost of flat patches where one stamp spans a "
                    "curve")

    # ---- output -----------------------------------------------------------
    working_dir: StringProperty(
        name="Working Dir", subtype='DIR_PATH', default="//AutoStroke/",
        description="Where baked maps are written. Relative paths need a saved .blend")
    resolution: EnumProperty(
        name="Resolution", default='2048',
        items=[('1024', "1024", ""), ('2048', "2048", ""),
               ('4096', "4096", ""), ('8192', "8192", "")])
    bake_device: EnumProperty(
        name="Bake With", default='GPU',
        items=[('GPU', "GPU", "Resolve strokes on the GPU with the live preview's own "
                              "search, so the bake matches what the viewport shows. Falls "
                              "back to the CPU by itself if the GPU fails or Blender runs "
                              "headless"),
               ('CPU', "CPU", "Resolve strokes in numpy on the CPU. Slower; use it if the "
                              "GPU bake misbehaves on this machine")],
        description="Which processor resolves the strokes into the texture")
    force_position_bake: BoolProperty(
        name="Force Re-bake Position & Normal Map", default=False,
        description="Both maps are cached against the evaluated mesh: its vertex "
                    "positions, UVs and connectivity. Ordinary edits invalidate them on "
                    "their own, so this is only for forcing a re-bake anyway")

    estimate_rate: FloatProperty(
        name="Sample Rate", default=0.15, min=0.01, max=10.0, soft_max=1.0,
        subtype='PERCENTAGE', precision=2,
        description="How much of the texture Check Coverage samples, as a percentage. "
                    "0.15%% of a 2K map is ~6,000 points, which lands within a point of a "
                    "full bake. Higher is slower and only slightly more precise")
    last_estimate: StringProperty(name="", default="")

    live_mode: EnumProperty(
        name="Live View", default='0', update=_redraw,
        items=[('0', "Shaded", "Strokes, lit by a headlight"),
               ('1', "Stroke Normals", "The normal each stroke paints, as colour"),
               ('2', "Cell Occupancy", "How many strokes each grid cell holds -- green "
                                       "is cheap, red is a long shader loop")],
        description="What the live preview draws. The debug modes are for telling a "
                    "placement problem apart from a binning problem")

    # ---- last-bake readout ------------------------------------------------
    last_report: StringProperty(name="", default="")
    est_scale: FloatProperty(
        name="Estimate Calibration", default=1.25, min=0.05, max=20.0,
        description="How far the bake-time estimate was off last time, folded back in. "
                    "Updated after every bake, so the prediction learns this machine and "
                    "this model rather than trusting a constant measured elsewhere")


def register():
    bpy.utils.register_class(AutoStrokeSettings)
    bpy.types.Scene.autostroke = PointerProperty(type=AutoStrokeSettings)


def unregister():
    del bpy.types.Scene.autostroke
    bpy.utils.unregister_class(AutoStrokeSettings)
