"""How long a bake will take, for the panel readout.

Fitted to 34 measured bakes across resolution 1024-2048, 2,000-20,000 strokes and Stroke
Size 1-10 (the range the UI allows). Max error 24% on the resolve term, 16% on the
direction field.

Two things this cannot know, which is why `scale` exists:

- **the machine.** The constants come from one developer machine; a real bake measured
  2.93s of resolve where this fixture gives ~1.5s, so a fixed constant is wrong by ~2x on
  other hardware.
- **the asset.** Cost depends on how tightly the brushes crop, how the UV islands pack,
  and how much the stamps overlap -- none of which is a panel setting.

So `ops/bake.py` divides the real bake time by this prediction and feeds the ratio back as
`scale`, smoothed. After one bake the readout is calibrated to the artist's own machine and
model; before that it uses a deliberately high default rather than promising too little.
"""

RESOLVE_K = 7.9616e-07
RESOLVE_RES = 1.352        # texels: sub-quadratic, the tile rejection sees to that
RESOLVE_N = 0.538          # strokes: sub-linear, since more strokes are smaller strokes
RESOLVE_SIZE = 0.271
"""Stroke Size matters and used to be missing entirely. Bigger stamps overlap more, so more
(stroke, tile) pairs survive rejection. Beyond the UI's cap of 10 the curve actually turns
DOWN -- once coverage saturates, the loop's early-out fires -- but that is out of range and
fitting it would make the in-range answer worse."""

DIR_K, DIR_N = 1.4766e-07, 1.702
"""The direction field is kNN plus smoothing over seeds only; it does not see the texture
at all. Near-quadratic, so it overtakes the resolve above roughly 20,000 strokes."""

OVERHEAD_AT_1K = 0.25
"""Seeds, brush decode, dilation and the EXR writes, scaled by area. Small and very
asset-dependent -- the dilation in particular does nothing when UV islands fill the map."""

DEFAULT_SCALE = 1.25
"""Until a real bake calibrates it. High on purpose: an estimate that reads long is
harmless, one that reads short erodes trust in the readout."""


def bake_seconds(res, strokes, stroke_size, scale=DEFAULT_SCALE):
    """Predicted wall clock for one bake, in seconds."""
    n = max(float(strokes), 1.0)
    s = max(float(stroke_size), 0.1)
    r = max(float(res), 1.0)
    return (RESOLVE_K * r ** RESOLVE_RES * n ** RESOLVE_N * s ** RESOLVE_SIZE
            + DIR_K * n ** DIR_N
            + OVERHEAD_AT_1K * (r / 1024.0) ** 2) * float(scale)


def calibration(actual, predicted_raw, previous=DEFAULT_SCALE, blend=0.5):
    """Fold a finished bake into the machine/asset factor.

    Blended rather than replaced so one unusual bake -- a cold cache, another app hogging
    the CPU -- cannot swing the readout, and clamped so a pathological ratio cannot make
    the estimate meaningless in either direction.
    """
    if predicted_raw <= 0.0 or actual <= 0.0:
        return previous
    ratio = min(max(actual / predicted_raw, 0.2), 5.0)
    return (1.0 - blend) * previous + blend * ratio
