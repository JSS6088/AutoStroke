"""bpy.types.Image <-> numpy, and the ONE place that owns the row flip.

Blender stores image pixels BOTTOM-UP; the EXR files this pipeline historically read
via OpenEXR are TOP-DOWN (verified: allclose(exr_array, bpy_array[::-1])).

Getting that wrong does not produce an upside-down picture. The indirection map stores
UV *pointers*, so a flipped map silently makes every cell point somewhere wrong. Hence:
exactly two functions convert, both flip, and both are unit-tested.
"""

import numpy as np
import bpy

# Row 0 is the TOP row throughout this module, matching the OpenEXR reader and uv_self.


def image_to_numpy(img, channels=3):
    """(H, W, channels) float32, TOP-DOWN."""
    w, h = img.size
    buf = np.empty(w * h * img.channels, np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(h, w, img.channels)[..., :channels]
    return np.ascontiguousarray(arr[::-1])          # bottom-up -> top-down


def numpy_to_image(arr, name, float_buffer=True, is_data=True):
    """TOP-DOWN (H, W, 3|4) float array -> a float image datablock, reusing by name.

    Reuses the datablock whenever one exists, resizing IN PLACE rather than recreating.
    bpy.data.images.remove() sets EVERY reference to the datablock to None -- including
    material nodes -- so removing it to change resolution silently unlinks the texture
    from the shader. img.scale() keeps the datablock, and therefore every reference.
    """
    h, w = arr.shape[:2]
    img = bpy.data.images.get(name)
    if img is not None and img.is_float != float_buffer:
        # float-ness cannot be changed in place; this is the one case that must recreate
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, w, h, float_buffer=float_buffer, is_data=is_data)
    else:
        # A previous bake left this datablock FILE-backed, so its buffer is whatever the
        # file holds -- not necessarily the size we are about to write. Return it to
        # GENERATED at the right dimensions first. save_image() rebinds it to FILE after.
        # Note we never call bpy.data.images.remove(): that nulls EVERY reference to the
        # datablock, silently unlinking it from any material using it.
        img.source = 'GENERATED'
        if (img.generated_width, img.generated_height) != (w, h):
            img.generated_width = w
            img.generated_height = h
    if is_data:
        img.colorspace_settings.name = 'Non-Color'

    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :arr.shape[2]] = arr
    if arr.shape[2] < 4:
        rgba[..., 3] = 1.0
    img.pixels.foreach_set(rgba[::-1].ravel())      # top-down -> bottom-up
    img.update()
    return img


def save_image(img, filepath):
    """Write a float image as 32-bit EXR and REBIND the datablock to that file.

    The rebind is the point. bpy.data.images.new() produces a GENERATED image, and
    save_render() only writes pixels OUT -- `filepath_raw` is a write target, not a
    source. A GENERATED image is not stored in the .blend, so its pixels vanish on
    reload while the material link survives: the texture reads black and looks unlinked.

    Order is forced and cannot be rearranged: setting source='FILE' before the file
    exists collapses the pixel buffer to size 0, so the data would have nowhere to go.
    Write first, then rebind.
    """
    img.file_format = 'OPEN_EXR'
    img.filepath_raw = filepath
    scn = bpy.context.scene
    prev_fmt = scn.render.image_settings.file_format
    prev_depth = scn.render.image_settings.color_depth
    try:
        scn.render.image_settings.file_format = 'OPEN_EXR'
        scn.render.image_settings.color_depth = '32'
        img.save_render(filepath)
    finally:
        scn.render.image_settings.file_format = prev_fmt
        scn.render.image_settings.color_depth = prev_depth

    img.source = 'FILE'
    img.filepath = filepath
    img.reload()                       # also invalidates the stale GPU texture
    # re-assert AFTER reload: an EXR otherwise loads as Linear Rec.709. That happens to
    # be identical to Non-Color for float data, but sRGB is not -- and on an indirection
    # map a colour transform would corrupt the R,G UV POINTERS, not merely dim it.
    img.colorspace_settings.name = 'Non-Color'
    _tag_redraw()
    return filepath


def _tag_redraw():
    """Nudge the UI. No-op in background mode, where there is no screen."""
    screen = getattr(bpy.context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type in {'VIEW_3D', 'IMAGE_EDITOR', 'NODE_EDITOR'}:
            area.tag_redraw()


def write_map(arr, name, filepath):
    """numpy array -> image datablock -> EXR on disk -> datablock rebound to that file.

    The single entry point for every baked map, so the GENERATED-image trap cannot be
    reintroduced one map at a time.
    """
    img = numpy_to_image(arr, name)
    save_image(img, filepath)
    return img
