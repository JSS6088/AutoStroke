"""Brush sets: a folder of stroke images the baker draws from at random.

A set is a subdirectory holding one or more images. The shipped root is assets/brushes/
(`standard`, `rough`, `thin`); pointing Brush Folder at another directory makes ITS
subdirectories the sets instead, which is how an artist uses their own pack.

Nothing here normalises the brushes against each other, and that is a decision rather than
an omission: brush packs arrive normalised, and an artist who draws one mark longer than
another meant to. Scaling every brush to a common painted extent would silently undo it.
"""

import os

import numpy as np
import bpy

from . import images as bi

EXTS = (".png", ".tga", ".tif", ".tiff", ".exr", ".jpg", ".jpeg")

PREFERRED_FIRST = ("standard",)
"""Sets listed before the rest, so `standard` is what a fresh scene gets. A dynamic enum
cannot take a `default=` -- Blender only accepts one when `items` is a fixed list -- so
being first in the list IS the default. Everything else stays alphabetical."""

BRUSH_MAX_PX = 1024
"""Masks are decoded to float32, so a 2048px brush would be 16.8 MB -- 67 MB for a set of
four. Halving costs at most 0.10 points of painted fraction and 0.2 deg of long axis
(measured on the standard and rough sets at 2048; they now ship at 1024) and brings a set
to 17 MB. A stamp covers ~40 texels on a 2K map, so 1024 is still ~25x more brush detail
than any stroke can show."""


def default_root():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "brushes")


def root_dir(st):
    """Where the sets live: the artist's Brush Folder if set, else the shipped one."""
    custom = (getattr(st, "brush_dir", "") or "").strip()
    if custom:
        return bpy.path.abspath(custom)
    return default_root()


def is_brush_file(name):
    """An image by extension -- and not a hidden file. A pack zipped on a Mac carries
    '._name.png' AppleDouble files beside every image: they end in .png but hold no
    pixels, and load as 0x0 images."""
    return not name.startswith(".") and name.lower().endswith(EXTS)


def _listdir(d):
    try:
        return os.listdir(d)
    except OSError:
        return []


def _scan(root):
    """Subdirectory names holding at least one image, sorted."""
    out = []
    for name in sorted(_listdir(root)):
        d = os.path.join(root, name)
        if (not name.startswith(".") and os.path.isdir(d)
                and any(is_brush_file(f) for f in _listdir(d))):
            out.append(name)
    rank = {n: i for i, n in enumerate(PREFERRED_FIRST)}
    return sorted(out, key=lambda n: (rank.get(n, len(rank)), n))


# ---- the enum, and why it is cached ---------------------------------------
# Blender calls an EnumProperty items callback on every UI redraw -- mouse movement over
# the panel, viewport orbit -- not just when the dropdown opens. Listing a directory in
# there is disk I/O tens of times a second, which on a network drive stutters the UI while
# looking like the addon being slow. So the scan is cached per root; `refresh()` is what
# picks up a file added while Blender is open.
_scan_cache = {}
_files_cache = {}

_SET_ITEMS = []
"""Keeps the enum's strings alive. When `items` is a callback Blender stores raw pointers
to the returned strings and holds no Python reference of its own, so a list that only
exists inside the callback is freed the moment it returns and Blender is left reading
freed memory -- garbled labels, or a crash on opening the dropdown. Documented Blender
behaviour, and the reason this is a module global rather than a local."""


def sets(root):
    if root not in _scan_cache:
        _scan_cache[root] = _scan(root)
    return _scan_cache[root]


def refresh():
    _scan_cache.clear()
    _files_cache.clear()


_ITEMS_BY_SETS = {}
"""Every item list ever returned, by the sets it lists, and never emptied. Blender keeps
raw pointers to an items callback's strings and may call the callback again while still
holding the previous result; rebuilding the list on every call freed those strings under
it -- garbled labels, or a crash opening the dropdown. Returning the SAME list for the
same folder contents keeps every string alive. Tiny: a few entries per brush folder."""


def set_items(self, context):
    """EnumProperty items callback. `self` is the settings group, so it carries brush_dir."""
    global _SET_ITEMS
    found = tuple(sets(root_dir(self)))
    items = _ITEMS_BY_SETS.get(found)
    if items is None:
        items = [(n, n.replace("_", " ").title(), "%s brush set" % n) for n in found]
        if not items:
            items = [('NONE', "(no brush sets found)", "No subfolder here holds an image")]
        _ITEMS_BY_SETS[found] = items
    _SET_ITEMS = items
    return items


def files(st):
    """Absolute paths of the chosen set's images, sorted so the draw is reproducible."""
    root = root_dir(st)
    name = st.brush_set
    if not name or name == 'NONE':
        avail = sets(root)
        if not avail:
            return root, None, []
        name = avail[0]
    d = os.path.join(root, name)
    # Cached for the same reason the set list is: the panel asks for this on every redraw
    # to show the brush count, and that would be a directory listing per redraw.
    if d not in _files_cache:
        found = sorted(f for f in _listdir(d) if is_brush_file(f))
        _files_cache[d] = [os.path.join(d, f) for f in found]
    return root, name, _files_cache[d]


def load_masks(st, cfg):
    """The chosen set as a list of (H,W) float32 masks. Raises with an artist-facing message.

    The image is only READ, never changed: brushes used to be shrunk with img.scale(),
    which marks the datablock modified -- and Blender's "Save Modified Images" (offered on
    every save and quit) then wrote the shrunken brush back over the artist's file. Large
    brushes are downsampled here in numpy instead. A datablock this call loaded just to
    read is removed again; one that already existed (the artist's own, or left by an
    older version) is kept, and reloaded from disk first if a past scale() left it dirty.
    """
    from ..core import baker

    root, name, paths = files(st)
    if not paths:
        raise RuntimeError(
            "No brush images in %s. A brush set is a subfolder of that directory holding "
            "one or more images (png, tga, tif, exr, jpg)."
            % (os.path.join(root, name) if name else root))
    masks, unreadable = [], []
    for path in paths:
        n_before = len(bpy.data.images)
        try:
            img = bpy.data.images.load(path, check_existing=True)
        except RuntimeError:
            unreadable.append(os.path.basename(path))
            continue
        ours = len(bpy.data.images) > n_before
        try:
            if not ours and img.is_dirty:
                img.reload()            # undo an older version's in-memory scale()
            # Non-Color: the mask is a shape, not a picture. Leaving it sRGB applies a
            # transfer curve to the brush and silently changes coverage.
            if img.colorspace_settings.name != 'Non-Color':
                img.colorspace_settings.name = 'Non-Color'
            if min(img.size) < 1:       # corrupt or not really an image: 0 x 0
                unreadable.append(os.path.basename(path))
                continue
            arr = downsample(bi.image_to_numpy(img, img.channels), BRUSH_MAX_PX)
            masks.append(baker.mask_from_rgba(arr, cfg))
        finally:
            if ours:
                bpy.data.images.remove(img)
    if not masks:
        raise RuntimeError("None of the images in %s could be read (%s)"
                           % (os.path.join(root, name), ", ".join(unreadable[:5])))
    return masks, name


def downsample(arr, max_px):
    """(H, W, C) -> at most max_px on its long side, aspect kept. Exact block means for
    whole-number factors (2048 -> 1024, the common case), nearest sampling otherwise."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_px:
        return arr
    k = max(h, w) / float(max_px)
    if k == int(k) and h % int(k) == 0 and w % int(k) == 0:
        f = int(k)
        return arr.reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3)).astype(arr.dtype)
    nh, nw = max(1, int(round(h / k))), max(1, int(round(w / k)))
    yi = np.minimum(((np.arange(nh) + 0.5) * (h / float(nh))).astype(np.int64), h - 1)
    xi = np.minimum(((np.arange(nw) + 0.5) * (w / float(nw))).astype(np.int64), w - 1)
    return arr[yi][:, xi]
