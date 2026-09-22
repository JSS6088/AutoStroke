"""Brush sets: a folder of stroke images the baker draws from at random.

A set is a subdirectory holding one or more images. The shipped root is assets/brushes/
(`standard`, `rough`); pointing Brush Folder at another directory makes ITS subdirectories
the sets instead, which is how an artist uses their own pack.

Nothing here normalises the brushes against each other, and that is a decision rather than
an omission: brush packs arrive normalised, and an artist who draws one mark longer than
another meant to. Scaling every brush to a common painted extent would silently undo it.
"""

import os

import bpy

from . import images as bi

EXTS = (".png", ".tga", ".tif", ".tiff", ".exr", ".jpg", ".jpeg")

PREFERRED_FIRST = ("standard",)
"""Sets listed before the rest, so `standard` is what a fresh scene gets. A dynamic enum
cannot take a `default=` -- Blender only accepts one when `items` is a fixed list -- so
being first in the list IS the default. Everything else stays alphabetical."""

BRUSH_MAX_PX = 1024
"""Masks are decoded to float32, so the shipped 2048px brushes would be 16.8 MB each --
134 MB for eight. Halving them costs at most 0.10 points of painted fraction and 0.2 deg
of long axis (measured across both sets) and brings that to 34 MB. A stamp covers ~40
texels on a 2K map, so 1024 is still ~25x more brush detail than any stroke can show."""


def default_root():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "brushes")


def root_dir(st):
    """Where the sets live: the artist's Brush Folder if set, else the shipped one."""
    custom = (getattr(st, "brush_dir", "") or "").strip()
    if custom:
        return bpy.path.abspath(custom)
    return default_root()


def _scan(root):
    """Subdirectory names holding at least one image, sorted."""
    try:
        names = os.listdir(root)
    except OSError:
        return []
    out = []
    for name in sorted(names):
        d = os.path.join(root, name)
        if os.path.isdir(d) and any(f.lower().endswith(EXTS) for f in os.listdir(d)):
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


def set_items(self, context):
    """EnumProperty items callback. `self` is the settings group, so it carries brush_dir."""
    global _SET_ITEMS
    found = sets(root_dir(self))
    _SET_ITEMS = [(n, n.replace("_", " ").title(), "%s brush set" % n) for n in found]
    if not _SET_ITEMS:
        _SET_ITEMS = [('NONE', "(no brush sets found)", "No subfolder here holds an image")]
    return _SET_ITEMS


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
        try:
            found = sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))
        except OSError:
            found = []
        _files_cache[d] = [os.path.join(d, f) for f in found]
    return root, name, _files_cache[d]


def load_masks(st, cfg):
    """The chosen set as a list of (H,W) float32 masks. Raises with an artist-facing message.

    One datablock per brush, loaded with check_existing so re-baking does not pile up
    copies, and scaled in place -- never images.remove(), which would null out every
    reference to the datablock including any material node pointing at it.
    """
    from ..core import baker

    root, name, paths = files(st)
    if not paths:
        raise RuntimeError(
            "No brush images in %s. A brush set is a subfolder of that directory holding "
            "one or more images (png, tga, tif, exr, jpg)."
            % (os.path.join(root, name) if name else root))
    masks = []
    for path in paths:
        img = bpy.data.images.load(path, check_existing=True)
        # Non-Color: the mask is a shape, not a picture. Leaving it sRGB applies a
        # transfer curve to the brush and silently changes coverage.
        if img.colorspace_settings.name != 'Non-Color':
            img.colorspace_settings.name = 'Non-Color'
        if max(img.size) > BRUSH_MAX_PX:
            w, h = img.size
            k = BRUSH_MAX_PX / float(max(w, h))
            img.scale(max(1, int(round(w * k))), max(1, int(round(h * k))))
        masks.append(baker.mask_from_rgba(bi.image_to_numpy(img, img.channels), cfg))
    return masks, name
