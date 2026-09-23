"""AutoStroke — painterly flat-per-cell texture baking for Blender."""

bl_info = {
    "name": "AutoStroke",
    "author": "Jason",
    "version": (0, 10, 7),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > AutoStroke",
    "description": "Bake painterly flat-per-cell indirection maps from a mesh",
    "category": "Material",
}

from . import livepreview, props
from .ops import setup, bake, material
from .ui import panel

_modules = (props, setup, bake, material, livepreview, panel)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
