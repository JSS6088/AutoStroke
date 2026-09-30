"""Multi-material objects: every material gets the strokes, originals stay untouched.

Runs headless (material nodes need no GPU), in an empty scene built here, and never saves:

    Blender -b --factory-startup --python tools/material_check.py

It builds one object with four slots -- a textured Principled material, a plain-colour
Principled one, an Emission-only one (no Principled BSDF) and an empty slot -- plus a
second object sharing the textured material, fakes the two baked maps in memory, and runs
ops/material.build() on it. Prints PASS/FAIL per check; exit code 1 on any failure.
"""

import os
import sys

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(name, ok, detail=""):
    print("   %-66s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def principled(mat):
    return next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")


def make_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    me = bpy.data.meshes.new("Crate")
    me.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0),
                    (3, 0, 0), (3, 1, 0), (4, 0, 0), (4, 1, 0)], [],
                   [(0, 1, 2, 3), (1, 4, 5, 2), (4, 6, 7, 5), (6, 8, 9, 7)])
    me.uv_layers.new(name="UVMap")
    obj = bpy.data.objects.new("Crate", me)
    bpy.context.scene.collection.objects.link(obj)

    wood = bpy.data.materials.new("Wood"); wood.use_nodes = True
    albedo = wood.node_tree.nodes.new("ShaderNodeTexImage")
    albedo.image = bpy.data.images.new("wood_albedo", 8, 8)
    wood.node_tree.links.new(albedo.outputs["Color"], principled(wood).inputs["Base Color"])
    bump = wood.node_tree.nodes.new("ShaderNodeBump")
    wood.node_tree.links.new(bump.outputs["Normal"], principled(wood).inputs["Normal"])
    metal = bpy.data.materials.new("Metal"); metal.use_nodes = True
    principled(metal).inputs["Base Color"].default_value = (0.2, 0.3, 0.4, 1.0)
    principled(metal).inputs["Roughness"].default_value = 0.25
    emit = bpy.data.materials.new("Emit"); emit.use_nodes = True
    nt = emit.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); e = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(e.outputs[0], out.inputs["Surface"])
    for m in (wood, metal, emit, None):
        me.materials.append(m)
    for i, p in enumerate(me.polygons):
        p.material_index = i

    other = bpy.data.objects.new("Other", bpy.data.meshes.new("OtherMesh"))
    other.data.materials.append(wood)
    bpy.context.scene.collection.objects.link(other)

    for key in ("stroke_normal", "stroke_indirection"):
        bpy.data.images.new("Crate_%s" % key, 8, 8, float_buffer=True)
    return obj, wood, metal, emit, other


def main():
    sys.path.insert(0, REPO)
    import autostroke
    autostroke.register()
    from autostroke.ops import material as M

    obj, wood, metal, emit, other = make_scene()
    wood_nodes = len(wood.node_tree.nodes)
    print("\nBUILD ON A FOUR-SLOT OBJECT")
    msg = M.build(obj)
    print("   report: %s" % msg)
    s = [slot.material for slot in obj.material_slots]

    check("textured slot gets a per-object copy of Wood",
          s[0] is not wood and s[0].get(M.SOURCE) == "Wood" and s[0].get(M.OWNER) == "Crate")
    check("...and Wood itself is untouched, still shared by Other",
          len(wood.node_tree.nodes) == wood_nodes and other.material_slots[0].material is wood
          and not any(n.get(M.TAG) for n in wood.node_tree.nodes))
    bs = principled(s[0])
    nlink = bs.inputs["Normal"].links[0].from_node if bs.inputs["Normal"].links else None
    check("stroke normal map -> Normal Map (object space) -> the BSDF's Normal",
          nlink is not None and nlink.bl_idname == "ShaderNodeNormalMap" and nlink.space == 'OBJECT'
          and nlink.inputs["Color"].links[0].from_node.get(M.TAG) == "stroke_normal")
    mul = bs.inputs["Base Color"].links[0].from_node if bs.inputs["Base Color"].links else None
    check("Base Color = the material's own albedo x the per-stroke tone",
          mul is not None and mul.get(M.TAG) == "stroke_tone_multiply"
          and mul.inputs[6].links[0].from_node.image.name == "wood_albedo"
          and mul.inputs[7].links[0].from_node.get(M.TAG) == "stroke_tone")
    mbs = principled(s[1])
    mmul = mbs.inputs["Base Color"].links[0].from_node
    check("plain-colour material keeps its colour and roughness under the tone",
          tuple(round(v, 3) for v in mmul.inputs[6].default_value) == (0.2, 0.3, 0.4, 1.0)
          and abs(mbs.inputs["Roughness"].default_value - 0.25) < 1e-6)
    grey = bpy.data.materials.get("Crate_AutoStroke")
    check("Emission-only material falls back to the grey AutoStroke material",
          s[2] == grey and "Emit" in msg)
    check("empty slot gets the grey material too", s[3] == grey)

    print("\nRE-BUILD (EVERY RE-BAKE DOES THIS)")
    n_mats = len(bpy.data.materials)
    n_nodes = len(s[0].node_tree.nodes)
    M.build(obj)
    s2 = [slot.material for slot in obj.material_slots]
    check("no new materials, no duplicated nodes",
          len(bpy.data.materials) == n_mats and s2 == s
          and len(s[0].node_tree.nodes) == n_nodes)

    print("\nDUPLICATED AFTER A BAKE")
    dup = obj.copy(); dup.data = obj.data.copy(); dup.name = "CrateCopy"
    bpy.context.scene.collection.objects.link(dup)
    for key in ("stroke_normal", "stroke_indirection"):
        bpy.data.images.new("CrateCopy_%s" % key, 8, 8, float_buffer=True)
    del dup[M.ORIGINALS]
    M.build(dup)
    d0 = dup.material_slots[0].material
    check("the duplicate's copy traces back to Wood, not to Crate's copy",
          d0.get(M.SOURCE) == "Wood" and d0.get(M.OWNER) == "CrateCopy" and d0 is not s[0])
    tex = next(n for n in d0.node_tree.nodes if n.get(M.TAG) == "stroke_normal")
    check("...and reads its own maps", tex.image.name == "CrateCopy_stroke_normal")

    print("\nREMOVE STROKES")
    M.restore(obj)
    r = [slot.material for slot in obj.material_slots]
    check("every slot is back to what it held before", r == [wood, metal, emit, None],
          str([m.name if m else None for m in r]))
    check("the restore record is cleared", M.ORIGINALS not in obj)

    print("\nOLDER FILES (one grey AutoStroke material in slot 0, nothing recorded)")
    legacy = bpy.data.objects.new("Legacy", bpy.data.meshes.new("LegacyMesh"))
    bpy.context.scene.collection.objects.link(legacy)
    for key in ("stroke_normal", "stroke_indirection"):
        bpy.data.images.new("Legacy_%s" % key, 8, 8, float_buffer=True)
    lgrey = bpy.data.materials.new("Legacy_AutoStroke")
    legacy.data.materials.append(lgrey)
    M.build(legacy)
    check("the grey material stays (the original is unknown)",
          legacy.material_slots[0].material == lgrey)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


code = 1
try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
sys.stdout.flush()
os._exit(code)
