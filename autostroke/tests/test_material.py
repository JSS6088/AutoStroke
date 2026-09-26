"""Texture filtering in the showcase material.

Only the indirection map must be sampled unfiltered: it stores UV POINTERS, and a blend
of two pointers points nowhere. The normal map is not read through it -- it is sampled
with the mesh's own UVs -- so Closest there only buys texel stair-stepping along every
cell edge.

This is a table test, not a Blender test: it checks the rule the material builder applies,
which is the part that can silently regress. Run directly:
    python3 autostroke/tests/test_material.py
"""

import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILED = []


def check(name, ok, detail=""):
    print("   %-52s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def main():
    print("\nMATERIAL FILTERING\n")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ops", "material.py")).read()
    table = re.search(r"INTERPOLATION = \{(.*?)\n\}", src, re.S)
    if table is None:
        check("the INTERPOLATION table exists", False)
        print("\nFAILED\n")
        return 1
    rules = dict(re.findall(r'"(\w+)":\s*\'(\w+)\'', table.group(1)))

    check("the indirection map is never filtered",
          rules.get("stroke_indirection") == "Closest",
          "-> %s" % rules.get("stroke_indirection"))
    check("the normal map is filtered Linear (Cubic blurs stroke edges)",
          rules.get("stroke_normal") == "Linear", "-> %s" % rules.get("stroke_normal"))

    # An unknown map must fall back to Closest: unfiltered is the safe default for a map
    # whose meaning the builder does not know.
    m = re.search(r"node\.interpolation = INTERPOLATION\.get\(key, '(\w+)'\)", src)
    check("an unlisted map defaults to Closest", m is not None and m.group(1) == "Closest",
          "-> %s" % (m.group(1) if m else "no fallback found"))

    # One place sets it, so a rebuild of an existing material cannot disagree with a
    # fresh build -- that divergence is exactly how a stale Closest would survive.
    check("interpolation is set in exactly one place",
          len(re.findall(r"node\.interpolation\s*=", src)) == 1,
          "%d assignment(s)" % len(re.findall(r"node\.interpolation\s*=", src)))

    legacy_tags()

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


class FakeNode(dict):
    """Enough of a Blender node for _tagged(): an id, and custom properties."""
    bl_idname = "ShaderNodeTexImage"


class FakeTree:
    def __init__(self, nodes):
        self.nodes = nodes


def legacy_tags():
    """A material built before 0.7.0 must be RE-POINTED, never rebuilt.

    build() decides between the two by asking _tagged() for a node. If the old tag went
    unrecognised, that check would say "nothing of ours is here" and _build_fresh would
    run -- and it opens with nt.nodes.clear(), taking any hand-wiring with it. So the
    migration is what stands between a rename and a wiped node graph.

    _tagged is lifted straight out of the shipping source rather than reimplemented here:
    importing ops.material would drag in bpy through its whole relative-import chain, and
    a reimplementation would pass while the real thing was broken.
    """
    print()
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ops", "material.py")).read()
    tree = ast.parse(src)
    wanted = ("TAG", "LEGACY_KEYS")
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == "_tagged")
            or (isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) in wanted for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<material>", "exec"), ns)
    tagged, TAG = ns["_tagged"], ns["TAG"]

    old_n, old_i = FakeNode({TAG: "debug_normal"}), FakeNode({TAG: "indirection"})
    tree_ = FakeTree([old_n, old_i])
    check("a pre-0.7.0 normal node is found under the new key",
          tagged(tree_, "stroke_normal") is old_n)
    check("a pre-0.7.0 indirection node is found under the new key",
          tagged(tree_, "stroke_indirection") is old_i)
    check("and both are re-tagged in place, so the rename happens once",
          old_n[TAG] == "stroke_normal" and old_i[TAG] == "stroke_indirection",
          "%s / %s" % (old_n[TAG], old_i[TAG]))
    check("a node already on the new tag is still found",
          tagged(FakeTree([FakeNode({TAG: "stroke_normal"})]), "stroke_normal") is not None)
    check("an untagged node is still ignored",
          tagged(FakeTree([FakeNode()]), "stroke_normal") is None)


if __name__ == "__main__":
    sys.exit(main())
