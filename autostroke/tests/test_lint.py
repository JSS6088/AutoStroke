"""Names referenced by property definitions actually exist.

This file exists because of a specific failure: a text splice that removed a block from
props.py took `_redraw` with it, two properties still pointed `update=_redraw` at it, and
the addon then failed to register with `name '_redraw' is not defined` -- visible only on
enabling it in Blender.

Nothing caught it. props.py imports bpy so no suite could import it, and the pyflakes run
was being filtered with `grep -v "undefined name"` to hide Blender's annotation noise --
which hid the real report along with it.

So: a dependency-free AST check for exactly that class of error, plus a pyflakes pass with
the bogus names listed by name rather than suppressed as a class.

Run directly: python3 autostroke/tests/test_lint.py
"""

import ast
import builtins
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Blender writes property annotations as `name: IntProperty(subtype='ANGLE', ...)`.
# pyflakes parses the annotation as a forward reference and reports the subtype and enum
# strings as undefined names. Listed individually on purpose: a blanket filter is what
# hid the real one.
PYFLAKES_OK = {"ANGLE", "DIR_PATH", "PERCENTAGE", "Resolution", "Shaded",
               "GPU", "CPU"}   # the bake_device enum identifiers

# keywords in a bpy.props call that take a module-level callable
CALLBACK_KW = ("update", "items", "get", "set", "poll")

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def modules():
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def module_names(tree):
    """Everything defined or imported at module level."""
    out = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, (ast.FunctionDef, ast.Lambda)):
            pass
    return out


def main():
    print("\nPROPERTY CALLBACKS RESOLVE")
    checked = 0
    for path in modules():
        rel = os.path.relpath(path, os.path.dirname(ROOT))
        tree = ast.parse(open(path).read())
        defined = module_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.annotation, ast.Call):
                continue
            for kw in node.annotation.keywords:
                if kw.arg in CALLBACK_KW and isinstance(kw.value, ast.Name):
                    checked += 1
                    prop = node.target.id if isinstance(node.target, ast.Name) else "?"
                    check("%s: %s(%s=%s)" % (rel, prop, kw.arg, kw.value.id),
                          kw.value.id in defined,
                          "" if kw.value.id in defined else "NOT DEFINED in this module")
    check("some callbacks were actually inspected", checked > 0, "%d found" % checked)

    print("\nPYFLAKES, WITH THE BOGUS NAMES LISTED RATHER THAN SUPPRESSED")
    try:
        import pyflakes.api
        import pyflakes.reporter
    except ImportError:
        print("   pyflakes not installed -- skipped (the AST check above still ran)")
    else:
        import io
        import re
        buf = io.StringIO()
        rep = pyflakes.reporter.Reporter(buf, buf)
        for path in modules():
            pyflakes.api.checkPath(path, rep)
        real = []
        for line in buf.getvalue().splitlines():
            if "syntax error in forward annotation" in line:
                continue
            m = re.search(r"undefined name '([^']+)'", line)
            if m and m.group(1) in PYFLAKES_OK:
                continue
            if line.strip():
                real.append(line)
        check("no unexplained pyflakes reports", not real,
              real[0] if real else "%d bogus reports accounted for" % 0)
        for line in real[:8]:
            print("      %s" % line)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
