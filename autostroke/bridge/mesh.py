"""Evaluated mesh -> the arrays the sampler and baker need.

Replaces the Geometry Nodes seeding path. Seeds are generated in Python now, so all the
node-tree machinery (the shipped .blend, socket name->identifier mapping, the depsgraph
point-cloud walk) is gone.
"""

import hashlib
from contextlib import contextmanager

import numpy as np
import bpy


class MeshError(RuntimeError):
    """Raised with a message meant for the artist, not the console."""


# Modifiers whose viewport result can differ from their render result through a LEVEL
# rather than a visibility toggle: (viewport attribute, render attribute).
RENDER_LEVELS = {'SUBSURF': ("levels", "render_levels"),
                 'MULTIRES': ("levels", "render_levels")}


def render_changes(obj):
    """[(modifier, attribute, render value)] needed to make the viewport evaluate `obj` the
    way the Cycles bake does. Empty for most meshes, so render_state() is then free.

    Cycles evaluates with RENDER settings: a modifier's show_render, and Subdivision /
    Multires at render_levels. It also bakes with geometry-nodes modifiers switched off
    (position.disabled_geometry_nodes). The depsgraph Python can read evaluates with
    VIEWPORT settings. When the two differ -- a Bevel enabled for render only, Subdivision
    at 1 in the viewport and 2 for render -- strokes were placed on one mesh while their
    texels were baked from another, and most strokes were rejected (measured on a staircase
    with a render-only Bevel: 45% coverage, 88% once both read the same mesh)."""
    changes = []
    for m in obj.modifiers:
        want = False if m.type == 'NODES' else m.show_render
        if m.show_viewport != want:
            changes.append((m, "show_viewport", want))
        if m.type in RENDER_LEVELS and want:
            vp, rd = RENDER_LEVELS[m.type]
            if getattr(m, vp) != getattr(m, rd):
                changes.append((m, vp, getattr(m, rd)))
    return changes


@contextmanager
def render_state(obj):
    """Evaluate `obj` as the bake sees it, then put every setting back exactly.

    Everything that reads the mesh -- the signature, the seeds, the live preview's own
    draw mesh -- goes through this, so strokes, reach sets, the preview and the Cycles
    position/normal maps all describe the SAME geometry. A no-op when viewport and render
    already agree."""
    changes = render_changes(obj)
    saved = [(m, attr, getattr(m, attr)) for m, attr, _ in changes]
    try:
        for m, attr, value in changes:
            setattr(m, attr, value)
        if changes:
            bpy.context.view_layer.update()
        yield bool(changes)
    finally:
        for m, attr, value in reversed(saved):
            setattr(m, attr, value)
        if changes:
            bpy.context.view_layer.update()


def signature(obj, extra=""):
    """A fingerprint of the geometry a bake will actually see.

    Hashes the EVALUATED mesh -- vertex positions, connectivity and active UVs -- rather
    than counting the BASE mesh's vertices and faces. Counts cannot see a vertex move, a
    re-unwrap onto the same UV layer (the old key held the layer's NAME, not its
    contents), or any modifier change that leaves base counts alone: Displace strength,
    Subdivision levels, a Shrinkwrap target moving, armature or shape-key deformation.

    Getting that wrong is worse than a stale map. Seeds are rebuilt from the evaluated
    mesh on every bake while these maps are cached, so a missed change resolves fresh
    strokes against a texel->3D mapping for geometry that no longer exists.

    Loop vertex indices are in the hash so a connectivity change with no vertex movement
    -- an edge flip -- still registers. Array sizes are implicit in the bytes, so every
    count change the old key caught is still caught.
    """
    if obj is None or obj.type != 'MESH':
        raise MeshError("AutoStroke needs a mesh object")
    with render_state(obj):
        return _signature(obj, extra)


def _signature(obj, extra):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        if not me.uv_layers.active:
            raise MeshError(
                "%s has no active UV map. Unwrap it first (Smart UV Project is fine)."
                % obj.name)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        loop_v = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loop_v)
        uv = np.empty(len(me.uv_layers.active.data) * 2, np.float32)
        me.uv_layers.active.data.foreach_get("uv", uv)
    finally:
        ev.to_mesh_clear()
    h = hashlib.blake2b(digest_size=16)
    for part in (co, loop_v, uv):
        h.update(part.tobytes())
    h.update(str(extra).encode("utf-8"))
    return h.hexdigest()


def read_triangles(obj):
    """Triangle corners, per-triangle face id and normal, UVs, and world-space areas.

    Positions come back in OBJECT space, matching the position map (which bakes
    Geometry->Position through a world->object Vector Transform). Areas are computed in
    WORLD space, because stroke density is per m^2 and has to respect object scale.
    """
    if obj is None or obj.type != 'MESH':
        raise MeshError("AutoStroke needs a mesh object")
    with render_state(obj):
        return _read_triangles(obj)


def _read_triangles(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        n_v, n_t = len(me.vertices), len(me.loop_triangles)
        if n_t == 0:
            raise MeshError("%s has no faces" % obj.name)
        if not me.uv_layers.active:
            raise MeshError(
                "%s has no active UV map. Unwrap it first (Smart UV Project is fine)."
                % obj.name)

        co = np.empty(n_v * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)

        tri_v = np.empty(n_t * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tri_v)
        tri_v = tri_v.reshape(-1, 3)

        tri_loops = np.empty(n_t * 3, np.int32)
        me.loop_triangles.foreach_get("loops", tri_loops)
        tri_loops = tri_loops.reshape(-1, 3)

        face_of_tri = np.empty(n_t, np.int32)
        me.loop_triangles.foreach_get("polygon_index", face_of_tri)

        tri_nrm = np.empty(n_t * 3, np.float32)
        me.loop_triangles.foreach_get("normal", tri_nrm)
        tri_nrm = tri_nrm.reshape(-1, 3)

        # SHADING normals, per loop corner: what the surface looks like, as opposed to
        # the flat facet normal above. On a smooth-shaded mesh these differ across a
        # single triangle, and a stroke that carried the facet normal instead painted a
        # flat patch into a curved surface -- the faceting artefact. Custom split normals
        # and Auto Smooth land here too, since this is the evaluated mesh.
        try:
            corner_n = np.empty(len(me.loops) * 3, np.float32)
            me.corner_normals.foreach_get("vector", corner_n)
            corner_n = corner_n.reshape(-1, 3)
        except (AttributeError, RuntimeError, ValueError):
            corner_n = None          # older build: fall back to the facet normal

        uv_all = np.empty(len(me.uv_layers.active.data) * 2, np.float32)
        me.uv_layers.active.data.foreach_get("uv", uv_all)
        uv_all = uv_all.reshape(-1, 2)
    finally:
        ev.to_mesh_clear()

    P, Q, R = co[tri_v[:, 0]], co[tri_v[:, 1]], co[tri_v[:, 2]]
    uvP, uvQ, uvR = (uv_all[tri_loops[:, i]] for i in range(3))
    if corner_n is None:
        nP = nQ = nR = tri_nrm
    else:
        nP, nQ, nR = (corner_n[tri_loops[:, i]] for i in range(3))

    # world-space area: density is strokes per m^2, so object scale must count
    M = np.array(obj.matrix_world.to_3x3())
    Pw, Qw, Rw = P @ M.T, Q @ M.T, R @ M.T
    area_w = 0.5 * np.linalg.norm(np.cross(Qw - Pw, Rw - Pw), axis=1)

    n_faces = int(face_of_tri.max()) + 1
    face_area_w = np.bincount(face_of_tri, weights=area_w, minlength=n_faces)
    return dict(P=P, Q=Q, R=R, uvP=uvP, uvQ=uvQ, uvR=uvR, normal=tri_nrm,
                nP=nP, nQ=nQ, nR=nR,
                face_of_tri=face_of_tri.astype(np.int64), tri_area_world=area_w,
                face_area_world=face_area_w, n_faces=n_faces)


def barycentric_weights(pos, P, Q, R):
    """(u, v, w) of each point in its own triangle, summing to 1."""
    v0, v1, v2 = Q - P, R - P, pos - P
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return 1.0 - v - w, v, w


def barycentric_uv(pos, P, Q, R, uvP, uvQ, uvR):
    """UV of each stroke, from its barycentric coordinates in its own triangle.

    The baker needs a rep-UV per seed to write into the indirection map, and it must be
    the UV of the point itself -- not the triangle's average, which would land in the
    wrong place for any triangle spanning a UV seam.
    """
    u, v, w = barycentric_weights(pos, P, Q, R)
    return (u[:, None] * uvP + v[:, None] * uvQ + w[:, None] * uvR).astype(np.float32)


def barycentric_normal(pos, P, Q, R, nP, nQ, nR):
    """Shading normal at each stroke, interpolated from its triangle's corner normals.

    Same weights as the UV, so a stroke's normal describes the surface at the point it
    actually sits on. Renormalised because interpolating unit vectors does not give one.
    On a flat-shaded mesh all three corners carry the facet normal and this is a no-op.
    """
    u, v, w = barycentric_weights(pos, P, Q, R)
    n = u[:, None] * nP + v[:, None] * nQ + w[:, None] * nR
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)
