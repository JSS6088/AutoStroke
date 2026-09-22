"""The modal bake operator: position -> seeds -> indirection -> maps."""

import os
import re
import time
from contextlib import contextmanager

import numpy as np
import bpy

from ..core.config import Config
from ..core import baker, geometry
from ..bridge import brushes as brush_bridge, cache as seed_cache
from ..bridge import images as bi, mesh as mesh_bridge
from ..bridge import position as pos_bridge
from ..bridge import seeds as seed_bridge
from . import setup as setup_ops

SIG_KEY = "autostroke_position_sig"


class Stages:
    """Wall clock per stage, so a slow bake says WHICH part was slow.

    The panel's estimate models only the numpy stages, because those are the ones that can
    be measured without Blender. Everything else -- reading the cached maps back through
    img.pixels, decoding the brushes, writing two EXRs, the Cycles bake, and the modal
    loop's duty cycle -- is invisible to it, and a bake that predicts 1s and takes 7s is
    entirely that gap. Whatever is left over after the named stages lands in `untimed`,
    which is the honest place for it.
    """

    def __init__(self):
        self.t = {}

    @contextmanager
    def __call__(self, name):
        t0 = time.time()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.time() - t0

    def add(self, name, secs):
        self.t[name] = self.t.get(name, 0.0) + secs

    def line(self, total, floor=0.05, top=5):
        """The biggest few, for the panel. A label longer than the panel is unreadable,
        and the tail is never the thing you are hunting."""
        ranked = sorted(self.t.items(), key=lambda kv: -kv[1])
        bits = ["%s %.1fs" % (k, v) for k, v in ranked if v >= floor][:top]
        untimed = total - sum(self.t.values())
        if untimed >= floor:
            bits.append("untimed %.1fs" % untimed)
        return " · ".join(bits) or "all stages under %.0fms" % (1000 * floor)

    def report(self, total):
        """Every stage, to the system console -- where there is room for all of it."""
        print("AutoStroke bake %.2fs:" % total)
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]):
            print("   %-18s %7.2fs  %4.0f%%" % (k, v, 100 * v / max(total, 1e-9)))
        untimed = total - sum(self.t.values())
        print("   %-18s %7.2fs  %4.0f%%   (modal gaps, Blender overhead)"
              % ("untimed", untimed, 100 * untimed / max(total, 1e-9)))


def safe_name(name):
    """Object name -> a filename stem that cannot escape the working directory."""
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name).strip('._') or "object"


def resolve_working_dir(st):
    """Absolute working directory, or raise with a message meant for the artist."""
    raw = st.working_dir or "//AutoStroke/"
    if raw.startswith("//") and not bpy.data.filepath:
        raise RuntimeError(
            "Working Dir is relative (%s) but this .blend has never been saved. "
            "Save the file, or set an absolute path." % raw)
    path = bpy.path.abspath(raw)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise RuntimeError("Cannot create Working Dir %s: %s" % (path, e))
    if not os.access(path, os.W_OK):
        raise RuntimeError("Working Dir is not writable: %s" % path)
    return path


def config_from_settings(st, brush_mask):
    """Panel settings -> core Config.

    Stroke Size IS mask_scale now: each stroke's radius already equals the patch it owns,
    so "1.0 = stamps just touch" needs no calibration. The old face-widths conversion
    described radius = sqrt(face area), which is no longer how radius is computed.
    """
    cfg = Config(
        mask_scale=float(st.stroke_size),
        target_strokes=int(st.target_strokes),
        min_strokes=int(st.min_strokes),
        max_strokes=int(st.max_strokes),
        stamp_rotate_deg=float(np.degrees(st.flow_angle)),
        rot_jitter_deg=float(np.degrees(st.rot_jitter)),
        crease_angle_deg=float(np.degrees(st.crease_angle)),
        size_random=float(st.size_random),
    )
    return cfg


def load_brush(st, cfg):
    """The chosen brush set as a list of (H,W) masks, plus the set's name."""
    return brush_bridge.load_masks(st, cfg)


def mesh_signature(obj):
    """A hash of the EVALUATED mesh -- see mesh.signature for what counting could not see.

    Deliberately carries NO resolution, so it can key two different things: the position
    and normal maps (which append the resolution), and the seed/direction cache (which
    does not depend on resolution at all). Computed once per bake, since it costs a
    to_mesh.
    """
    return mesh_bridge.signature(obj)


class AUTOSTROKE_OT_bake(bpy.types.Operator):
    bl_idname = "autostroke.bake"
    bl_label = "Bake"
    bl_description = "Bake the indirection map and its debug maps"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None

    @classmethod
    def poll(cls, context):
        return not setup_ops.validate(context.active_object)

    # ---- setup ------------------------------------------------------------
    def invoke(self, context, event):
        st = context.scene.autostroke
        obj = context.active_object
        self._t0 = time.time()
        self.stages = Stages()
        try:
            self.workdir = resolve_working_dir(st)
            self.stem = safe_name(obj.name)
            with self.stages("brushes"):
                self.mask, self.brush_set = load_brush(st, Config())
            self.cfg = config_from_settings(st, self.mask)
            res = int(st.resolution)

            # --- position map: cached against mesh + UV + resolution ---------
            with self.stages("signature"):
                mesh_sig = mesh_signature(obj)
                sig = "%s|res=%d" % (mesh_sig, res)
            # keyed per resolution, or previewing and baking would evict each other's
            # position map on every switch
            sig_key = "%s_%d" % (SIG_KEY, res)
            path = os.path.join(self.workdir, "%s_position.exr" % self.stem)
            npath = os.path.join(self.workdir, "%s_surface_normal.exr" % self.stem)
            cached = (not st.force_position_bake and obj.get(sig_key) == sig
                      and os.path.exists(path) and os.path.exists(npath))
            if cached:
                # Not free: two 1024^2 float images come back through img.pixels, which
                # is a Python-side copy of ~8M floats.
                with self.stages("read maps"):
                    pos_img = bi.image_to_numpy(
                        bpy.data.images.load(path, check_existing=True), 3)
                    self.surf_nrm = bi.image_to_numpy(
                        bpy.data.images.load(npath, check_existing=True), 3)
                self.pos_note = "position: cached"
            else:
                context.window.cursor_set('WAIT')
                with self.stages("cycles bake"):
                    pos_img, img = pos_bridge.bake(obj, res, channel="position")
                    bi.save_image(img, path)
                # The TRUE surface normal, so texels no stamp covered fall back to real
                # shading instead of the (-1,-1,-1) that an all-zero debug map decodes to.
                with self.stages("cycles bake"):
                    self.surf_nrm, nimg = pos_bridge.bake(obj, res, channel="normal")
                    bi.save_image(nimg, npath)
                obj[sig_key] = sig
                context.window.cursor_set('DEFAULT')
                self.pos_note = "position: baked"

            # --- seeds and direction: cached across look changes -------------
            # Neither depends on resolution, brush, Stroke Size, Size Variation, rotation,
            # jitter or cutoff -- every look dial except Stroke Count and the clamps. The
            # direction field is ~N^1.7, so at high stroke counts this IS the preview
            # budget. Keyed on the evaluated-mesh hash, so a moved vertex invalidates it.
            ckey = seed_cache.key(mesh_sig, self.cfg.target_strokes, self.cfg.min_strokes,
                                  self.cfg.max_strokes, self.cfg.aspect_alpha)
            hit = seed_cache.get(ckey)
            if hit is not None:
                seeds, sstats, seed_tan, self.curv = hit
            else:
                with self.stages("seeds"):
                    context.view_layer.update()
                    seeds, sstats = seed_bridge.build_seeds(
                        obj, self.cfg.target_strokes, min_strokes=self.cfg.min_strokes,
                        max_strokes=self.cfg.max_strokes, alpha=self.cfg.aspect_alpha)
                seed_tan, self.curv = None, None
                if self.cfg.direction_source == "curvature":
                    with self.stages("direction"):
                        seed_tan, self.curv = geometry.compute_direction_field(
                            seeds["position"].astype(np.float32),
                            seeds["normal"].astype(np.float32), self.cfg)
                seed_cache.put(ckey, seeds, sstats, seed_tan, self.curv)
            self.seeds = seeds
            self.sstats = sstats
            self.seed_tan = seed_tan

            # Flow angle only: lining each brush up with the stroke direction is now
            # resolve_uv's job, because every brush in the set is drawn at its own angle
            # and needs its own offset.
            stamp_rot = self.cfg.stamp_rotate_deg

            # --- texel setup -------------------------------------------------
            H, W, _ = pos_img.shape
            self.H, self.W = H, W
            flat = pos_img.reshape(-1, 3)
            self.valid = ((np.abs(flat) > self.cfg.bg_eps).any(1)
                          & np.isfinite(flat).all(1))
            ys, xs = np.divmod(np.arange(H * W), W)
            self.uv_self = np.empty((H * W, 2), np.float32)
            self.uv_self[:, 0] = (xs + 0.5) / W
            self.uv_self[:, 1] = 1.0 - (ys + 0.5) / H

            seed_r = seeds["radius"].astype(np.float32)
            # The crease guard compares each texel's own normal against the stroke's, so
            # it needs the surface normal map -- already baked above for the uncovered
            # fallback, and in the same object space as the seed normals.
            flat_n = self.surf_nrm.reshape(-1, 3)

            self.gen = baker.resolve_uv(
                flat[self.valid], self.uv_self[self.valid],
                seeds["position"].astype(np.float32), seeds["id"].astype(np.int64),
                seeds["UVMap"].astype(np.float32), seed_r,
                seeds["normal"].astype(np.float32),
                None, self.mask, self.cfg, seed_tan, stamp_rot,
                pt_nrm=flat_n[self.valid])
        except (RuntimeError, mesh_bridge.MeshError) as e:
            context.window.cursor_set('DEFAULT')
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0.0, 1.0)
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        self._progress = 0.0
        return {'RUNNING_MODAL'}

    # ---- the chunked loop --------------------------------------------------
    def modal(self, context, event):
        if event.type == 'ESC':
            return self._finish(context, cancelled=True)
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        deadline = time.time() + 0.1          # keep the UI at ~10fps
        t0 = time.time()
        try:
            while time.time() < deadline:
                self._progress = next(self.gen)
        except StopIteration as stop:
            self.stages.add("resolve", time.time() - t0)
            self.result = stop.value
            return self._finish(context)
        self.stages.add("resolve", time.time() - t0)
        context.window_manager.progress_update(self._progress)
        context.area.header_text_set("AutoStroke  %.0f%%   (Esc to cancel)"
                                     % (100 * self._progress))
        return {'RUNNING_MODAL'}

    # ---- teardown ----------------------------------------------------------
    def _finish(self, context, cancelled=False):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        if context.area:
            context.area.header_text_set(None)
        if cancelled:
            self.report({'WARNING'}, "Bake cancelled")
            return {'CANCELLED'}
        try:
            self._write(context)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}

    def _write(self, context):
        st = context.scene.autostroke
        uv_out, dbg_out, covered, lum_out = self.result
        H, W, cfg, valid = self.H, self.W, self.cfg, self.valid

        indir = np.zeros((H * W, 3), np.float32)
        indir[:, :2] = self.uv_self
        indir[:, 2] = 0.5
        indir[valid, :2] = uv_out
        indir[valid, 2] = lum_out
        # Uncovered texels keep the TRUE surface normal rather than black: those regions
        # are pass-through, so they should shade like the untouched model.
        dbg = np.zeros((H * W, 3), np.float32)
        dbg[valid] = dbg_out
        nrm = self.surf_nrm.reshape(-1, 3)
        nl = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.divide(nrm, nl, out=np.zeros_like(nrm), where=nl > 1e-9)
        uncovered = np.zeros(H * W, bool)
        uncovered[np.flatnonzero(valid)[~covered]] = True
        dbg[uncovered] = (nrm[uncovered] * 0.5 + 0.5).astype(np.float32)

        vimg = valid.reshape(H, W)
        with self.stages("dilate"):
            maps = {
                "stroke_indirection":
                    baker.dilate_fill(indir.reshape(H, W, 3), vimg, cfg.dilate_px),
                "stroke_normal":
                    baker.dilate_fill(dbg.reshape(H, W, 3), vimg, cfg.dilate_px),
            }
        # numpy -> image datablock (img.pixels again) -> EXR on disk
        with self.stages("write maps"):
            for key, arr in maps.items():
                bi.write_map(arr, "%s_%s" % (self.stem, key),
                             os.path.join(self.workdir, "%s_%s.exr" % (self.stem, key)))

        n_val, n_cov = int(valid.sum()), int(covered.sum())
        secs = time.time() - self._t0
        ss = self.sstats
        st.last_report = (
            "%.1f%% coverage · %s strokes · %s|"
            "stroke radius %.4f u · faces at min %.0f%%, at max %.0f%%|"
            "brushes: %s (%d) · %s|%s"
            % (100.0 * n_cov / max(n_val, 1), "{:,}".format(ss["strokes"]),
               ("%.0fs" % secs) if secs < 90 else ("%.1f min" % (secs / 60)),
               ss["radius_med"], 100 * ss["at_min"], 100 * ss["at_max"],
               self.brush_set, len(self.mask), self.pos_note,
               self.stages.line(secs)))
        self.stages.report(secs)
        # Feed the real time back so the panel's estimate learns this machine and model.
        from ..core import cost
        st.est_scale = cost.calibration(
            secs, cost.bake_seconds(int(st.resolution), max(ss["strokes"], 1),
                                    st.stroke_size, scale=1.0),
            previous=st.est_scale)
        obj = context.active_object
        try:
            from . import material as material_ops
            with self.stages("material"):
                material_ops.build(obj)
        except RuntimeError as e:
            self.report({'WARNING'}, "Maps baked but material failed: %s" % e)

        self.report({'INFO'}, "AutoStroke: %.1f%% coverage in %.0fs"
                    % (100.0 * n_cov / max(n_val, 1), secs))


classes = (AUTOSTROKE_OT_bake,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
