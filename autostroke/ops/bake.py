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


def targets(context):
    """Objects this bake will process, in a stable order.

    Every selected MESH that passes validate(), sorted by name -- context's own
    selection order is Blender's internal selection stack (last-clicked last, or
    arbitrary after undo/linking), which an artist has no way to see or predict, while
    name order is exactly what the Outliner already shows them.

    Falls back to [active_object] when that set is empty, so "nothing extra selected,
    just press Bake" -- every workflow this addon has ever supported -- behaves
    identically to before this function existed. An object can be active without being
    selected (see bridge/position.py's own selectable() docstring for how), so the
    active object is folded in even when it is not in context.selected_objects, matching
    what poll()/the panel have always keyed off.
    """
    seen = {o.name: o for o in context.selected_objects if o.type == 'MESH'}
    active = context.active_object
    if active is not None and active.type == 'MESH':
        seen.setdefault(active.name, active)
    ok = sorted((o for o in seen.values() if not setup_ops.validate(o)),
               key=lambda o: o.name)
    if ok:
        return ok
    if active is not None and not setup_ops.validate(active):
        return [active]
    return []


class Job:
    """One object's worth of bake state -- what used to be flat attributes on the
    operator itself, back when there was only ever one object to bake.

    `t0` is this object's OWN start time, not the batch's: `_write` measures "how long
    did THIS bake take" from it, and that feeds directly into est_scale's calibration
    (which assumes elapsed time corresponds to this object's own stroke count). Sharing
    one clock across every job in a batch would make every object after the first look
    slower than it was and would corrupt the calibration on every one of them.
    """

    def __init__(self, obj):
        self.obj = obj
        self.t0 = None
        self.stem = None
        self.pos_note = None
        self.surf_nrm = None
        self.seeds = None
        self.sstats = None
        self.seed_tan = None
        self.curv = None
        self.H = None
        self.W = None
        self.valid = None
        self.uv_self = None
        self.gen = None
        self.result = None
        self.error = None          # set on failure; the job is skipped, not fatal
        self.secs = None           # set by _write(): this job's own elapsed time
        self.coverage = None       # set by _write(): percent of valid texels covered


class AUTOSTROKE_OT_bake(bpy.types.Operator):
    bl_idname = "autostroke.bake"
    bl_label = "Bake"
    bl_description = "Bake the indirection map and its debug maps for every selected mesh"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None

    @classmethod
    def poll(cls, context):
        return bool(targets(context))

    # ---- setup ------------------------------------------------------------
    def invoke(self, context, event):
        st = context.scene.autostroke
        objs = targets(context)
        if not objs:
            self.report({'ERROR'}, "No valid mesh objects selected")
            return {'CANCELLED'}

        self._batch_t0 = time.time()
        self.stages = Stages()
        self.done, self.failed = [], []
        try:
            # Everything here is scene-derived, not object-dependent -- computed once
            # and shared read-only across every job, exactly as it was the only bake.
            self.workdir = resolve_working_dir(st)
            with self.stages("brushes"):
                self.mask, self.brush_set = load_brush(st, Config())
            self.cfg = config_from_settings(st, self.mask)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        self.jobs = [Job(o) for o in objs]
        self.i = 0

        wm = context.window_manager
        wm.progress_begin(0.0, 1.0)
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        self._progress = 0.0
        if self._start_job(context, self.jobs[0]):
            return {'RUNNING_MODAL'}
        # The very first object failed before it could even start resolving --
        # _advance() is the same recovery path modal() uses mid-batch, so this does
        # not need its own separate handling.
        return self._advance(context)

    def _start_job(self, context, job):
        """Position/normal map -> seeds -> direction field -> the resolve_uv generator,
        for one object. Everything ops/bake.py used to do once in invoke(), now scoped
        to `job` instead of `self` so a batch can run it once per object.

        Returns True on success (job.gen is ready for modal() to advance) or False,
        having recorded job.error, on a caught (RuntimeError, MeshError) -- a failure
        here skips this one object rather than cancelling the whole operator.
        """
        st = context.scene.autostroke
        obj = job.obj
        job.t0 = time.time()
        try:
            job.stem = safe_name(obj.name)
            res = int(st.resolution)

            # --- position map: cached against mesh + UV + resolution ---------
            with self.stages("signature"):
                mesh_sig = mesh_signature(obj)
                sig = "%s|res=%d" % (mesh_sig, res)
            # keyed per resolution, or previewing and baking would evict each other's
            # position map on every switch
            sig_key = "%s_%d" % (SIG_KEY, res)
            path = os.path.join(self.workdir, "%s_position.exr" % job.stem)
            npath = os.path.join(self.workdir, "%s_surface_normal.exr" % job.stem)
            cached = (not st.force_position_bake and obj.get(sig_key) == sig
                      and os.path.exists(path) and os.path.exists(npath))
            if cached:
                # Not free: two 1024^2 float images come back through img.pixels, which
                # is a Python-side copy of ~8M floats.
                with self.stages("read maps"):
                    pos_img = bi.image_to_numpy(
                        bpy.data.images.load(path, check_existing=True), 3)
                    job.surf_nrm = bi.image_to_numpy(
                        bpy.data.images.load(npath, check_existing=True), 3)
                job.pos_note = "position: cached"
            else:
                context.window.cursor_set('WAIT')
                # Exactly one Cycles bake in flight at a time, always: pos_bridge.bake
                # reuses a single shared material+image datablock per channel across
                # every object, and is only safe because each call fully completes and
                # saves before the next one starts -- which invoke()/_advance() never
                # violate, since a new job's position bake only ever begins after the
                # previous job's _write() has already returned.
                with self.stages("cycles bake"):
                    pos_img, img = pos_bridge.bake(obj, res, channel="position")
                    bi.save_image(img, path)
                # The TRUE surface normal, so texels no stamp covered fall back to real
                # shading instead of the (-1,-1,-1) that an all-zero debug map decodes to.
                with self.stages("cycles bake"):
                    job.surf_nrm, nimg = pos_bridge.bake(obj, res, channel="normal")
                    bi.save_image(nimg, npath)
                obj[sig_key] = sig
                context.window.cursor_set('DEFAULT')
                job.pos_note = "position: baked"

            # --- seeds and direction: cached across look changes -------------
            # Neither depends on resolution, brush, Stroke Size, Size Variation, rotation,
            # jitter or cutoff -- every look dial except Stroke Count and the clamps. The
            # direction field is ~N^1.7, so at high stroke counts this IS the preview
            # budget. Keyed on the evaluated-mesh hash, so a moved vertex invalidates it.
            #
            # One slot, deliberately: this evicts whatever the PREVIOUS job in this same
            # batch cached, exactly as switching objects and re-baking already did before
            # a batch could exist. A batch still processes one object at a time in
            # sequence, so an unbounded per-object cache would just be a memory leak with
            # extra steps.
            ckey = seed_cache.key(mesh_sig, self.cfg.target_strokes, self.cfg.min_strokes,
                                  self.cfg.max_strokes, self.cfg.aspect_alpha)
            hit = seed_cache.get(ckey)
            if hit is not None:
                seeds, sstats, seed_tan, job.curv = hit
            else:
                with self.stages("seeds"):
                    context.view_layer.update()
                    seeds, sstats = seed_bridge.build_seeds(
                        obj, self.cfg.target_strokes, min_strokes=self.cfg.min_strokes,
                        max_strokes=self.cfg.max_strokes, alpha=self.cfg.aspect_alpha)
                seed_tan, job.curv = None, None
                if self.cfg.direction_source == "curvature":
                    with self.stages("direction"):
                        seed_tan, job.curv = geometry.compute_direction_field(
                            seeds["position"].astype(np.float32),
                            seeds["normal"].astype(np.float32), self.cfg)
                seed_cache.put(ckey, seeds, sstats, seed_tan, job.curv)
            job.seeds = seeds
            job.sstats = sstats
            job.seed_tan = seed_tan

            # Flow angle only: lining each brush up with the stroke direction is now
            # resolve_uv's job, because every brush in the set is drawn at its own angle
            # and needs its own offset.
            stamp_rot = self.cfg.stamp_rotate_deg

            # --- texel setup -------------------------------------------------
            H, W, _ = pos_img.shape
            job.H, job.W = H, W
            flat = pos_img.reshape(-1, 3)
            job.valid = ((np.abs(flat) > self.cfg.bg_eps).any(1)
                        & np.isfinite(flat).all(1))
            ys, xs = np.divmod(np.arange(H * W), W)
            job.uv_self = np.empty((H * W, 2), np.float32)
            job.uv_self[:, 0] = (xs + 0.5) / W
            job.uv_self[:, 1] = 1.0 - (ys + 0.5) / H

            seed_r = seeds["radius"].astype(np.float32)
            # The crease guard compares each texel's own normal against the stroke's, so
            # it needs the surface normal map -- already baked above for the uncovered
            # fallback, and in the same object space as the seed normals.
            flat_n = job.surf_nrm.reshape(-1, 3)

            job.gen = baker.resolve_uv(
                flat[job.valid], job.uv_self[job.valid],
                seeds["position"].astype(np.float32), seeds["id"].astype(np.int64),
                seeds["UVMap"].astype(np.float32), seed_r,
                seeds["normal"].astype(np.float32),
                None, self.mask, self.cfg, seed_tan, stamp_rot,
                pt_nrm=flat_n[job.valid])
            return True
        except (RuntimeError, mesh_bridge.MeshError) as e:
            context.window.cursor_set('DEFAULT')
            job.error = str(e)
            return False

    # ---- the chunked loop --------------------------------------------------
    def modal(self, context, event):
        if event.type == 'ESC':
            return self._finish(context, cancelled=True)
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        job = self.jobs[self.i]
        deadline = time.time() + 0.1          # keep the UI at ~10fps
        t0 = time.time()
        try:
            while time.time() < deadline:
                self._progress = next(job.gen)
        except StopIteration as stop:
            self.stages.add("resolve", time.time() - t0)
            job.result = stop.value
            try:
                self._write(context, job)
                self.done.append(job)
            except RuntimeError as e:
                self.failed.append((job.obj.name, str(e)))
            return self._advance(context)
        self.stages.add("resolve", time.time() - t0)
        frac = (self.i + self._progress) / len(self.jobs)
        context.window_manager.progress_update(frac)
        context.area.header_text_set(
            "AutoStroke  Object %d/%d: %s — %.0f%%   (Esc to cancel)"
            % (self.i + 1, len(self.jobs), job.obj.name, 100 * self._progress))
        return {'RUNNING_MODAL'}

    def _advance(self, context):
        """Move past self.i, starting the next job's position bake -- only once the
        one that just finished has fully written its output and built its material, so
        two objects' Cycles bakes can never overlap. Skips (rather than aborts on) any
        job whose own setup fails, continuing until one starts or the queue is empty.
        """
        self.i += 1
        while self.i < len(self.jobs):
            job = self.jobs[self.i]
            if self._start_job(context, job):
                return {'RUNNING_MODAL'}
            self.failed.append((job.obj.name, job.error))
            self.i += 1
        return self._finish(context)

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
        self._report_batch(context)
        if not self.done:
            return {'CANCELLED'}
        return {'FINISHED'}

    def _write(self, context, job):
        """Turn one finished job's resolve_uv output into two EXRs plus a material.
        Raises RuntimeError on failure -- the CALLER (modal()) decides whether that
        skips this one object or is fatal; this function itself never touches self.i
        or the job queue."""
        uv_out, dbg_out, covered, lum_out = job.result
        H, W, cfg, valid = job.H, job.W, self.cfg, job.valid

        indir = np.zeros((H * W, 3), np.float32)
        indir[:, :2] = job.uv_self
        indir[:, 2] = 0.5
        indir[valid, :2] = uv_out
        indir[valid, 2] = lum_out
        # Uncovered texels keep the TRUE surface normal rather than black: those regions
        # are pass-through, so they should shade like the untouched model.
        dbg = np.zeros((H * W, 3), np.float32)
        dbg[valid] = dbg_out
        nrm = job.surf_nrm.reshape(-1, 3)
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
                bi.write_map(arr, "%s_%s" % (job.stem, key),
                             os.path.join(self.workdir, "%s_%s.exr" % (job.stem, key)))

        n_val, n_cov = int(valid.sum()), int(covered.sum())
        job.secs = time.time() - job.t0
        job.coverage = 100.0 * n_cov / max(n_val, 1)
        ss = job.sstats
        # Feed the real time back so the panel's estimate learns this machine and model
        # -- once per job, each against its OWN elapsed time, not the batch's.
        from ..core import cost
        st = context.scene.autostroke
        st.est_scale = cost.calibration(
            job.secs, cost.bake_seconds(int(st.resolution), max(ss["strokes"], 1),
                                        st.stroke_size, scale=1.0),
            previous=st.est_scale)
        try:
            from . import material as material_ops
            with self.stages("material"):
                material_ops.build(job.obj)
        except RuntimeError as e:
            self.report({'WARNING'}, "Maps baked but material failed for %s: %s"
                        % (job.obj.name, e))

    def _report_batch(self, context):
        """Assemble st.last_report (still one '|'-joined string, so the panel's
        existing render needs no change) and the final transient self.report().
        Single-object output is byte-identical to before batching existed; anything
        more becomes a summary line plus one compact line per object."""
        st = context.scene.autostroke
        total_secs = time.time() - self._batch_t0

        if len(self.jobs) == 1 and len(self.done) == 1:
            job = self.done[0]
            ss = job.sstats
            st.last_report = (
                "%.1f%% coverage · %s strokes · %s|"
                "stroke radius %.4f u · faces at min %.0f%%, at max %.0f%%|"
                "brushes: %s (%d) · %s|%s"
                % (job.coverage, "{:,}".format(ss["strokes"]),
                   ("%.0fs" % job.secs) if job.secs < 90 else ("%.1f min" % (job.secs / 60)),
                   ss["radius_med"], 100 * ss["at_min"], 100 * ss["at_max"],
                   self.brush_set, len(self.mask), job.pos_note,
                   self.stages.line(job.secs)))
            self.stages.report(job.secs)
            self.report({'INFO'}, "AutoStroke: %.1f%% coverage in %.0fs"
                        % (job.coverage, job.secs))
            return

        lines = ["%d object%s · %d baked, %d failed · %s total" % (
            len(self.jobs), "" if len(self.jobs) == 1 else "s",
            len(self.done), len(self.failed),
            ("%.0fs" % total_secs) if total_secs < 90 else ("%.1f min" % (total_secs / 60)))]
        for job in self.done:
            lines.append("%s: %.1f%% coverage · %s strokes · %.0fs"
                         % (job.obj.name, job.coverage,
                            "{:,}".format(job.sstats["strokes"]), job.secs))
        for name, detail in self.failed:
            lines.append("failed: %s — %s" % (name, detail))
        st.last_report = "|".join(lines)
        self.stages.report(total_secs)

        if self.failed:
            names = ", ".join(n for n, _d in self.failed[:5])
            more = "" if len(self.failed) <= 5 else " (+%d more)" % (len(self.failed) - 5)
            self.report({'WARNING'}, "AutoStroke: %d/%d baked in %.0fs — failed: %s%s"
                        % (len(self.done), len(self.jobs), total_secs, names, more))
        else:
            self.report({'INFO'}, "AutoStroke: %d/%d baked in %.0fs"
                        % (len(self.done), len(self.jobs), total_secs))


classes = (AUTOSTROKE_OT_bake,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
