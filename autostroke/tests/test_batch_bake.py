"""Baking multiple selected objects with the same settings.

The bake operator used to be entirely single-object shaped: one set of instance
attributes, one generator, one object read from context.active_object. Turning that into
a batch means the OBJECT-LEVEL state has to move onto a per-object Job, while the
scene-derived settings (workdir/mask/brush_set/cfg) stay shared -- and the modal loop has
to advance through a QUEUE of jobs without ever letting two Cycles bakes overlap, since
bridge/position.py's bake() reuses one shared material+image datablock per channel across
every object and is only safe when one object's bake fully completes before the next
starts.

None of this touches core/baker.py or the numpy resolve pipeline -- it is operator
plumbing, tested here against a stubbed bpy the way test_import.py/test_selectable.py
already do for bpy-touching code, not against real Cycles or real geometry.

Run directly: python3 autostroke/tests/test_batch_bake.py
"""

import os
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILED = []


def check(name, ok, detail=""):
    print("   %-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


class FakeObj:
    """Just enough of bpy.types.Object for setup_ops.validate()."""

    def __init__(self, name, has_uv=True, has_faces=True):
        self.name = name
        self.type = 'MESH'

        class _Data:
            polygons = [object()] if has_faces else []

            class _UV:
                active = object() if has_uv else None
            uv_layers = _UV()
        self.data = _Data()


def stub_bpy():
    """Enough bpy for ops.setup / ops.bake / ops.material to import, plus the pieces
    modal()/_advance()/_finish() actually touch: window_manager (timer, progress),
    window (cursor), area (header text). Deliberately minimal -- _start_job/_write are
    monkeypatched per-test rather than exercised for real, so bpy.data/materials/images
    are stubbed only where a test specifically needs them (see per_job_timing())."""
    class _Type:
        pass

    def _prop(*a, **kw):
        return ("prop", a, kw)

    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(
        PropertyGroup=type("PropertyGroup", (_Type,), {}),
        Operator=type("Operator", (_Type,), {}),
        Panel=type("Panel", (_Type,), {}),
        Scene=type("Scene", (_Type,), {}),
        Object=type("Object", (_Type,), {}),
    )
    bpy.props = types.SimpleNamespace(**{
        n: _prop for n in ("BoolProperty", "EnumProperty", "FloatProperty", "IntProperty",
                           "PointerProperty", "StringProperty")})
    bpy.utils = types.SimpleNamespace(register_class=lambda c: None,
                                      unregister_class=lambda c: None)
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    bpy.data = types.SimpleNamespace(
        objects={}, images={}, materials={},
        filepath="/tmp/fake.blend")
    bpy.context = None
    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy.types
    sys.modules["bpy.props"] = bpy.props
    sys.modules["bpy.utils"] = bpy.utils
    for name in ("gpu", "gpu_extras", "gpu_extras.batch", "bmesh", "mathutils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    return bpy


def load_bake():
    """ops/bake.py uses `from ..core...` -- it has to be imported as a real submodule
    of the autostroke package (parent-of-autostroke on sys.path, full dotted import),
    not as a bare top-level `ops.bake` (which test_selectable.py's single-dot bridge/
    imports can get away with, but this file's double-dot ones cannot)."""
    sys.path.insert(0, os.path.dirname(ROOT))
    from autostroke.ops import bake as bake_mod   # noqa: E402
    return bake_mod


class FakeWM:
    def __init__(self):
        self.timers = []
        self.progress = []

    def event_timer_add(self, interval, window=None):
        t = object()
        self.timers.append(t)
        return t

    def event_timer_remove(self, t):
        if t in self.timers:
            self.timers.remove(t)

    def modal_handler_add(self, op):
        pass

    def progress_begin(self, a, b):
        pass

    def progress_update(self, f):
        self.progress.append(f)

    def progress_end(self):
        pass


class FakeArea:
    def __init__(self):
        self.header = None

    def header_text_set(self, text):
        self.header = text


class FakeContext:
    def __init__(self, selected=(), active=None):
        self.selected_objects = list(selected)
        self.active_object = active
        self.window_manager = FakeWM()
        self.window = types.SimpleNamespace(cursor_set=lambda mode: None)
        self.area = FakeArea()
        self.scene = types.SimpleNamespace(
            autostroke=types.SimpleNamespace(est_scale=1.0, last_report="", resolution='1024',
                                             stroke_size=4.0),
            render=None)
        self.view_layer = types.SimpleNamespace(update=lambda: None)


def one_shot_gen(progress, result):
    """A resolve_uv-shaped generator: yields `progress` once, then returns `result`."""
    yield progress
    return result


def main():
    stub_bpy()
    bake_mod = load_bake()

    targets_tests(bake_mod)
    sequencing_tests(bake_mod)
    sequencing_test_can_fail(bake_mod)
    partial_failure_tests(bake_mod)
    per_job_timing(bake_mod)
    report_batch_tests(bake_mod)
    device_tests(bake_mod)

    print("\n%s\n" % ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


def targets_tests(bake_mod):
    print("\nTARGETS(): WHICH OBJECTS BAKE WILL PROCESS")
    a, b, c = FakeObj("A"), FakeObj("B", has_uv=False), FakeObj("C")

    ctx = FakeContext(selected=[c, a, b], active=a)
    got = bake_mod.targets(ctx)
    check("mixed valid/invalid selection keeps only the valid ones",
          [o.name for o in got] == ["A", "C"], [o.name for o in got])

    ctx2 = FakeContext(selected=[a], active=a)
    check("selection = just the active object reproduces the single-object list",
          [o.name for o in bake_mod.targets(ctx2)] == ["A"])

    ctx3 = FakeContext(selected=[], active=a)
    check("empty selection falls back to [active_object]",
          [o.name for o in bake_mod.targets(ctx3)] == ["A"])

    ctx4 = FakeContext(selected=[], active=None)
    check("nothing selected and no active object -> empty",
          bake_mod.targets(ctx4) == [])

    ctx5 = FakeContext(selected=[b], active=b)          # only an invalid object
    check("an invalid active object with no other selection -> empty",
          bake_mod.targets(ctx5) == [])

    d = FakeObj("D")
    ctx6 = FakeContext(selected=[a, c], active=d)        # active not in selected_objects
    got6 = [o.name for o in bake_mod.targets(ctx6)]
    check("an active object outside selected_objects still counts (matches poll() today)",
          "D" in got6 and got6 == sorted(got6), got6)


def _wire_stub_job_methods(op, order, fail_on=None):
    """Replace _start_job/_write with marker-recording stand-ins, matching the plan's
    own suggested approach: this proves the SEQUENCING (never two starts before an
    intervening write), not _start_job's real per-object body, which needs real Cycles
    and mesh data that has no place in this pure-Python suite."""
    def start(context, job):
        order.append("start(%s)" % job.obj.name)
        if fail_on and job.obj.name == fail_on:
            job.error = "stub failure"
            return False
        job.gen = one_shot_gen(1.0, ("uv", "dbg", "covered", "lum"))
        job.t0 = time.time()
        return True

    def write(context, job):
        order.append("write(%s)" % job.obj.name)
        job.result = ("uv", "dbg", "covered", "lum")
        job.secs = time.time() - job.t0
        job.coverage = 100.0
        job.sstats = {"strokes": 10, "radius_med": 0.1, "at_min": 0.0, "at_max": 0.0}

    op._start_job = start
    op._write = write


def _drive_to_completion(op, ctx, max_ticks=50):
    """Feed TIMER events until the operator finishes (mirrors what Blender's own event
    loop does, minus everything unrelated to this operator)."""
    event = types.SimpleNamespace(type='TIMER')
    for _ in range(max_ticks):
        result = op.modal(ctx, event)
        if result != {'RUNNING_MODAL'}:
            return result
    raise AssertionError("did not finish within %d ticks" % max_ticks)


def sequencing_tests(bake_mod):
    print("\nJOBS NEVER OVERLAP: WRITE(N) ALWAYS BEFORE START(N+1)")
    objs = [FakeObj(n) for n in ("A", "B", "C")]
    op = bake_mod.AUTOSTROKE_OT_bake()
    op.jobs = [bake_mod.Job(o) for o in objs]
    op.i = 0
    op.stages = bake_mod.Stages()
    op.done, op.failed = [], []
    op._batch_t0 = time.time()
    op._timer = object()
    op.workdir, op.mask, op.brush_set, op.cfg = "/tmp", [None], "standard", None
    op.report = lambda level, msg: None

    order = []
    _wire_stub_job_methods(op, order)
    op._start_job(FakeContext(), op.jobs[0])             # what invoke() would do first
    order.clear()                                        # only care about modal()'s own order
    ctx = FakeContext()
    result = _drive_to_completion(op, ctx)

    check("finishes cleanly", result == {'FINISHED'}, result)
    expected = ["write(A)", "start(B)", "write(B)", "start(C)", "write(C)"]
    check("strict start/write alternation, never two starts in a row", order == expected,
          order)
    check("all three jobs completed", len(op.done) == 3, len(op.done))


def sequencing_test_can_fail(bake_mod):
    """Per the plan: prove the ordering assertion above is not vacuous. Monkeypatches
    the REAL _advance() on a second operator instance to start the next job's position
    bake BEFORE writing the current one -- the exact regression the ordering check
    exists to catch -- and drives it through the same modal() loop the real test uses,
    confirming the resulting order fails the clean-alternation assertion."""
    print("\n(VERIFYING THE ABOVE TEST CAN ACTUALLY FAIL)")
    objs = [FakeObj(n) for n in ("A", "B", "C")]
    op = bake_mod.AUTOSTROKE_OT_bake()
    op.jobs = [bake_mod.Job(o) for o in objs]
    op.i = 0
    op.stages = bake_mod.Stages()
    op.done, op.failed = [], []
    op._batch_t0 = time.time()
    op._timer = object()
    op.workdir, op.mask, op.brush_set, op.cfg = "/tmp", [None], "standard", None
    op.report = lambda level, msg: None

    order = []
    _wire_stub_job_methods(op, order)

    def broken_modal(context, event):
        """The real invariant lives in modal()'s own body -- it calls _write on the
        job that just finished, THEN _advance (which starts the next one). Overriding
        _advance alone cannot break this, because modal() would still call the real
        _write before ever reaching it. This override starts the NEXT job's bake
        BEFORE writing the CURRENT one's output -- the actual shape of the hazard
        _write()/_advance() exist to prevent (two Cycles bakes overlapping)."""
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        job = op.jobs[op.i]
        try:
            next(job.gen)
        except StopIteration as stop:
            job.result = stop.value
            nxt = op.i + 1
            if nxt < len(op.jobs):
                op._start_job(context, op.jobs[nxt])      # BUG: before this job is written
            op._write(context, job)
            op.done.append(job)
            op.i = nxt
            return {'RUNNING_MODAL'} if op.i < len(op.jobs) else {'FINISHED'}
        return {'RUNNING_MODAL'}

    op.modal = broken_modal
    op._start_job(FakeContext(), op.jobs[0])
    order.clear()
    _drive_to_completion(op, FakeContext())

    expected = ["write(A)", "start(B)", "write(B)", "start(C)", "write(C)"]
    check("the broken ordering produces something OTHER than the clean pattern",
          order != expected, order)
    check("...specifically, a start lands before the write it should follow",
          order[:2] == ["start(B)", "write(A)"], order)


def partial_failure_tests(bake_mod):
    print("\nONE FAILED OBJECT SKIPS AND CONTINUES, DOES NOT ABORT THE BATCH")
    objs = [FakeObj(n) for n in ("A", "B", "C")]
    op = bake_mod.AUTOSTROKE_OT_bake()
    op.jobs = [bake_mod.Job(o) for o in objs]
    op.i = 0
    op.stages = bake_mod.Stages()
    op.done, op.failed = [], []
    op._batch_t0 = time.time()
    op._timer = object()
    op.workdir, op.mask, op.brush_set, op.cfg = "/tmp", [None], "standard", None
    op.report = lambda level, msg: None

    order = []
    _wire_stub_job_methods(op, order, fail_on="B")
    op._start_job(FakeContext(), op.jobs[0])
    ctx = FakeContext()
    result = _drive_to_completion(op, ctx)

    check("batch still finishes ({'FINISHED'}) with a partial failure",
          result == {'FINISHED'}, result)
    check("A and C completed", {j.obj.name for j in op.done} == {"A", "C"},
          [j.obj.name for j in op.done])
    check("B is recorded as failed, with a reason",
          op.failed == [("B", "stub failure")], op.failed)
    check("B's start attempt is in the order, its write is not",
          "start(B)" in order and "write(B)" not in order, order)

    print("\nEVERY OBJECT FAILING -> {'CANCELLED'}, NOT {'FINISHED'}")
    objs2 = [FakeObj(n) for n in ("X", "Y")]
    op2 = bake_mod.AUTOSTROKE_OT_bake()
    op2.jobs = [bake_mod.Job(o) for o in objs2]
    op2.i = 0
    op2.stages = bake_mod.Stages()
    op2.done, op2.failed = [], []
    op2._batch_t0 = time.time()
    op2._timer = object()
    op2.workdir, op2.mask, op2.brush_set, op2.cfg = "/tmp", [None], "standard", None

    def always_fail(context, job):
        job.error = "nope"
        return False
    op2._start_job = always_fail
    op2.report = lambda level, msg: None
    started = op2._start_job(FakeContext(), op2.jobs[0])
    result2 = op2._advance(op2.jobs and FakeContext()) if not started else None
    if started is False:
        # invoke()'s own recovery path: first job failed immediately
        op2.failed.append((op2.jobs[0].obj.name, op2.jobs[0].error))
        result2 = op2._advance(FakeContext())
    check("all-failed batch returns {'CANCELLED'}", result2 == {'CANCELLED'}, result2)
    check("nothing recorded as done", op2.done == [])


def per_job_timing(bake_mod):
    print("\nEACH JOB'S secs IS ITS OWN ELAPSED TIME, NOT THE CUMULATIVE BATCH TIME")
    # Exercises the REAL _write(), not a stub -- this is exactly the bug class the
    # design doc calls out: a naive implementation sharing one clock across jobs would
    # make every object after the first look as slow as the whole batch so far.
    op = bake_mod.AUTOSTROKE_OT_bake()
    op.stages = bake_mod.Stages()
    op.workdir = "/tmp"
    op.mask = [np.ones((4, 4), np.float32)]
    op.brush_set = "standard"
    from autostroke.core.config import Config
    op.cfg = Config(dilate_px=1)
    op.report = lambda level, msg: None

    import autostroke.bridge.images as bi
    import autostroke.ops.material as material_ops
    orig_write_map, orig_build = bi.write_map, material_ops.build
    bi.write_map = lambda arr, name, path: None
    material_ops.build = lambda obj, **kw: None
    try:
        H = W = 4
        valid = np.ones(H * W, bool)
        job1 = bake_mod.Job(FakeObj("Slow"))
        job1.t0 = time.time() - 5.0          # this job "took" 5s
        job1.H, job1.W, job1.valid = H, W, valid
        job1.uv_self = np.zeros((H * W, 2), np.float32)
        job1.surf_nrm = np.tile([0, 0, 1.0], (H, W, 1)).astype(np.float32)
        job1.result = (np.zeros((H * W, 2), np.float32), np.zeros((H * W, 3), np.float32),
                      np.ones(H * W, bool), np.full(H * W, 0.5, np.float32))
        job1.stem = "slow"
        job1.sstats = {"strokes": 5, "radius_med": 0.1, "at_min": 0.0, "at_max": 0.0}

        ctx = FakeContext()
        op._write(ctx, job1)
        check("job with t0=now-5s reports ~5s, not batch time", 4.5 < job1.secs < 5.5,
              job1.secs)

        # a second job whose OWN elapsed time is much shorter, starting from a batch
        # that has already been "running" far longer than either job individually --
        # if secs were computed from a shared/batch clock this would read as ~8s+, not ~1s
        op._batch_t0 = time.time() - 20.0
        job2 = bake_mod.Job(FakeObj("Fast"))
        job2.t0 = time.time() - 1.0
        job2.H, job2.W, job2.valid = H, W, valid
        job2.uv_self = np.zeros((H * W, 2), np.float32)
        job2.surf_nrm = np.tile([0, 0, 1.0], (H, W, 1)).astype(np.float32)
        job2.result = (np.zeros((H * W, 2), np.float32), np.zeros((H * W, 3), np.float32),
                      np.ones(H * W, bool), np.full(H * W, 0.5, np.float32))
        job2.stem = "fast"
        job2.sstats = {"strokes": 5, "radius_med": 0.1, "at_min": 0.0, "at_max": 0.0}
        op._write(ctx, job2)
        check("second job with t0=now-1s reports ~1s, unaffected by batch age or job1",
              0.5 < job2.secs < 1.5, job2.secs)

        print("\nEST_SCALE LEARNS ONLY FROM CPU JOBS (it models numpy time)")
        ctx.scene.autostroke.est_scale = 1.0
        job2.device, job2.t0 = 'GPU', time.time() - 0.05
        op._write(ctx, job2)
        check("a GPU job leaves the CPU cost calibration alone",
              ctx.scene.autostroke.est_scale == 1.0, ctx.scene.autostroke.est_scale)
        job2.device, job2.t0 = 'CPU', time.time() - 1.0
        op._write(ctx, job2)
        check("a CPU job still calibrates it", ctx.scene.autostroke.est_scale != 1.0,
              ctx.scene.autostroke.est_scale)
    finally:
        bi.write_map = orig_write_map
        material_ops.build = orig_build


def report_batch_tests(bake_mod):
    print("\nSINGLE-OBJECT REPORT STRING IS BYTE-IDENTICAL TO BEFORE BATCHING")
    op = bake_mod.AUTOSTROKE_OT_bake()
    op.stages = bake_mod.Stages()
    op.brush_set = "standard"
    op.mask = [None, None]
    op.report = lambda level, msg: None
    op._batch_t0 = time.time() - 12.0

    job = bake_mod.Job(FakeObj("Solo"))
    job.secs = 12.0
    job.coverage = 94.2
    job.pos_note = "position: baked"
    job.sstats = {"strokes": 4821, "radius_med": 0.0123, "at_min": 0.1, "at_max": 0.0}
    op.jobs = [job]
    op.done = [job]
    op.failed = []

    ctx = FakeContext()
    op._report_batch(ctx)
    expected = (
        "94.2% coverage · 4,821 strokes · 12s|"
        "stroke radius 0.0123 u · faces at min 10%, at max 0%|"
        "brushes: standard (2) · position: baked|untimed 12.0s")
    check("matches the exact single-object format", ctx.scene.autostroke.last_report == expected,
          ctx.scene.autostroke.last_report)

    print("\nMULTI-OBJECT REPORT SUMMARISES, NAMES FAILURES")
    op2 = bake_mod.AUTOSTROKE_OT_bake()
    op2.stages = bake_mod.Stages()
    op2.brush_set = "standard"
    op2.mask = [None]
    op2.report = lambda level, msg: None
    op2._batch_t0 = time.time() - 59.0

    j1 = bake_mod.Job(FakeObj("Suzanne"))
    j1.secs, j1.coverage = 38.0, 94.2
    j1.sstats = {"strokes": 4821, "radius_med": 0.01, "at_min": 0.0, "at_max": 0.0}
    j2 = bake_mod.Job(FakeObj("Cube"))
    j2.secs, j2.coverage = 21.0, 88.1
    j2.sstats = {"strokes": 2004, "radius_med": 0.01, "at_min": 0.0, "at_max": 0.0}
    op2.jobs = [j1, j2, bake_mod.Job(FakeObj("Plane"))]
    op2.done = [j1, j2]
    op2.failed = [("Plane", "no UV map")]

    ctx2 = FakeContext()
    op2._report_batch(ctx2)
    lines = ctx2.scene.autostroke.last_report.split("|")
    check("header line names the counts", lines[0] == "3 objects · 2 baked, 1 failed · 59s total",
          lines[0])
    check("one line per successful object",
          any(l.startswith("Suzanne:") for l in lines) and any(l.startswith("Cube:") for l in lines),
          lines)
    check("the failure is named with its reason", any(l == "failed: Plane — no UV map"
                                                       for l in lines), lines)


def drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def device_tests(bake_mod):
    import contextlib
    import io
    print("\nDEVICE CHOICE")
    check("GPU requested, windowed -> GPU", bake_mod.pick_device('GPU', False) == ('GPU', ""))
    check("CPU requested -> CPU", bake_mod.pick_device('CPU', False)[0] == 'CPU')
    d, why = bake_mod.pick_device('GPU', True)
    check("GPU requested but headless -> CPU, and says why", d == 'CPU' and "headless" in why,
          why)

    print("\nA FAILING GPU FINISHES THE SAME JOB ON THE CPU -- AND THE REST OF THE BATCH")
    err = bake_mod.gpu_resolve.GPUResolveError

    def gpu_ok():
        yield 0.5
        return "gpu-result"

    def cpu_ok():
        yield 0.5
        return "cpu-result"

    def gpu_bad():
        yield 0.3                           # fails AFTER making progress, the harder case
        raise err("12 texels were never written by the GPU")

    op = bake_mod.AUTOSTROKE_OT_bake()
    op.device, op.device_note = 'GPU', ""
    warnings = []
    op.report = lambda level, msg: warnings.append(msg) if level == {'WARNING'} else None

    def job(name, note):
        j = bake_mod.Job(FakeObj(name))
        j.device, j.pos_note = 'GPU', note
        return j

    quiet = io.StringIO()                   # the fallback prints its traceback to console
    a = job("A", "position: baked · GPU")
    with contextlib.redirect_stdout(quiet):
        res = drain(op._gpu_or_cpu(a, gpu_ok, cpu_ok))
    check("GPU success: GPU result, job stays GPU",
          res == "gpu-result" and a.device == 'GPU' and op.device == 'GPU')

    b = job("B", "position: cached · GPU")
    with contextlib.redirect_stdout(quiet):
        res = drain(op._gpu_or_cpu(b, gpu_bad, cpu_ok))
    check("GPU failure mid-job: the SAME job completes with the CPU result",
          res == "cpu-result", res)
    check("...the job is recorded as CPU", b.device == 'CPU')
    check("...its report names the fallback and why",
          "CPU (GPU failed: 12 texels were never written" in b.pos_note, b.pos_note)
    check("...the rest of the batch switches to CPU", op.device == 'CPU')
    check("...and the artist is warned once", len(warnings) == 1, warnings)

    c = job("C", "position: baked · GPU")
    with contextlib.redirect_stdout(quiet):
        drain(op._gpu_or_cpu(c, gpu_bad, cpu_ok))
    check("a second failure in the same batch does not warn again", len(warnings) == 1)

    print("\nFALLBACK INSIDE THE REAL MODAL LOOP: SEQUENCING UNCHANGED")
    objs = [FakeObj(n) for n in ("A", "B", "C")]
    op2 = bake_mod.AUTOSTROKE_OT_bake()
    op2.jobs = [bake_mod.Job(o) for o in objs]
    op2.i = 0
    op2.stages = bake_mod.Stages()
    op2.done, op2.failed = [], []
    op2._batch_t0 = time.time()
    op2._timer = object()
    op2.workdir, op2.mask, op2.brush_set, op2.cfg = "/tmp", [None], "standard", None
    op2.device, op2.device_note = 'GPU', ""
    op2.report = lambda level, msg: None
    order, used = [], []

    def start(context, j):
        order.append("start(%s)" % j.obj.name)
        j.t0 = time.time()
        j.pos_note = "position: baked"
        if op2.device == 'GPU':
            gpu = gpu_bad if j.obj.name == "B" else gpu_ok
            j.device = 'GPU'
            j.pos_note += " · GPU"
            j.gen = op2._gpu_or_cpu(j, gpu, cpu_ok)
        else:
            j.device = 'CPU'
            j.gen = cpu_ok()
        return True

    def write(context, j):
        order.append("write(%s)" % j.obj.name)
        used.append((j.obj.name, j.device))
        j.secs, j.coverage = 0.0, 100.0
        j.sstats = {"strokes": 1, "radius_med": 0.1, "at_min": 0.0, "at_max": 0.0}

    op2._start_job, op2._write = start, write
    op2._start_job(FakeContext(), op2.jobs[0])
    with contextlib.redirect_stdout(quiet):
        result = _drive_to_completion(op2, FakeContext())
    check("batch finishes", result == {'FINISHED'}, result)
    check("strict start/write alternation still holds",
          order == ["start(A)", "write(A)", "start(B)", "write(B)", "start(C)", "write(C)"],
          order)
    check("A on GPU, B fell back mid-job, C started on CPU",
          used == [("A", 'GPU'), ("B", 'CPU'), ("C", 'CPU')], used)


if __name__ == "__main__":
    sys.exit(main())
