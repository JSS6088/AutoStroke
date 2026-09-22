"""Analytic validation of the curvature direction field.

Runs in plain terminal python -- this is exactly the tooling the bpy-free `core/`
split exists to preserve.  python3 autostroke/tests/test_geometry.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all="ignore")
from core.config import Config
from core import geometry as g

TOL = 1.0   # degrees


def _err(P, N, truth, cfg):
    nb = g.knn_indices(P, cfg.curv_k, cfg.curv_max_bytes)
    w = np.ones((len(P), min(cfg.curv_k, len(P) - 1)))
    D, bad = g.principal_min_direction(P, N, nb, w)
    a = np.degrees(np.arccos(np.abs((D * truth).sum(1)).clip(0, 1)))
    return np.median(a), bad


def main():
    cfg = Config()
    rng = np.random.default_rng(0)
    fails = []

    n = 3000
    th = rng.uniform(0, 2*np.pi, n); z = rng.uniform(-2, 2, n)
    P = np.stack([np.cos(th), np.sin(th), z], 1)
    N = np.stack([np.cos(th), np.sin(th), np.zeros(n)], 1)
    e, bad = _err(P, N, np.tile([0, 0, 1.], (n, 1)), cfg)
    print("cylinder (truth = axis)         %6.2f deg   degenerate %d" % (e, bad))
    if e > TOL: fails.append("cylinder")

    th = rng.uniform(0, 2*np.pi, n); t = rng.uniform(0.5, 2.0, n); k = 0.5
    P = np.stack([t*k*np.cos(th), t*k*np.sin(th), t], 1)
    d = np.stack([k*np.cos(th), k*np.sin(th), np.ones(n)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    N = np.stack([np.cos(th), np.sin(th), -k*np.ones(n)], 1)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    e, bad = _err(P, N, d, cfg)
    print("cone (truth = slant line)       %6.2f deg   degenerate %d" % (e, bad))
    if e > TOL: fails.append("cone")

    m = 4000; R, r = 2.0, 0.6
    u = rng.uniform(0, 2*np.pi, m); v = rng.uniform(0, 2*np.pi, m)
    P = np.stack([(R+r*np.cos(v))*np.cos(u), (R+r*np.cos(v))*np.sin(u), r*np.sin(v)], 1)
    N = np.stack([np.cos(v)*np.cos(u), np.cos(v)*np.sin(u), np.sin(v)], 1)
    tru = np.stack([-np.sin(u), np.cos(u), np.zeros(m)], 1)
    sel = np.cos(v) > 0.35
    e, bad = _err(P[sel], N[sel], tru[sel], cfg)
    print("torus outer band (truth = u)    %6.2f deg   degenerate %d" % (e, bad))
    if e > TOL: fails.append("torus")

    # a flat plane and a cube: hard-surface cases where curvature is zero
    P = np.stack([rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000), np.zeros(2000)], 1)
    N = np.tile([0, 0, 1.], (2000, 1))
    T, st = g.compute_direction_field(P, N, cfg)
    nb = g.knn_indices(P, cfg.curv_k, cfg.curv_max_bytes)
    coh = np.degrees(np.arccos(np.abs((T[nb]*T[:, None, :]).sum(-1)).clip(0, 1))).mean()
    print("flat plane coherence            %6.2f deg   (0 = one coherent direction)" % coh)
    if coh > TOL: fails.append("flat plane")

    print("\n%s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
