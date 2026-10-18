# -*- coding: utf-8 -*-
"""Post-hoc multi-objective metrics for three minimization objectives."""
import numpy as np
from core.q1_common import nondominated_indices


def normalize_with_bounds(F, lo, hi):
    F = np.asarray(F, dtype=float); lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    r = hi - lo; r[r <= 1e-12] = 1.0
    return (F - lo) / r


def igd_plus(approx, reference):
    """IGD+ for minimization. Lower is better."""
    A = np.asarray(approx, dtype=float); R = np.asarray(reference, dtype=float)
    if len(A) == 0 or len(R) == 0:
        return float("nan")
    ds = []
    for r in R:
        # d+(r,a) = || max(a-r,0) ||
        d = np.sqrt(np.sum(np.maximum(A - r, 0.0) ** 2, axis=1))
        ds.append(np.min(d))
    return float(np.mean(ds))


def spacing(F):
    F = np.asarray(F, dtype=float)
    if len(F) < 2:
        return 0.0
    D = np.sqrt(np.sum((F[:, None, :] - F[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(D, np.inf)
    nearest = np.min(D, axis=1)
    if len(nearest) <= 1:
        return 0.0
    return float(np.std(nearest, ddof=1))


def hypervolume_3d_min(F, reference=(1.1, 1.1, 1.1)):
    """Exact dominated hypervolume in 3D minimization by x-slicing.

    Assumes points and reference are in the same normalized space. Points outside
    the reference box are clipped away.
    """
    F = np.asarray(F, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if F.size == 0:
        return 0.0
    mask = np.all(F < ref, axis=1)
    F = F[mask]
    if len(F) == 0:
        return 0.0
    # keep nondominated only
    fits = [{"spread":p[0], "makespan":p[1], "resources":p[2]} for p in F]
    nd = nondominated_indices(fits)
    F = F[nd]
    xs = sorted(set([float(x) for x in F[:,0]] + [float(ref[0])]))
    hv = 0.0
    for a, b in zip(xs[:-1], xs[1:]):
        active = F[F[:,0] <= a]
        if len(active) == 0:
            continue
        # 2D dominated area in yz plane
        yz = active[:,1:]
        order = np.argsort(yz[:,0])
        yz = yz[order]
        area = 0.0
        best_z = ref[2]
        prev_y = None
        # integrate staircase from low y toward ref_y
        # first obtain nondominated decreasing-z frontier
        frontier = []
        zbest = ref[2]
        for y,z in yz:
            if z < zbest:
                frontier.append((float(y), float(z)))
                zbest = z
        for i,(y,z) in enumerate(frontier):
            y_next = frontier[i+1][0] if i+1 < len(frontier) else ref[1]
            if y < ref[1] and z < ref[2] and y_next > y:
                area += (min(y_next, ref[1]) - y) * (ref[2] - z)
        hv += (b - a) * area
    return float(hv)
