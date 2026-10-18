# -*- coding: utf-8 -*-
"""Common Q1 experiment utilities.

All optimizer-internal objective values are treated as minimization objectives:
    spread, makespan (communication-latency proxy), resources (unused-resource fraction)
For reporting only, Resource Utilization = 1 - resources.
"""
from __future__ import division
import copy
import math
import numpy as np

OBJ_KEYS = ("spread", "makespan", "resources")


def finite_fitness(f):
    return all(np.isfinite(float(f.get(k, np.inf))) for k in OBJ_KEYS)


def obj_vector(f):
    return np.asarray([float(f[k]) for k in OBJ_KEYS], dtype=float)


def dominates_weak(a, b):
    """Standard Pareto dominance for minimization: <= all and < at least one."""
    av = obj_vector(a); bv = obj_vector(b)
    return bool(np.all(av <= bv) and np.any(av < bv))


def nondominated_sort(fitness):
    """Return list of fronts, each a list of indices, using standard weak Pareto dominance."""
    n = len(fitness)
    S = [set() for _ in range(n)]
    n_dom = [0] * n
    fronts = [[]]
    for p in range(n):
        if not finite_fitness(fitness[p]):
            continue
        for q in range(n):
            if p == q or not finite_fitness(fitness[q]):
                continue
            if dominates_weak(fitness[p], fitness[q]):
                S[p].add(q)
            elif dominates_weak(fitness[q], fitness[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    seen = set(fronts[0])
    while i < len(fronts) and fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0 and q not in seen:
                    nxt.append(q); seen.add(q)
        if nxt:
            fronts.append(nxt)
        i += 1
    # Invalid/infinite points are placed last rather than lost.
    invalid = [i for i, f in enumerate(fitness) if i not in seen]
    if invalid:
        fronts.append(invalid)
    return fronts


def nondominated_indices(fitness):
    fronts = nondominated_sort(fitness)
    return fronts[0] if fronts else []


def crowding_distance(fitness, front):
    """NSGA-II crowding distance in raw objective space with per-front range normalization."""
    d = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front: d[i] = float("inf")
        return d
    for key in OBJ_KEYS:
        order = sorted(front, key=lambda i: float(fitness[i].get(key, np.inf)))
        lo = float(fitness[order[0]].get(key, np.inf)); hi = float(fitness[order[-1]].get(key, np.inf))
        d[order[0]] = d[order[-1]] = float("inf")
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        for j in range(1, len(order)-1):
            if np.isinf(d[order[j]]):
                continue
            prevv = float(fitness[order[j-1]][key]); nextv = float(fitness[order[j+1]][key])
            d[order[j]] += (nextv - prevv) / (hi - lo)
    return d


def rank_and_crowding(fitness):
    fronts = nondominated_sort(fitness)
    rank = {}
    crowd = {}
    for r, fr in enumerate(fronts):
        cd = crowding_distance(fitness, fr)
        for i in fr:
            rank[i] = r; crowd[i] = cd.get(i, 0.0)
    return fronts, rank, crowd


def reporting_row(f):
    """Convert internal objective dictionary to manuscript-facing orientation."""
    unused = float(f["resources"])
    return {
        "service_spread": float(f["spread"]),
        "communication_latency_proxy": float(f["makespan"]),
        "unused_resource_fraction": unused,
        "resource_utilization": 1.0 - unused,
    }


def das_dennis_reference_vectors(m=3, p=19):
    """Das-Dennis simplex-lattice reference directions, normalized to unit L2 length."""
    out = []
    def rec(prefix, left, dim):
        if dim == 1:
            vals = prefix + [left]
            v = np.asarray(vals, dtype=float) / float(p)
            n = np.linalg.norm(v)
            if n > 0: out.append(v / n)
            return
        for i in range(left + 1):
            rec(prefix + [i], left - i, dim - 1)
    rec([], p, m)
    return np.asarray(out, dtype=float)


def normalize_objectives(F):
    """Stable ideal/range normalization used for association and diagnostics."""
    F = np.asarray(F, dtype=float)
    ideal = np.nanmin(F, axis=0)
    nadir = np.nanmax(F, axis=0)
    rng = nadir - ideal
    rng[rng <= 1e-12] = 1.0
    return (F - ideal) / rng, ideal, nadir


def associate_to_vectors(F, V):
    Fn, _, _ = normalize_objectives(F)
    dirs = Fn.copy()
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    dirs /= norms
    assoc = np.zeros(len(dirs), dtype=int)
    dist = np.zeros(len(dirs), dtype=float)
    for i, d in enumerate(dirs):
        perps = []
        for v in V:
            proj = np.dot(d, v) * v
            perps.append(np.linalg.norm(d - proj))
        assoc[i] = int(np.argmin(perps)); dist[i] = float(perps[assoc[i]])
    return assoc, dist


def environmental_select_nsga3(fitness, n_select, ref_vectors, rng):
    """Reference-direction environmental selection suitable for the standard NSGA-III comparator.

    Whole nondominated fronts are accepted. The splitting front is selected by reference-direction
    niching using nearest perpendicular distance and current niche occupancy.
    """
    fronts = nondominated_sort(fitness)
    chosen = []
    split = None
    for fr in fronts:
        if len(chosen) + len(fr) <= n_select:
            chosen.extend(fr)
        else:
            split = list(fr); break
    if len(chosen) >= n_select or not split:
        return chosen[:n_select]

    valid_all = [i for i, f in enumerate(fitness) if finite_fitness(f)]
    F = np.vstack([obj_vector(fitness[i]) for i in valid_all])
    assoc_all, dist_all = associate_to_vectors(F, ref_vectors)
    pos = {idx: j for j, idx in enumerate(valid_all)}
    rho = np.zeros(len(ref_vectors), dtype=int)
    for i in chosen:
        if i in pos: rho[assoc_all[pos[i]]] += 1

    candidates = list(split)
    while len(chosen) < n_select and candidates:
        by_ref = {}
        for i in candidates:
            if i not in pos: continue
            k = int(assoc_all[pos[i]])
            by_ref.setdefault(k, []).append(i)
        if not by_ref:
            chosen.extend(candidates[:n_select-len(chosen)]); break
        min_rho = min(rho[k] for k in by_ref)
        refs = [k for k in by_ref if rho[k] == min_rho]
        k = refs[rng.randint(0, len(refs)-1)]
        cands = by_ref[k]
        if rho[k] == 0:
            pick = min(cands, key=lambda i: dist_all[pos[i]])
        else:
            pick = cands[rng.randint(0, len(cands)-1)]
        chosen.append(pick); candidates.remove(pick); rho[k] += 1
    return chosen[:n_select]


def save_front_csv(path, fitness, indices):
    import csv, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "spread", "communication_latency_proxy", "unused_resource_fraction", "resource_utilization", "wspr", "wmak", "wres", "total"])
        for i in indices:
            x = fitness[i]
            w.writerow([i, x.get("spread"), x.get("makespan"), x.get("resources"),
                        1.0 - float(x.get("resources")), x.get("wspr"), x.get("wmak"), x.get("wres"), x.get("total")])
