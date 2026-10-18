# -*- coding: utf-8 -*-
"""Stage-2B candidate: Adaptive Reference-Vector Guided NSGAIII-MOPSO V2.

This development candidate preserves the binary service-placement representation,
MOPSO position/velocity equations, crossover/mutation operators, and three QoS
objectives, while correcting five mechanisms exposed by the Stage-2A diagnostic:

1. standard weak Pareto dominance for internal sorting/archive/pBest logic;
2. ideal-nadir normalization before reference-vector association/adaptation;
3. sparse-reference-region archive leader selection instead of random leaders;
4. Pareto-aware pBest update with a scalar convergence tie-break for mutually
   nondominated alternatives;
5. controlled reference-vector adaptation plus parameterized occupancy scoring.

The historical Original and Adaptive-V1 implementations remain untouched.
"""
from __future__ import division
import copy
import numpy as np
import POPULATION as pop

from HYBRID_MOPSO_NSGA3_VECTOR import HYBRID_MOPSO_NSGA3_VECTOR
from core.q1_common import (
    obj_vector, dominates_weak, nondominated_sort, nondominated_indices,
    normalize_objectives,
)


class ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR_V2(HYBRID_MOPSO_NSGA3_VECTOR):
    """Corrected development candidate for Q1 Stage-2B."""

    name = "adaptive_nsgaiii_mopso_v2"

    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_)
        self.initialReferenceVectors = copy.deepcopy(self.referenceVectors)
        self.adaptation_strength = 0.35
        self.adaptation_interval = 2
        self.adaptation_warmup = 0.10
        self.target_anchor_weight = 0.50
        self.occupancy_lambda = 0.50
        self._v2_step = 0
        # Historical constructor builds pBest before one redundant evaluation.
        # Ensure pBest fitness is fully synchronized with the initialized particles.
        self.pbest = copy.deepcopy(self.particles)
        self._rebuild_repository_from_current()

    # ------------------------------------------------------------------
    # Standard Pareto semantics
    # ------------------------------------------------------------------
    def fastNonDominatedSort(self, popT):
        popT.fronts = nondominated_sort(popT.fitness)
        # Preserve fields used by the legacy population container.
        n = len(popT.population)
        popT.dominationCount = [0] * n
        popT.dominatedBy = [set() for _ in range(n)]
        popT.dominatesTo = [set() for _ in range(n)]

    @staticmethod
    def _same_objectives(a, b, tol=1e-12):
        return bool(np.allclose(obj_vector(a), obj_vector(b), atol=tol, rtol=0.0))

    # ------------------------------------------------------------------
    # Ideal-nadir reference geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized_matrix(popT):
        F = np.vstack([obj_vector(f) for f in popT.fitness])
        Fn, ideal, nadir = normalize_objectives(F)
        return Fn, ideal, nadir

    @staticmethod
    def _unit_rows(Fn):
        D = np.asarray(Fn, dtype=float).copy()
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        return D / norms

    def _associate_matrix(self, popT):
        Fn, ideal, nadir = self._normalized_matrix(popT)
        dirs = self._unit_rows(Fn)
        assoc = np.zeros(len(dirs), dtype=int)
        dist = np.zeros(len(dirs), dtype=float)
        for i, d in enumerate(dirs):
            perps = []
            for v in self.referenceVectors:
                proj = np.dot(d, v) * v
                perps.append(np.linalg.norm(d - proj))
            assoc[i] = int(np.argmin(perps))
            dist[i] = float(perps[assoc[i]])
        return assoc, dist, Fn, ideal, nadir

    def assignReferenceVectors(self, popT):
        assoc, dist, Fn, ideal, nadir = self._associate_matrix(popT)
        occ = np.bincount(assoc, minlength=len(self.referenceVectors)).astype(int)
        self.vector_occupancy = occ
        for i, fit in enumerate(popT.fitness):
            k = int(assoc[i])
            fit["referenceVectorIndex"] = k
            fit["referenceVector"] = self.referenceVectors[k].tolist()
            fit["referenceDistance"] = float(dist[i])
            fit["normalizedObjectivesV2"] = Fn[i].tolist()
        return assoc, dist, occ

    # ------------------------------------------------------------------
    # Controlled Pareto-informed vector adaptation
    # ------------------------------------------------------------------
    def adaptReferenceVectors(self, popT):
        fronts = nondominated_sort(popT.fitness)
        if not fronts or not fronts[0]:
            return
        Fn, _, _ = self._normalized_matrix(popT)
        dirs = self._unit_rows(Fn[fronts[0]])
        alpha = float(self.adaptation_strength)
        beta = float(self.target_anchor_weight)
        updated_vectors = []
        for j, ref in enumerate(self.referenceVectors):
            perps = []
            for d in dirs:
                projection = np.dot(d, ref) * ref
                perps.append(np.linalg.norm(d - projection))
            target = dirs[int(np.argmin(perps))]
            anchor = self.initialReferenceVectors[j]
            blended = (1.0 - beta) * target + beta * anchor
            n = np.linalg.norm(blended)
            if n > 1e-12:
                blended /= n
            newv = (1.0 - alpha) * ref + alpha * blended
            n = np.linalg.norm(newv)
            updated_vectors.append(newv / n if n > 1e-12 else ref)
        self.referenceVectors = np.asarray(updated_vectors, dtype=float)

    # ------------------------------------------------------------------
    # Repository: weak Pareto + objective dedup + reference-region diversity
    # ------------------------------------------------------------------
    def _select_archive_indices(self, combined, repo_size):
        nd = nondominated_indices(combined.fitness)
        # Deduplicate identical objective vectors deterministically.
        unique = []
        seen = set()
        for i in nd:
            key = tuple(np.round(obj_vector(combined.fitness[i]), 12))
            if key not in seen:
                seen.add(key); unique.append(i)
        if len(unique) <= repo_size:
            return unique

        # Reference-region round-robin, preferring closest solution per direction.
        sub = pop.POPULATION(0)
        sub.population = [combined.population[i] for i in unique]
        sub.fitness = [copy.deepcopy(combined.fitness[i]) for i in unique]
        assoc, dist, _, _, _ = self._associate_matrix(sub)
        buckets = {}
        for local_i, k in enumerate(assoc):
            buckets.setdefault(int(k), []).append((float(dist[local_i]), local_i))
        for k in buckets:
            buckets[k].sort(key=lambda x: x[0])
        chosen_local = []
        active = sorted(buckets.keys())
        while len(chosen_local) < repo_size and active:
            next_active = []
            for k in active:
                if buckets[k] and len(chosen_local) < repo_size:
                    _, li = buckets[k].pop(0)
                    chosen_local.append(li)
                if buckets[k]:
                    next_active.append(k)
            active = next_active
        return [unique[li] for li in chosen_local[:repo_size]]

    def _rebuild_repository_from_current(self):
        empty = pop.POPULATION(0)
        self.gbest = empty.populationUnion(empty, self.particles)
        idx = self._select_archive_indices(self.gbest, max(1, self.populationSize // 2))
        src = self.gbest
        repo = pop.POPULATION(0)
        repo.population = [copy.deepcopy(src.population[i]) for i in idx]
        repo.fitness = [copy.deepcopy(src.fitness[i]) for i in idx]
        self.gbest = repo

    def updateRepository(self):
        combined = self.gbest.populationUnion(self.gbest, self.particles)
        # Fitness already exists for both repository and particles; avoid redundant calls.
        idx = self._select_archive_indices(combined, max(1, self.populationSize // 2))
        repo = pop.POPULATION(0)
        repo.population = [copy.deepcopy(combined.population[i]) for i in idx]
        repo.fitness = [copy.deepcopy(combined.fitness[i]) for i in idx]
        self.gbest = repo

    # ------------------------------------------------------------------
    # Sparse-region leader selection
    # ------------------------------------------------------------------
    def _leader_groups(self):
        if not self.gbest.population:
            return {}
        assoc, dist, _, _, _ = self._associate_matrix(self.gbest)
        groups = {}
        for i, k in enumerate(assoc):
            groups.setdefault(int(k), []).append((float(dist[i]), i))
        for k in groups:
            groups[k].sort(key=lambda x: x[0])
        return groups

    def updateVelocities(self):
        groups = self._leader_groups()
        occupied = list(groups.keys())
        # Sparse by archive membership count; among tied regions sample reproducibly.
        min_occ = min([len(groups[k]) for k in occupied]) if occupied else 0
        sparse_refs = [k for k in occupied if len(groups[k]) == min_occ]
        for i in range(self.populationSize):
            if sparse_refs:
                k = sparse_refs[self.corega.rndEVOL.randint(0, len(sparse_refs)-1)]
                # choose among the closest half of solutions in that sparse region
                members = groups[k]
                topn = max(1, (len(members)+1)//2)
                leader_idx = members[self.corega.rndEVOL.randint(0, topn-1)][1]
                leader = self.gbest.population[leader_idx]
            else:
                leader = self.particles.population[i]
            for service_idx in range(len(self.particles.population[i])):
                for fog_idx in range(len(self.particles.population[i][service_idx])):
                    r1 = self.corega.rndEVOL.random()
                    r2 = self.corega.rndEVOL.random()
                    cur = self.particles.population[i][service_idx][fog_idx]
                    pb = self.pbest.population[i][service_idx][fog_idx]
                    gb = leader[service_idx][fog_idx]
                    cognitive = self.c1 * r1 * (pb - cur)
                    social = self.c2 * r2 * (gb - cur)
                    self.velocities[i][(service_idx, fog_idx)] = (
                        self.w * self.velocities[i].get((service_idx, fog_idx), 0.0)
                        + cognitive + social
                    )

    # ------------------------------------------------------------------
    # Pareto-aware pBest update
    # ------------------------------------------------------------------
    def updatePbestV2(self):
        for i in range(self.populationSize):
            cur = self.particles.fitness[i]
            old = self.pbest.fitness[i]
            replace = False
            if dominates_weak(cur, old):
                replace = True
            elif dominates_weak(old, cur):
                replace = False
            else:
                # Mutually nondominated: convergence-oriented tie-break using the
                # inherited equal-weight normalized scalar score; exact ties random.
                ct = float(cur.get("total", np.inf))
                ot = float(old.get("total", np.inf))
                if ct < ot - 1e-12:
                    replace = True
                elif abs(ct - ot) <= 1e-12:
                    replace = self.corega.rndEVOL.random() < 0.5
            if replace:
                self.pbest.population[i] = copy.deepcopy(self.particles.population[i])
                self.pbest.fitness[i] = copy.deepcopy(cur)

    # ------------------------------------------------------------------
    # Occupancy-aware environmental selection
    # ------------------------------------------------------------------
    def _environmental_select(self, combined, n_select):
        fronts = nondominated_sort(combined.fitness)
        selected = []
        split = None
        for fr in fronts:
            if len(selected) + len(fr) <= n_select:
                selected.extend(fr)
            else:
                split = list(fr)
                break
        if len(selected) >= n_select or not split:
            return selected[:n_select]

        assoc, dist, _, _, _ = self._associate_matrix(combined)
        rho = np.zeros(len(self.referenceVectors), dtype=int)
        for i in selected:
            rho[int(assoc[i])] += 1
        candidates = list(split)
        lam = float(self.occupancy_lambda)
        while len(selected) < n_select and candidates:
            best = None
            best_score = -np.inf
            for i in candidates:
                k = int(assoc[i])
                closeness = 1.0 / (1.0 + float(dist[i]))
                sparsity = 1.0 / (1.0 + float(rho[k]))
                score = lam * closeness + (1.0 - lam) * sparsity
                if score > best_score + 1e-15:
                    best_score = score; best = i
                elif abs(score - best_score) <= 1e-15 and best is not None:
                    # deterministic convergence tie-break
                    if float(combined.fitness[i].get("total", np.inf)) < float(combined.fitness[best].get("total", np.inf)):
                        best = i
            selected.append(best)
            rho[int(assoc[best])] += 1
            candidates.remove(best)
        return selected[:n_select]

    # ------------------------------------------------------------------
    # One Stage-2B evaluation-budget step
    # ------------------------------------------------------------------
    def step(self, progress=0.0):
        self._v2_step += 1
        self.updateVelocities()
        self.updatePositions()          # evaluates moved particles
        self.updatePbestV2()
        self.updateRepository()         # no redundant reevaluation

        orderedFathers = self.corega.orderPopulation(self.particles)
        offspring = self.evolveToOffspring(orderedFathers)
        self.corega.calculatePopulationFitnessObjectives(offspring)
        combined = self.particles.populationUnion(self.particles, offspring)

        # Controlled adaptation: warm-up first, then every I complete V2 steps.
        if (float(progress) >= self.adaptation_warmup and
                self._v2_step % max(1, int(self.adaptation_interval)) == 0):
            self.adaptReferenceVectors(combined)

        self.assignReferenceVectors(combined)
        idx = self._environmental_select(combined, self.populationSize)
        final = pop.POPULATION(self.populationSize)
        for j, src_i in enumerate(idx):
            final.population[j] = copy.deepcopy(combined.population[src_i])
            final.fitness[j] = copy.deepcopy(combined.fitness[src_i])
        self.particles = final
        self.corega.populationPt = self.particles
        # Fitness was copied from evaluated parents/offspring; do not reevaluate.
