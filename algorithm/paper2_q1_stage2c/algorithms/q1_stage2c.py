# -*- coding: utf-8 -*-
"""Stage-2C: Adaptive reference-vector guided selective MOPSO refinement.

Design principle: NSGA-III remains the primary engine. A small subset of solutions
from sparsely occupied reference regions is periodically refined by a binary MOPSO
move, and refinements compete with the NSGA-III population through the same
reference-direction environmental selection. This avoids the population-wide PSO
randomization that stalled Stage-2B while retaining adaptive-vector-guided swarm
exploitation.
"""
from __future__ import division
import copy
import numpy as np
import POPULATION as pop

from algorithms.q1_comparators import NSGA3_Q1
from core.q1_common import (
    obj_vector, nondominated_indices, das_dennis_reference_vectors,
    environmental_select_nsga3, normalize_objectives, associate_to_vectors,
)


class ADAPTIVE_SELECTIVE_MOPSO_NSGA3_STAGE2C(NSGA3_Q1):
    name = "adaptive_selective_mopso_stage2c"

    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_)
        self.initialReferenceVectors = self.referenceVectors.copy()
        self.adaptation_strength = 0.20
        self.anchor_weight = 0.50
        self.refinement_interval = 5
        self.refinement_fraction = 0.20
        self.adaptation_warmup = 0.10
        self.w = 0.70
        self.c1 = 1.50
        self.c2 = 1.50
        self.position_memory = 2.0
        self._step_count = 0
        self._archive = pop.POPULATION(0)
        self._update_archive(self.corega.populationPt)

    # ---------- adaptive reference vectors ----------
    @staticmethod
    def _unit_rows(Fn):
        D = np.asarray(Fn, dtype=float).copy()
        n = np.linalg.norm(D, axis=1, keepdims=True)
        n[n <= 1e-12] = 1.0
        return D / n

    def _adapt_vectors(self, C):
        nd = nondominated_indices(C.fitness)
        if not nd:
            return
        F = np.vstack([obj_vector(C.fitness[i]) for i in nd])
        Fn, _, _ = normalize_objectives(F)
        dirs = self._unit_rows(Fn)
        a = float(self.adaptation_strength)
        b = float(self.anchor_weight)
        out = []
        for j, ref in enumerate(self.referenceVectors):
            perps = [np.linalg.norm(d - np.dot(d, ref) * ref) for d in dirs]
            target = dirs[int(np.argmin(perps))]
            anchor = self.initialReferenceVectors[j]
            goal = (1.0 - b) * target + b * anchor
            ng = np.linalg.norm(goal)
            if ng > 1e-12:
                goal = goal / ng
            nv = (1.0 - a) * ref + a * goal
            nn = np.linalg.norm(nv)
            out.append(nv / nn if nn > 1e-12 else ref)
        self.referenceVectors = np.asarray(out, dtype=float)

    # ---------- external nondominated archive ----------
    def _update_archive(self, P):
        C = self._archive.populationUnion(self._archive, P)
        nd = nondominated_indices(C.fitness)
        seen, keep = set(), []
        for i in nd:
            key = tuple(np.round(obj_vector(C.fitness[i]), 12))
            if key not in seen:
                seen.add(key); keep.append(i)
        # Keep at most populationSize diverse archive members via NSGA-III niching.
        if len(keep) > self.populationSize:
            sub = pop.POPULATION(0)
            sub.population = [C.population[i] for i in keep]
            sub.fitness = [C.fitness[i] for i in keep]
            local = environmental_select_nsga3(
                sub.fitness, self.populationSize, self.referenceVectors, self.corega.rndEVOL
            )
            keep = [keep[i] for i in local]
        A = pop.POPULATION(0)
        A.population = [copy.deepcopy(C.population[i]) for i in keep]
        A.fitness = [copy.deepcopy(C.fitness[i]) for i in keep]
        self._archive = A

    # ---------- sparse-region candidate and leader choice ----------
    def _select_refinement_indices(self, P, n_refine):
        F = np.vstack([obj_vector(f) for f in P.fitness])
        assoc, dist = associate_to_vectors(F, self.referenceVectors)
        occ = np.bincount(assoc, minlength=len(self.referenceVectors))
        # Priority: sparse occupied region first, then closeness to its vector.
        order = sorted(range(len(P.population)), key=lambda i: (int(occ[assoc[i]]), float(dist[i])))
        return order[:max(1, min(n_refine, len(order)))]

    def _choose_leader(self, particle_fit):
        if not self._archive.population:
            return None
        AF = np.vstack([obj_vector(f) for f in self._archive.fitness])
        a_assoc, a_dist = associate_to_vectors(AF, self.referenceVectors)
        occ = np.bincount(a_assoc, minlength=len(self.referenceVectors))
        # Candidate's nearest direction; prefer archive leader there, otherwise a sparse region.
        PF = np.vstack([obj_vector(particle_fit)])
        p_assoc, _ = associate_to_vectors(PF, self.referenceVectors)
        k = int(p_assoc[0])
        members = [i for i in range(len(AF)) if int(a_assoc[i]) == k]
        if not members:
            occupied = np.where(occ > 0)[0]
            if len(occupied):
                min_occ = np.min(occ[occupied])
                sparse = [int(x) for x in occupied if occ[x] == min_occ]
                k = sparse[self.corega.rndEVOL.randint(0, len(sparse)-1)]
                members = [i for i in range(len(AF)) if int(a_assoc[i]) == k]
        if not members:
            return self._archive.population[self.corega.rndEVOL.randint(0, len(self._archive.population)-1)]
        bestd = min(float(a_dist[i]) for i in members)
        best = [i for i in members if abs(float(a_dist[i]) - bestd) <= 1e-12]
        return self._archive.population[best[self.corega.rndEVOL.randint(0, len(best)-1)]]

    # ---------- selective binary MOPSO refinement ----------
    def _refine(self, P):
        n_refine = max(1, int(round(self.populationSize * float(self.refinement_fraction))))
        selected = self._select_refinement_indices(P, n_refine)
        R = pop.POPULATION(0); R.population = []
        for i in selected:
            current = P.population[i]
            leader = self._choose_leader(P.fitness[i])
            if leader is None:
                continue
            child = copy.deepcopy(current)
            # pBest is the selected NSGA-III solution itself. Cognitive displacement is
            # therefore zero for this one-step local refinement; social guidance comes
            # from the adaptive-vector archive leader. A memory logit prevents v=0 from
            # turning every bit into a 0.5 coin flip.
            for s in range(len(child)):
                for f in range(len(child[s])):
                    cur = current[s][f]
                    gb = leader[s][f]
                    r2 = self.corega.rndEVOL.random()
                    velocity = self.c2 * r2 * (gb - cur)
                    base = self.position_memory * (1.0 if cur == 1 else -1.0)
                    z = np.clip(base + velocity, -20.0, 20.0)
                    prob1 = 1.0 / (1.0 + np.exp(-z))
                    child[s][f] = 1 if self.corega.rndEVOL.random() < prob1 else 0
                    if f == self.corega.system.cloudDeviceId:
                        child[s][f] = 1
            for fogId in range(self.corega.system.fogNumber):
                self.corega.notEnoughResourceFogDeviceWithRepair(child, fogId)
            R.population.append(child)
        if R.population:
            self.corega.calculatePopulationFitnessObjectives(R)
        return R

    def step(self, progress=0.0):
        self._step_count += 1
        # 1) Full NSGA-III generation: primary exploration/convergence engine.
        O = self._offspring()
        P = self.corega.populationPt
        C = P.populationUnion(P, O)

        # 2) Adaptive vectors guide environmental selection after warm-up.
        if float(progress) >= float(self.adaptation_warmup):
            self._adapt_vectors(C)
        idx = environmental_select_nsga3(
            C.fitness, self.populationSize, self.referenceVectors, self.corega.rndEVOL
        )
        self._set_population_from_indices(C, idx)
        Pnext = self.corega.populationPt
        self._update_archive(Pnext)

        # 3) Periodic selective MOPSO exploitation only.
        if self._step_count % max(1, int(self.refinement_interval)) == 0:
            R = self._refine(Pnext)
            if R.population:
                CR = Pnext.populationUnion(Pnext, R)
                idx2 = environmental_select_nsga3(
                    CR.fitness, self.populationSize, self.referenceVectors, self.corega.rndEVOL
                )
                self._set_population_from_indices(CR, idx2)
                self._update_archive(self.corega.populationPt)
