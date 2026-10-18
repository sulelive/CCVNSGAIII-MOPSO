# -*- coding: utf-8 -*-
"""Q1 comparator implementations for the existing binary fog-placement encoding.

The implementations reuse the project's initialization, crossover, mutation, repair,
and objective functions, but use standard weak Pareto dominance in their own selection logic.
"""
from __future__ import division
import copy
import math
import numpy as np
import POPULATION as pop
import GAcore as gacore

from core.q1_common import (
    OBJ_KEYS, obj_vector, dominates_weak, nondominated_sort, rank_and_crowding,
    crowding_distance, das_dennis_reference_vectors, environmental_select_nsga3,
    normalize_objectives, associate_to_vectors
)


class _BaseEA(object):
    name = "base"
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        self.corega = gacore.GAcore(system, populationSeed, evolutionSeed, conf_)
        self.conf = conf_; self.populationSize = conf_.populationSize; self.maxGenerations = conf_.numberGenerations
        self.corega.generatePopulation(self.corega.populationPt)
        self.history = []

    def _offspring(self, parent_indices=None):
        P = self.corega.populationPt
        offspring = pop.POPULATION(0); offspring.population = []
        if parent_indices is None:
            parent_indices = list(range(len(P.population)))
        while len(offspring.population) < self.populationSize:
            a = parent_indices[self.corega.rndEVOL.randint(0, len(parent_indices)-1)]
            b = a
            while b == a:
                b = parent_indices[self.corega.rndEVOL.randint(0, len(parent_indices)-1)]
            c1, c2 = self.corega.crossover(P.population[a], P.population[b])
            for c in (c1, c2):
                if len(offspring.population) >= self.populationSize: break
                if self.corega.rndEVOL.random() < self.corega.mutationProbability:
                    self.corega.mutate(c)
                for fogId in range(self.corega.system.fogNumber):
                    self.corega.notEnoughResourceFogDeviceWithRepair(c, fogId)
                offspring.population.append(c)
        self.corega.calculatePopulationFitnessObjectives(offspring)
        return offspring

    def _set_population_from_indices(self, combined, idxs):
        P = pop.POPULATION(self.populationSize)
        for j, i in enumerate(idxs[:self.populationSize]):
            P.population[j] = copy.deepcopy(combined.population[i])
            P.fitness[j] = copy.deepcopy(combined.fitness[i])
        self.corega.populationPt = P


class NSGA2_Q1(_BaseEA):
    name = "nsga2"
    def _binary_tournament(self):
        fit = self.corega.populationPt.fitness
        _, rank, crowd = rank_and_crowding(fit)
        def pick():
            a = self.corega.rndEVOL.randint(0, self.populationSize-1)
            b = self.corega.rndEVOL.randint(0, self.populationSize-1)
            if rank[a] < rank[b]: return a
            if rank[b] < rank[a]: return b
            if crowd[a] > crowd[b]: return a
            if crowd[b] > crowd[a]: return b
            return a if self.corega.rndEVOL.random() < 0.5 else b
        return pick

    def evolve(self):
        for _ in range(self.maxGenerations):
            P = self.corega.populationPt
            picker = self._binary_tournament()
            O = pop.POPULATION(0); O.population = []
            while len(O.population) < self.populationSize:
                a, b = picker(), picker()
                while b == a: b = picker()
                c1, c2 = self.corega.crossover(P.population[a], P.population[b])
                for c in (c1, c2):
                    if len(O.population) >= self.populationSize: break
                    if self.corega.rndEVOL.random() < self.corega.mutationProbability: self.corega.mutate(c)
                    for fogId in range(self.corega.system.fogNumber): self.corega.notEnoughResourceFogDeviceWithRepair(c, fogId)
                    O.population.append(c)
            self.corega.calculatePopulationFitnessObjectives(O)
            C = P.populationUnion(P, O)
            fronts = nondominated_sort(C.fitness)
            idxs = []
            for fr in fronts:
                if len(idxs) + len(fr) <= self.populationSize:
                    idxs.extend(fr)
                else:
                    cd = crowding_distance(C.fitness, fr)
                    idxs.extend(sorted(fr, key=lambda i: cd[i], reverse=True)[:self.populationSize-len(idxs)])
                    break
            self._set_population_from_indices(C, idxs)


class NSGA3_Q1(_BaseEA):
    name = "nsga3"
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_)
        self.referenceVectors = das_dennis_reference_vectors(conf_.numObjectives, p=19)

    def evolve(self):
        for _ in range(self.maxGenerations):
            O = self._offspring()
            P = self.corega.populationPt
            C = P.populationUnion(P, O)
            idxs = environmental_select_nsga3(C.fitness, self.populationSize, self.referenceVectors, self.corega.rndEVOL)
            self._set_population_from_indices(C, idxs)


class MOEAD_Q1(_BaseEA):
    name = "moead"
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_)
        V = das_dennis_reference_vectors(conf_.numObjectives, p=19)
        if len(V) >= self.populationSize:
            # deterministic spread-preserving subsample
            idx = np.linspace(0, len(V)-1, self.populationSize).round().astype(int)
            self.weights = V[idx]
        else:
            reps = int(math.ceil(self.populationSize / float(len(V))))
            self.weights = np.vstack([V for _ in range(reps)])[:self.populationSize]
        D = np.linalg.norm(self.weights[:,None,:] - self.weights[None,:,:], axis=2)
        T = min(int(getattr(conf_, "T", 20)), self.populationSize)
        self.neighbors = np.argsort(D, axis=1)[:, :T]
        F = np.vstack([obj_vector(f) for f in self.corega.populationPt.fitness])
        self.ideal = np.min(F, axis=0)

    def _scalar(self, f, w):
        x = obj_vector(f)
        ww = np.maximum(w, 1e-6)
        return float(np.max(ww * np.abs(x - self.ideal)))

    def evolve(self):
        P = self.corega.populationPt
        for _ in range(self.maxGenerations):
            for i in range(self.populationSize):
                neigh = self.neighbors[i]
                a = int(neigh[self.corega.rndEVOL.randint(0, len(neigh)-1)])
                b = a
                while b == a: b = int(neigh[self.corega.rndEVOL.randint(0, len(neigh)-1)])
                c1, _ = self.corega.crossover(P.population[a], P.population[b])
                child = c1
                if self.corega.rndEVOL.random() < self.corega.mutationProbability: self.corega.mutate(child)
                for fogId in range(self.corega.system.fogNumber): self.corega.notEnoughResourceFogDeviceWithRepair(child, fogId)
                fit = self.corega.calculateFitnessObjectives(child); fit["index"] = -1
                self.ideal = np.minimum(self.ideal, obj_vector(fit))
                # canonical neighborhood replacement; cap replacements to avoid takeover
                replaced = 0
                order = list(neigh); self.corega.rndEVOL.shuffle(order)
                for j in order:
                    j = int(j)
                    if self._scalar(fit, self.weights[j]) <= self._scalar(P.fitness[j], self.weights[j]):
                        P.population[j] = copy.deepcopy(child); P.fitness[j] = copy.deepcopy(fit); P.fitness[j]["index"] = j
                        replaced += 1
                        if replaced >= 2: break
            self.corega.populationPt = P


class RVEA_Q1(_BaseEA):
    name = "rvea"
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_)
        V = das_dennis_reference_vectors(conf_.numObjectives, p=19)
        if len(V) >= self.populationSize:
            idx = np.linspace(0, len(V)-1, self.populationSize).round().astype(int)
            V = V[idx]
        self.initialVectors = V.copy(); self.referenceVectors = V.copy()
        # minimum inter-vector angle for APD scaling
        cos = np.clip(self.referenceVectors @ self.referenceVectors.T, -1.0, 1.0)
        np.fill_diagonal(cos, -1.0)
        self.gamma = np.min(np.arccos(cos), axis=1)
        self.gamma[self.gamma <= 1e-12] = 1.0

    def _adapt_vectors(self, F):
        _, ideal, nadir = normalize_objectives(F)
        ranges = nadir - ideal; ranges[ranges <= 1e-12] = 1.0
        V = self.initialVectors * ranges
        n = np.linalg.norm(V, axis=1, keepdims=True); n[n <= 1e-12] = 1.0
        self.referenceVectors = V / n
        cos = np.clip(self.referenceVectors @ self.referenceVectors.T, -1.0, 1.0)
        np.fill_diagonal(cos, -1.0)
        self.gamma = np.min(np.arccos(cos), axis=1); self.gamma[self.gamma <= 1e-12] = 1.0

    def _environmental(self, C, gen):
        valid = [i for i,f in enumerate(C.fitness) if all(np.isfinite(obj_vector(f)))]
        F = np.vstack([obj_vector(C.fitness[i]) for i in valid])
        ideal = np.min(F, axis=0); shifted = F - ideal
        norms = np.linalg.norm(shifted, axis=1); norms[norms <= 1e-12] = 1e-12
        dirs = shifted / norms[:,None]
        cos = np.clip(dirs @ self.referenceVectors.T, -1.0, 1.0)
        angles = np.arccos(cos); assoc = np.argmin(angles, axis=1)
        theta = 2.0; m = self.conf.numObjectives
        progress = float(gen + 1) / float(max(1, self.maxGenerations))
        selected = []
        for k in range(len(self.referenceVectors)):
            members = np.where(assoc == k)[0]
            if len(members) == 0: continue
            apd = norms[members] * (1.0 + m * (progress ** theta) * angles[members, k] / self.gamma[k])
            j = members[int(np.argmin(apd))]
            selected.append(valid[int(j)])
        if len(selected) < self.populationSize:
            # fill with best Pareto-ranked remaining candidates
            fronts = nondominated_sort(C.fitness)
            for fr in fronts:
                for i in fr:
                    if i not in selected:
                        selected.append(i)
                        if len(selected) >= self.populationSize: break
                if len(selected) >= self.populationSize: break
        return selected[:self.populationSize]

    def evolve(self):
        for gen in range(self.maxGenerations):
            O = self._offspring()
            P = self.corega.populationPt; C = P.populationUnion(P, O)
            F = np.vstack([obj_vector(f) for f in C.fitness if all(np.isfinite(obj_vector(f)))])
            if gen > 0 and gen % max(1, int(0.1 * self.maxGenerations)) == 0:
                self._adapt_vectors(F)
            idxs = self._environmental(C, gen)
            self._set_population_from_indices(C, idxs)
