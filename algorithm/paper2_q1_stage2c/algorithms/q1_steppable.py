# -*- coding: utf-8 -*-
"""Steppable versions of the six Q1 algorithms.

Each ``step(progress)`` performs one complete algorithmic iteration/generation while
preserving state.  ``progress`` is in [0,1] and is used by RVEA's APD schedule.
The original and adaptive hybrids preserve their supplied operators; only the outer
multi-generation loop is exposed one generation at a time for evaluation-budget control.
"""
from __future__ import division
import copy
import numpy as np
import POPULATION as pop

from algorithms.q1_comparators import NSGA2_Q1, NSGA3_Q1, MOEAD_Q1, RVEA_Q1
from core.q1_common import obj_vector, nondominated_sort
from HYBRID_MOPSO_NSGA3 import HYBRID_MOPSO_NSGA3
from ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR import ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR


class SteppableNSGA2(NSGA2_Q1):
    name = "nsga2"
    def step(self, progress=0.0):
        old = self.maxGenerations
        self.maxGenerations = 1
        try:
            NSGA2_Q1.evolve(self)
        finally:
            self.maxGenerations = old


class SteppableNSGA3(NSGA3_Q1):
    name = "nsga3"
    def step(self, progress=0.0):
        old = self.maxGenerations
        self.maxGenerations = 1
        try:
            NSGA3_Q1.evolve(self)
        finally:
            self.maxGenerations = old


class SteppableMOEAD(MOEAD_Q1):
    name = "moead"
    def step(self, progress=0.0):
        old = self.maxGenerations
        self.maxGenerations = 1
        try:
            MOEAD_Q1.evolve(self)
        finally:
            self.maxGenerations = old


class SteppableRVEA(RVEA_Q1):
    name = "rvea"

    def _environmental_budget(self, C, progress):
        valid = [i for i, f in enumerate(C.fitness) if all(np.isfinite(obj_vector(f)))]
        F = np.vstack([obj_vector(C.fitness[i]) for i in valid])
        ideal = np.min(F, axis=0)
        shifted = F - ideal
        norms = np.linalg.norm(shifted, axis=1)
        norms[norms <= 1e-12] = 1e-12
        dirs = shifted / norms[:, None]
        cos = np.clip(dirs @ self.referenceVectors.T, -1.0, 1.0)
        angles = np.arccos(cos)
        assoc = np.argmin(angles, axis=1)
        theta = 2.0
        m = self.conf.numObjectives
        progress = min(1.0, max(0.0, float(progress)))
        selected = []
        for k in range(len(self.referenceVectors)):
            members = np.where(assoc == k)[0]
            if len(members) == 0:
                continue
            apd = norms[members] * (
                1.0 + m * (progress ** theta) * angles[members, k] / self.gamma[k]
            )
            j = members[int(np.argmin(apd))]
            selected.append(valid[int(j)])
        if len(selected) < self.populationSize:
            fronts = nondominated_sort(C.fitness)
            for fr in fronts:
                for i in fr:
                    if i not in selected:
                        selected.append(i)
                        if len(selected) >= self.populationSize:
                            break
                if len(selected) >= self.populationSize:
                    break
        return selected[:self.populationSize]

    def step(self, progress=0.0):
        O = self._offspring()
        P = self.corega.populationPt
        C = P.populationUnion(P, O)
        F = np.vstack([
            obj_vector(f) for f in C.fitness if all(np.isfinite(obj_vector(f)))
        ])
        # Standard RVEA periodically adapts reference vectors. Under an NFE budget,
        # use progress rather than a nominal generation count. Approx. every 10%.
        bucket = int(min(9, max(0, np.floor(progress * 10.0))))
        last_bucket = getattr(self, "_last_adapt_bucket", -1)
        if bucket > 0 and bucket != last_bucket:
            self._adapt_vectors(F)
            self._last_adapt_bucket = bucket
        idxs = self._environmental_budget(C, progress)
        self._set_population_from_indices(C, idxs)


class SteppableOriginalHybrid(HYBRID_MOPSO_NSGA3):
    name = "original_nsgaiii_mopso"
    def step(self, progress=0.0):
        # One exact generation from the supplied historical hybrid implementation.
        self.updateVelocities()
        self.updatePositions()
        for i in range(self.populationSize):
            if self.corega.dominates(self.particles.fitness[i], self.pbest.fitness[i]):
                self.pbest.population[i] = copy.deepcopy(self.particles.population[i])
                self.pbest.fitness[i] = copy.deepcopy(self.particles.fitness[i])
        self.updateRepository()
        orderedFathers = self.corega.orderPopulation(self.particles)
        offspring = self.evolveToOffspring(orderedFathers)
        self.corega.calculatePopulationFitnessObjectives(offspring)
        combined_pop = self.particles.populationUnion(self.particles, offspring)
        self.assignReferencePoints(combined_pop)
        self.fastNonDominatedSort(combined_pop)
        self.calculateCrowdingDistances(combined_pop)
        ordered_elements = self.crowdedComparisonOrder(combined_pop)
        final_population = pop.POPULATION(self.populationSize)
        for i in range(self.populationSize):
            idx = ordered_elements[i]["index"]
            final_population.population[i] = copy.deepcopy(combined_pop.population[idx])
            final_population.fitness[i] = copy.deepcopy(combined_pop.fitness[idx])
        self.particles = final_population
        self.corega.populationPt = self.particles
        self.corega.calculatePopulationFitnessObjectives(self.particles)


class SteppableAdaptiveHybrid(ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR):
    name = "adaptive_nsgaiii_mopso"
    def step(self, progress=0.0):
        self.updateVelocities()
        self.updatePositions()
        for i in range(self.populationSize):
            if self.corega.dominates(self.particles.fitness[i], self.pbest.fitness[i]):
                self.pbest.population[i] = copy.deepcopy(self.particles.population[i])
                self.pbest.fitness[i] = copy.deepcopy(self.particles.fitness[i])
        self.updateRepository()
        orderedFathers = self.corega.orderPopulation(self.particles)
        offspring = self.evolveToOffspring(orderedFathers)
        self.corega.calculatePopulationFitnessObjectives(offspring)
        combined_pop = self.particles.populationUnion(self.particles, offspring)
        self.corega.calculatePopulationFitnessObjectives(combined_pop)
        self.adaptReferenceVectors(combined_pop)
        self.assignReferenceVectors(combined_pop)
        self.fastNonDominatedSort(combined_pop)
        self.calculateCrowdingDistances(combined_pop)
        ordered = self.crowdedComparisonOrder(combined_pop)
        final_population = pop.POPULATION(self.populationSize)
        for i in range(self.populationSize):
            idx = ordered[i]["index"]
            final_population.population[i] = copy.deepcopy(combined_pop.population[idx])
            final_population.fitness[i] = copy.deepcopy(combined_pop.fitness[idx])
        self.particles = final_population
        self.corega.populationPt = self.particles
        self.corega.calculatePopulationFitnessObjectives(self.particles)


STEPPABLE_ALGORITHMS = {
    "nsga2": SteppableNSGA2,
    "nsga3": SteppableNSGA3,
    "moead": SteppableMOEAD,
    "rvea": SteppableRVEA,
    "original_nsgaiii_mopso": SteppableOriginalHybrid,
    "adaptive_nsgaiii_mopso": SteppableAdaptiveHybrid,
}
