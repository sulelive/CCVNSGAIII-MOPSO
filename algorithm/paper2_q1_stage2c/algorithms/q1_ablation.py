# -*- coding: utf-8 -*-
"""Component-ablation variants for the adaptive reference-vector extension.

The historical Original NSGAIII-MOPSO remains a separate baseline.  These vector-based
variants isolate the proposed extension's three mechanisms:
  A: simplex vectors only (closeness score; no adaptation; no occupancy term)
  B: simplex + adaptive movement (no occupancy term)
  C: simplex + occupancy-aware score (no adaptive movement)
  Proposed: simplex + adaptive movement + occupancy-aware score
"""
import copy
import numpy as np
import POPULATION as pop
from HYBRID_MOPSO_NSGA3_VECTOR import HYBRID_MOPSO_NSGA3_VECTOR


class ABLATION_VECTOR_HYBRID(HYBRID_MOPSO_NSGA3_VECTOR):
    def __init__(self, system, populationSeed, evolutionSeed, conf_, use_adaptation=False, use_occupancy=False, alpha=0.35):
        super().__init__(system, populationSeed, evolutionSeed, conf_)
        self.use_adaptation = bool(use_adaptation)
        self.use_occupancy = bool(use_occupancy)
        self.adaptation_strength = float(alpha)
        self.initialReferenceVectors = copy.deepcopy(self.referenceVectors)
        self.vector_occupancy = np.zeros(len(self.referenceVectors), dtype=int)

    @staticmethod
    def _safe_unit_rows(X):
        X = np.asarray(X, dtype=float)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        return X / n

    def adaptReferenceVectors(self, popT):
        self.fastNonDominatedSort(popT)
        if not popT.fronts or not popT.fronts[0]:
            return
        F = np.asarray([[popT.fitness[i]["wspr"], popT.fitness[i]["wmak"], popT.fitness[i]["wres"]] for i in popT.fronts[0]], dtype=float)
        dirs = self._safe_unit_rows(F)
        alpha = self.adaptation_strength
        new_vectors = []
        for j, ref in enumerate(self.referenceVectors):
            perp = []
            for d in dirs:
                projection = np.dot(d, ref) * ref
                perp.append(np.linalg.norm(d - projection))
            target = dirs[int(np.argmin(perp))]
            anchor = self.initialReferenceVectors[j]
            blended = 0.5 * target + 0.5 * anchor
            n = np.linalg.norm(blended)
            if n > 0:
                blended = blended / n
            updated = (1.0 - alpha) * ref + alpha * blended
            n = np.linalg.norm(updated)
            new_vectors.append(updated / n if n > 0 else ref)
        self.referenceVectors = np.asarray(new_vectors)

    def assignReferenceVectors(self, popT):
        self.vector_occupancy = np.zeros(len(self.referenceVectors), dtype=int)
        for i in range(len(popT.population)):
            fit = popT.fitness[i]
            x = np.asarray([fit["wspr"], fit["wmak"], fit["wres"]], dtype=float)
            n = np.linalg.norm(x)
            if n == 0.0:
                n = 1.0
            d = x / n
            perps = []
            for ref in self.referenceVectors:
                proj = np.dot(d, ref) * ref
                perps.append(np.linalg.norm(d - proj))
            k = int(np.argmin(perps))
            fit["referenceVectorIndex"] = k
            fit["referenceVector"] = self.referenceVectors[k].tolist()
            fit["referenceDistance"] = float(perps[k])
            self.vector_occupancy[k] += 1

    def crowdingDistancesAssignments(self, popT, front):
        for i in front:
            fit = popT.fitness[i]
            d = float(fit.get("referenceDistance", 0.0))
            score = 1.0 / (1.0 + d)  # closeness is always the direction term
            if self.use_occupancy:
                k = int(fit.get("referenceVectorIndex", 0))
                occ = max(1, int(self.vector_occupancy[k]))
                score += 1.0 / occ
            popT.crowdingDistances[i] = score

    def step(self, progress=0.0):
        self.updateVelocities(); self.updatePositions()
        for i in range(self.populationSize):
            if self.corega.dominates(self.particles.fitness[i], self.pbest.fitness[i]):
                self.pbest.population[i] = copy.deepcopy(self.particles.population[i])
                self.pbest.fitness[i] = copy.deepcopy(self.particles.fitness[i])
        self.updateRepository()
        orderedFathers = self.corega.orderPopulation(self.particles)
        offspring = self.evolveToOffspring(orderedFathers)
        self.corega.calculatePopulationFitnessObjectives(offspring)
        combined = self.particles.populationUnion(self.particles, offspring)
        self.corega.calculatePopulationFitnessObjectives(combined)
        if self.use_adaptation:
            self.adaptReferenceVectors(combined)
        self.assignReferenceVectors(combined)
        self.fastNonDominatedSort(combined)
        self.calculateCrowdingDistances(combined)
        ordered = self.crowdedComparisonOrder(combined)
        final = pop.POPULATION(self.populationSize)
        for i in range(self.populationSize):
            idx = ordered[i]["index"]
            final.population[i] = copy.deepcopy(combined.population[idx])
            final.fitness[i] = copy.deepcopy(combined.fitness[idx])
        self.particles = final
        self.corega.populationPt = self.particles
        self.corega.calculatePopulationFitnessObjectives(self.particles)


class AblationA_SimplexOnly(ABLATION_VECTOR_HYBRID):
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_, False, False)

class AblationB_SimplexAdaptive(ABLATION_VECTOR_HYBRID):
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_, True, False)

class AblationC_SimplexOccupancy(ABLATION_VECTOR_HYBRID):
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_, False, True)

class AblationProposed(ABLATION_VECTOR_HYBRID):
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_, True, True)
