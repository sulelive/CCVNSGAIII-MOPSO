# ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR.py
# Genuine adaptive reference-vector extension for Paper 2.
#
# It inherits the surviving vector-guided hybrid implementation and changes only
# reference-vector adaptation / association / within-front ordering. The MOPSO,
# crossover, mutation, objective functions, population size and generation budget
# remain inherited from the original implementation.

import copy
import numpy as np
import POPULATION as pop
from HYBRID_MOPSO_NSGA3_VECTOR import HYBRID_MOPSO_NSGA3_VECTOR


class ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR(HYBRID_MOPSO_NSGA3_VECTOR):
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        super().__init__(system, populationSeed, evolutionSeed, conf_)
        self.initialReferenceVectors = copy.deepcopy(self.referenceVectors)
        self.adaptation_interval = 1
        self.adaptation_strength = 0.35
        self.vector_occupancy = np.zeros(len(self.referenceVectors), dtype=int)

    @staticmethod
    def _objective_matrix(popT):
        return np.asarray([
            [f["wspr"], f["wmak"], f["wres"]]
            for f in popT.fitness
        ], dtype=float)

    @staticmethod
    def _safe_unit_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms

    def _first_front_indices(self, popT):
        # Reuse inherited nondominated sorting.
        self.fastNonDominatedSort(popT)
        if not popT.fronts:
            return []
        return list(popT.fronts[0])

    def adaptReferenceVectors(self, popT):
        """Adapt directions toward the current first-front distribution while
        retaining the original simplex directions as anchors.

        1) obtain normalized directions of nondominated solutions;
        2) for each reference vector, find the closest current front direction;
        3) blend the original/current reference vector toward that direction;
        4) renormalize to unit length.

        alpha=0.35 prevents collapse and preserves broad coverage.
        """
        front = self._first_front_indices(popT)
        if not front:
            return

        F = self._objective_matrix(popT)[front]
        dirs = self._safe_unit_rows(F)

        new_vectors = []
        alpha = self.adaptation_strength

        for j, ref in enumerate(self.referenceVectors):
            # nearest current nondominated direction by perpendicular distance
            perp = []
            for d in dirs:
                projection = np.dot(d, ref) * ref
                perp.append(np.linalg.norm(d - projection))
            target = dirs[int(np.argmin(perp))]

            # regularize toward the original vector to avoid directional collapse
            anchor = self.initialReferenceVectors[j]
            blended_target = 0.5 * target + 0.5 * anchor
            n = np.linalg.norm(blended_target)
            if n > 0:
                blended_target = blended_target / n

            updated = (1.0 - alpha) * ref + alpha * blended_target
            n = np.linalg.norm(updated)
            new_vectors.append(updated / n if n > 0 else ref)

        self.referenceVectors = np.asarray(new_vectors)

    def assignReferenceVectors(self, popT):
        """Associate each solution with the nearest vector by perpendicular distance
        and store the vector index for occupancy-aware niching.
        """
        self.vector_occupancy = np.zeros(len(self.referenceVectors), dtype=int)

        for i in range(len(popT.population)):
            fitness = popT.fitness[i]
            objectives = np.asarray(
                [fitness["wspr"], fitness["wmak"], fitness["wres"]], dtype=float
            )
            norm = np.linalg.norm(objectives)
            if norm == 0.0:
                norm = 1.0
            direction = objectives / norm

            distances = []
            for ref in self.referenceVectors:
                projection = np.dot(direction, ref) * ref
                distances.append(np.linalg.norm(direction - projection))

            idx = int(np.argmin(distances))
            fitness["referenceVectorIndex"] = idx
            fitness["referenceVector"] = self.referenceVectors[idx].tolist()
            fitness["referenceDistance"] = float(distances[idx])
            self.vector_occupancy[idx] += 1

    def crowdingDistancesAssignments(self, popT, front):
        """Store a selection score in popT.crowdingDistances.

        Higher score is better because inherited crowdedComparisonOrder sorts
        by -distance. Therefore the score rewards:
        - closeness to assigned reference vector;
        - under-occupied reference vectors.

        This fixes the surviving implementation's inversion where larger
        perpendicular distance was preferred.
        """
        for i in front:
            fit = popT.fitness[i]
            ref_dist = float(fit.get("referenceDistance", 0.0))
            ref_idx = int(fit.get("referenceVectorIndex", 0))
            occ = max(1, int(self.vector_occupancy[ref_idx]))

            closeness = 1.0 / (1.0 + ref_dist)
            sparsity = 1.0 / occ
            popT.crowdingDistances[i] = closeness + sparsity

    def evolve(self):
        """Exactly conf.numberGenerations generations; no outer generation loop."""
        print("Starting ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR.evolve")

        for gen in range(self.maxGenerations):
            # MOPSO phase
            self.updateVelocities()
            self.updatePositions()

            for i in range(self.populationSize):
                if self.corega.dominates(
                    self.particles.fitness[i], self.pbest.fitness[i]
                ):
                    self.pbest.population[i] = copy.deepcopy(
                        self.particles.population[i]
                    )
                    self.pbest.fitness[i] = copy.deepcopy(
                        self.particles.fitness[i]
                    )

            self.updateRepository()

            # offspring phase
            orderedFathers = self.corega.orderPopulation(self.particles)
            offspring = self.evolveToOffspring(orderedFathers)
            self.corega.calculatePopulationFitnessObjectives(offspring)

            combined_pop = self.particles.populationUnion(
                self.particles, offspring
            )
            self.corega.calculatePopulationFitnessObjectives(combined_pop)

            # Genuine adaptation uses the evolving population distribution.
            if gen % self.adaptation_interval == 0:
                self.adaptReferenceVectors(combined_pop)

            self.assignReferenceVectors(combined_pop)
            self.fastNonDominatedSort(combined_pop)
            self.calculateCrowdingDistances(combined_pop)
            ordered = self.crowdedComparisonOrder(combined_pop)

            final_population = pop.POPULATION(self.populationSize)
            for i in range(self.populationSize):
                idx = ordered[i]["index"]
                final_population.population[i] = copy.deepcopy(
                    combined_pop.population[idx]
                )
                final_population.fitness[i] = copy.deepcopy(
                    combined_pop.fitness[idx]
                )

            self.particles = final_population
            self.corega.populationPt = self.particles
            self.corega.calculatePopulationFitnessObjectives(self.particles)

        print("Finished ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR.evolve")
