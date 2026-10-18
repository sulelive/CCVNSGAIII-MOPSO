# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:38:02 2025

@author: HP
"""

import numpy as np
import random
import copy
from datetime import datetime
import POPULATION as pop
import GAcore as gacore
import CONFIG as config
import SYSTEMMODEL as systemmodel

class HYBRID_MOPSO_NSGA3_VECTOR:
    def __init__(self, system, populationSeed, evolutionSeed, conf_):
        self.corega = gacore.GAcore(system, populationSeed, evolutionSeed, conf_)
        self.conf = conf_
        self.populationSize = conf_.populationSize
        self.maxGenerations = conf_.numberGenerations
        self.w = 0.7  # Inertia weight (MOPSO)
        self.c1 = 1.5  # Cognitive parameter (MOPSO)
        self.c2 = 1.5  # Social parameter (MOPSO)
        self.referenceVectors = self.generateReferenceVectors(conf_.numObjectives)
        self.initializePopulation()

    def generateReferenceVectors(self, numObjectives):
        """Generate uniformly distributed reference vectors on the unit hyperplane."""
        def generate_vectors(p, numObjectives, current_vector=None, start_idx=0):
            if current_vector is None:
                current_vector = []
            if len(current_vector) == numObjectives - 1:
                current_vector.append(p - sum(current_vector))
                if current_vector[-1] >= 0:
                    # Normalize to unit vector
                    vector = np.array([x / p for x in current_vector])
                    norm = np.linalg.norm(vector)
                    return [vector / norm] if norm > 0 else []
                return []
            vectors = []
            for i in range(p + 1 - sum(current_vector)):
                new_vector = current_vector + [i]
                vectors.extend(generate_vectors(p, numObjectives, new_vector, start_idx + 1))
            return vectors

        p = 4  # Number of divisions (controls number of vectors)
        vectors = generate_vectors(p, numObjectives)
        return np.array(vectors)

    def initializePopulation(self):
        """Initialize particles, velocities, and repository."""
        self.particles = pop.POPULATION(self.populationSize)
        self.corega.generatePopulation(self.particles)
        self.velocities = [{} for _ in range(self.populationSize)]  # Velocities per (service, fog) tuple
        self.pbest = copy.deepcopy(self.particles)
        self.gbest = pop.POPULATION(0)
        self.corega.calculatePopulationFitnessObjectives(self.particles)
        self.updateRepository()  # Initialize gbest with non-dominated solutions

    def updateVelocities(self):
        """Update particle velocities (MOPSO)."""
        for i in range(self.populationSize):
            for service_idx in range(len(self.particles.population[i])):
                for fog_idx in range(len(self.particles.population[i][service_idx])):
                    r1 = self.corega.rndEVOL.random()
                    r2 = self.corega.rndEVOL.random()
                    current_val = self.particles.population[i][service_idx][fog_idx]
                    pbest_val = self.pbest.population[i][service_idx][fog_idx]
                    gbest_idx = self.corega.rndEVOL.randint(0, len(self.gbest.population)-1) if self.gbest.population else 0
                    gbest_val = self.gbest.population[gbest_idx][service_idx][fog_idx] if self.gbest.population else current_val
                    cognitive = self.c1 * r1 * (pbest_val - current_val)
                    social = self.c2 * r2 * (gbest_val - current_val)
                    self.velocities[i][(service_idx, fog_idx)] = (
                        self.w * self.velocities[i].get((service_idx, fog_idx), 0) + cognitive + social
                    )

    def updatePositions(self):
        """Update particle positions (MOPSO)."""
        for i in range(self.populationSize):
            for service_idx in range(len(self.particles.population[i])):
                for fog_idx in range(len(self.particles.population[i][service_idx])):
                    velocity = self.velocities[i].get((service_idx, fog_idx), 0)
                    prob = 1 / (1 + np.exp(-velocity))  # Sigmoid for binary
                    self.particles.population[i][service_idx][fog_idx] = 1 if self.corega.rndEVOL.random() < prob else 0
                    if fog_idx == self.corega.system.cloudDeviceId:
                        self.particles.population[i][service_idx][fog_idx] = 1
        self.corega.calculatePopulationFitnessObjectives(self.particles)

    def updateRepository(self):
        """Update global best repository with non-dominated solutions."""
        combined_pop = self.gbest.populationUnion(self.gbest, self.particles)
        self.corega.calculatePopulationFitnessObjectives(combined_pop)
        combined_pop.fronts = [[] for _ in range(len(combined_pop.population))]
        combined_pop.crowdingDistances = [0.0] * len(combined_pop.population)
        self.fastNonDominatedSort(combined_pop)
        self.gbest = pop.POPULATION(0)
        front_idx = 0
        repo_size = self.populationSize // 2  # Limit repository size
        while (front_idx < len(combined_pop.fronts) and
               len(self.gbest.population) + len(combined_pop.fronts[front_idx]) <= repo_size):
            self.gbest.population.extend([combined_pop.population[i] for i in combined_pop.fronts[front_idx]])
            front_idx += 1
        if front_idx < len(combined_pop.fronts) and len(self.gbest.population) < repo_size:
            last_front = sorted(combined_pop.fronts[front_idx],
                                key=lambda x: combined_pop.crowdingDistances[x], reverse=True)
            remaining = repo_size - len(self.gbest.population)
            self.gbest.population.extend([combined_pop.population[i] for i in last_front[:remaining]])

    def fastNonDominatedSort(self, popT):
        """Perform non-dominated sorting (NSGA-III)."""
        popT.dominationCount = [0] * len(popT.population)
        popT.dominatedBy = [set() for _ in range(len(popT.population))]
        popT.dominatesTo = [set() for _ in range(len(popT.population))]
        for p in range(len(popT.population)):
            for q in range(p + 1, len(popT.population)):
                if self.corega.dominates(popT.fitness[p], popT.fitness[q]):
                    popT.dominatesTo[p].add(q)
                    popT.dominatedBy[q].add(p)
                elif self.corega.dominates(popT.fitness[q], popT.fitness[p]):
                    popT.dominatedBy[p].add(q)
                    popT.dominatesTo[q].add(p)
        popT.fronts = []
        remaining = set(range(len(popT.population)))
        while remaining:
            front = [i for i in remaining if popT.dominatedBy[i] == set()]
            popT.fronts.append(front)
            for i in front:
                for j in popT.dominatesTo[i]:
                    popT.dominatedBy[j].remove(i)
            remaining -= set(front)

    def assignReferenceVectors(self, popT):
        """Assign solutions to reference vectors based on perpendicular distance."""
        for i in range(len(popT.population)):
            fitness = popT.fitness[i]
            objectives = np.array([fitness['wspr'], fitness['wmak'], fitness['wres']])
            # Normalize objectives
            norm = np.linalg.norm(objectives)
            if norm == 0:
                norm = 1.0  # Avoid division by zero
            normalized_obj = objectives / norm
            # Compute perpendicular distances to each reference vector
            distances = []
            for ref_vec in self.referenceVectors:
                # Projection of normalized_obj onto ref_vec
                projection = np.dot(normalized_obj, ref_vec) * ref_vec
                # Perpendicular distance = norm of (normalized_obj - projection)
                perp_dist = np.linalg.norm(normalized_obj - projection)
                distances.append(perp_dist)
            closest_ref_idx = np.argmin(distances)
            fitness['referenceVector'] = self.referenceVectors[closest_ref_idx].tolist()

    def crowdingDistancesAssignments(self, popT, front):
        """Assign crowding distances based on perpendicular distance to reference vectors."""
        for i in front:
            popT.crowdingDistances[i] = 0.0
        for i in front:
            fitness = popT.fitness[i]
            ref_vec = np.array(fitness['referenceVector'])
            objectives = np.array([fitness['wspr'], fitness['wmak'], fitness['wres']])
            norm = np.linalg.norm(objectives)
            if norm == 0:
                norm = 1.0
            normalized_obj = objectives / norm
            projection = np.dot(normalized_obj, ref_vec) * ref_vec
            dist = np.linalg.norm(normalized_obj - projection)
            popT.crowdingDistances[i] = dist

    def calculateCrowdingDistances(self, popT):
        """Calculate crowding distances for all fronts."""
        for front in popT.fronts:
            self.crowdingDistancesAssignments(popT, front)

    def crowdedComparisonOrder(self, popT):
        """Order population by rank and crowding distance."""
        valuesToOrder = []
        for i, v in enumerate(popT.crowdingDistances):
            citizen = {"index": i, "distance": v, "rank": 0}
            valuesToOrder.append(citizen)
        for f, front in enumerate(popT.fronts):
            for i in front:
                valuesToOrder[i]["rank"] = f
        return sorted(valuesToOrder, key=lambda k: (k["rank"], -k["distance"]))

    def evolveToOffspring(self, orderedFathers):
        """Generate offspring using crossover and mutation (NSGA-III)."""
        offspring = pop.POPULATION(self.populationSize)
        offspring.population = []
        while len(offspring.population) < self.populationSize:
            father1 = self.corega.fatherSelection(orderedFathers)
            father2 = father1
            while father1 == father2:
                father2 = self.corega.fatherSelection(orderedFathers)
            c1, c2 = self.corega.crossover(self.particles.population[father1], self.particles.population[father2])
            if len(offspring.population) < self.populationSize:
                offspring.population.append(c1)
            if len(offspring.population) < self.populationSize:
                offspring.population.append(c2)
        for child in offspring.population:
            if self.corega.rndEVOL.uniform(0, 1) < self.corega.mutationProbability:
                self.corega.mutate(child)
            for fogId in range(self.corega.system.fogNumber):
                self.corega.notEnoughResourceFogDeviceWithRepair(child, fogId)
        return offspring

    def evolve(self):
        """Main evolution loop combining MOPSO and NSGA-III."""
        print("Starting HYBRID_MOPSO_NSGA3_VECTOR.evolve")
        for gen in range(self.maxGenerations):
            # MOPSO Phase: Update particles
            self.updateVelocities()
            self.updatePositions()
            for i in range(self.populationSize):
                if self.corega.dominates(self.particles.fitness[i], self.pbest.fitness[i]):
                    self.pbest.population[i] = copy.deepcopy(self.particles.population[i])
                    self.pbest.fitness[i] = copy.deepcopy(self.particles.fitness[i])
            self.updateRepository()

            # NSGA-III Phase: Generate offspring
            orderedFathers = self.corega.orderPopulation(self.particles)
            offspring = self.evolveToOffspring(orderedFathers)
            self.corega.calculatePopulationFitnessObjectives(offspring)

            # Combine and select next population
            combined_pop = self.particles.populationUnion(self.particles, offspring)
            self.assignReferenceVectors(combined_pop)
            self.fastNonDominatedSort(combined_pop)
            self.calculateCrowdingDistances(combined_pop)
            ordered_elements = self.crowdedComparisonOrder(combined_pop)

            # Create next population
            final_population = pop.POPULATION(self.populationSize)
            for i in range(self.populationSize):
                final_population.population[i] = copy.deepcopy(combined_pop.population[ordered_elements[i]["index"]])
                final_population.fitness[i] = copy.deepcopy(combined_pop.fitness[ordered_elements[i]["index"]])
            self.particles = final_population
            self.corega.populationPt = self.particles
            self.corega.calculatePopulationFitnessObjectives(self.particles)

        print("Finished HYBRID_MOPSO_NSGA3_VECTOR.evolve")