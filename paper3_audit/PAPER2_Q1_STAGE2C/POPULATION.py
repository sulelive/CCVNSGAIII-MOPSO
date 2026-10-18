#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import copy


class POPULATION(object):

    def __init__(self, size):
        """
        Initialize a population container with independent objects.
        """
        self.population = [[] for _ in range(size)]
        self.fitness = [{} for _ in range(size)]
        self.fitnessNormalized = [{} for _ in range(size)]
        self.dominatesTo = [set() for _ in range(size)]
        self.dominatedBy = [set() for _ in range(size)]
        self.fronts = [[] for _ in range(size)]
        self.crowdingDistances = [0.0 for _ in range(size)]
        self.vmsUsages = [[] for _ in range(size)]
        self.pmsUsages = [[] for _ in range(size)]
        self.dominationCount = [0 for _ in range(size)]

    def __len__(self):
        return len(self.population)

    def populationUnion(self, pop1, pop2):
        """
        Create the union of two populations with safe deep copies.
        """
        total_size = len(pop1.population) + len(pop2.population)
        newPop = POPULATION(0)

        newPop.population = copy.deepcopy(pop1.population) + copy.deepcopy(pop2.population)
        newPop.fitness = copy.deepcopy(pop1.fitness) + copy.deepcopy(pop2.fitness)
        newPop.fitnessNormalized = copy.deepcopy(pop1.fitnessNormalized) + copy.deepcopy(pop2.fitnessNormalized)
        newPop.dominatesTo = copy.deepcopy(pop1.dominatesTo) + copy.deepcopy(pop2.dominatesTo)
        newPop.dominatedBy = copy.deepcopy(pop1.dominatedBy) + copy.deepcopy(pop2.dominatedBy)
        newPop.fronts = copy.deepcopy(pop1.fronts) + copy.deepcopy(pop2.fronts)
        newPop.crowdingDistances = copy.deepcopy(pop1.crowdingDistances) + copy.deepcopy(pop2.crowdingDistances)
        newPop.vmsUsages = copy.deepcopy(pop1.vmsUsages) + copy.deepcopy(pop2.vmsUsages)
        newPop.pmsUsages = copy.deepcopy(pop1.pmsUsages) + copy.deepcopy(pop2.pmsUsages)
        newPop.dominationCount = copy.deepcopy(pop1.dominationCount) + copy.deepcopy(pop2.dominationCount)

        while len(newPop.fitness) < total_size:
            newPop.fitness.append({})
        while len(newPop.fitnessNormalized) < total_size:
            newPop.fitnessNormalized.append({})
        while len(newPop.dominatesTo) < total_size:
            newPop.dominatesTo.append(set())
        while len(newPop.dominatedBy) < total_size:
            newPop.dominatedBy.append(set())
        while len(newPop.fronts) < total_size:
            newPop.fronts.append([])
        while len(newPop.crowdingDistances) < total_size:
            newPop.crowdingDistances.append(0.0)
        while len(newPop.vmsUsages) < total_size:
            newPop.vmsUsages.append([])
        while len(newPop.pmsUsages) < total_size:
            newPop.pmsUsages.append([])
        while len(newPop.dominationCount) < total_size:
            newPop.dominationCount.append(0)

        for i, fit in enumerate(newPop.fitness):
            if isinstance(fit, dict):
                fit["index"] = i

        return newPop

    def paretoExport(self):
        """
        Export the first Pareto front into a new population.
        """
        if len(self.fronts) == 0:
            return POPULATION(0)

        first_front = self.fronts[0]
        paretoPop = POPULATION(0)

        paretoPop.population = []
        paretoPop.fitness = []
        paretoPop.fitnessNormalized = []
        paretoPop.dominatesTo = []
        paretoPop.dominatedBy = []
        paretoPop.fronts = []
        paretoPop.crowdingDistances = []
        paretoPop.vmsUsages = []
        paretoPop.pmsUsages = []
        paretoPop.dominationCount = []

        for idx in first_front:
            paretoPop.population.append(copy.deepcopy(self.population[idx]))
            paretoPop.fitness.append(copy.deepcopy(self.fitness[idx]))
            paretoPop.fitnessNormalized.append(copy.deepcopy(self.fitnessNormalized[idx]))
            paretoPop.dominatesTo.append(copy.deepcopy(self.dominatesTo[idx]))
            paretoPop.dominatedBy.append(copy.deepcopy(self.dominatedBy[idx]))
            paretoPop.fronts.append([])
            paretoPop.crowdingDistances.append(copy.deepcopy(self.crowdingDistances[idx]))
            paretoPop.vmsUsages.append(copy.deepcopy(self.vmsUsages[idx]))
            paretoPop.pmsUsages.append(copy.deepcopy(self.pmsUsages[idx]))
            paretoPop.dominationCount.append(copy.deepcopy(self.dominationCount[idx]))

        return paretoPop

    def replaceSolution(self, newPop, newId, oldPop, oldId):
        oldPop.population[oldId] = copy.deepcopy(newPop.population[newId])
        oldPop.fitness[oldId] = copy.deepcopy(newPop.fitness[newId])
        oldPop.fitnessNormalized[oldId] = copy.deepcopy(newPop.fitnessNormalized[newId])
        oldPop.dominatesTo[oldId] = copy.deepcopy(newPop.dominatesTo[newId])
        oldPop.dominatedBy[oldId] = copy.deepcopy(newPop.dominatedBy[newId])
        oldPop.fronts[oldId] = copy.deepcopy(newPop.fronts[newId])
        oldPop.crowdingDistances[oldId] = copy.deepcopy(newPop.crowdingDistances[newId])
        oldPop.vmsUsages[oldId] = copy.deepcopy(newPop.vmsUsages[newId])
        oldPop.pmsUsages[oldId] = copy.deepcopy(newPop.pmsUsages[newId])
        oldPop.dominationCount[oldId] = copy.deepcopy(newPop.dominationCount[newId])

    def remove(self, idx):
        del self.population[idx]
        del self.fitness[idx]
        del self.fitnessNormalized[idx]
        del self.dominatesTo[idx]
        del self.dominatedBy[idx]
        del self.fronts[idx]
        del self.crowdingDistances[idx]
        del self.vmsUsages[idx]
        del self.pmsUsages[idx]
        del self.dominationCount[idx]

    def appendAndIncrease(self, solution):
        self.population.append(copy.deepcopy(solution))
        self.fitness.append({})
        self.fitnessNormalized.append({})
        self.dominatesTo.append(set())
        self.dominatedBy.append(set())
        self.fronts.append([])
        self.crowdingDistances.append(0.0)
        self.vmsUsages.append([])
        self.pmsUsages.append([])
        self.dominationCount.append(0)