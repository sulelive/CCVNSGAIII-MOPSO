"""Phase 5 CPU research implementation: chaotic initialization, context-fresh pools,
strict preflight search-evaluation budgets. Not GPU accelerated.
"""
import copy
import numpy as np
import POPULATION as pop
from q1_cooperative_phase4 import CCVNSGAIII_MOPSO_CPU_P4
from core.q1_common import environmental_select_nsga3

class CCVNSGAIII_MOPSO_CPU_P5(CCVNSGAIII_MOPSO_CPU_P4):
    name = 'ccv_nsgaiii_mopso_cpu_phase5'
    def __init__(self, *args, chaos_seed=0.317, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 < chaos_seed < 1.0:
            raise ValueError('chaos_seed must lie strictly between 0 and 1')
        self.chaos_seed = float(chaos_seed)
        self.pool_refresh_evaluations = 0
        self.chaotic_initial_evaluations = 0
        self._chaotic_initialize()

    def _chaotic_initialize(self):
        P = self.corega.populationPt
        z = self.chaos_seed
        cloud = self.corega.system.cloudDeviceId
        for chromosome in P.population:
            for row in chromosome:
                for fog in range(self.corega.system.fogNumber):
                    z = 4.0 * z * (1.0 - z)
                    row[fog] = int(z >= 0.5)
                row[cloud] = 1
            self._repair(chromosome)
        before = self.corega.evaluation_count
        self.corega.calculatePopulationFitnessObjectives(P)
        self.chaotic_initial_evaluations = self.corega.evaluation_count - before
        self._archive = pop.POPULATION(0)
        self._update_archive(P)

    def _cooperative_round(self):
        current = self.corega.populationPt
        if not current.population:
            return
        groups = self._partition(len(current.population[0]))
        self.last_groups = groups
        snapshot = copy.deepcopy(current.population)
        fits = copy.deepcopy(current.fitness)
        if self.subpopulations is None:
            self.subpopulations = self._init_subpopulations(groups, snapshot, fits)
        # A single frozen canonical context makes all pool entries in this round
        # comparable; historical context-dependent fitness is NEVER reused.
        canonical = snapshot[0]
        refresh = pop.POPULATION(0)
        refresh.population = []
        for group, pool in zip(groups, self.subpopulations):
            for genes, _old_fitness in pool:
                candidate = copy.deepcopy(canonical)
                for local_i, service in enumerate(group):
                    candidate[service] = copy.deepcopy(genes[local_i])
                self._repair(candidate)
                refresh.population.append(candidate)
                self.context_reconstructions += 1
        before = self.corega.evaluation_count
        self.corega.calculatePopulationFitnessObjectives(refresh)
        self.pool_refresh_evaluations += self.corega.evaluation_count - before
        self.context_evaluations += self.corega.evaluation_count - before
        offset = 0
        for group_id, group in enumerate(groups):
            updated = []
            for i, (genes, _old_fitness) in enumerate(self.subpopulations[group_id]):
                # Repair may change other groups; retain effective active genes.
                complete = refresh.population[offset + i]
                updated.append((copy.deepcopy([complete[s] for s in group]),
                                copy.deepcopy(refresh.fitness[offset + i])))
            self.subpopulations[group_id] = updated
            offset += len(updated)
        proposals = pop.POPULATION(0)
        proposals.population = []
        origins = []
        for group_id, group in enumerate(groups):
            pool = self.subpopulations[group_id]
            for _ in range(self.proposals_per_group):
                donor_idx = self.corega.rndEVOL.randint(0, len(pool)-1)
                candidate = copy.deepcopy(canonical)
                for local_i, service in enumerate(group):
                    candidate[service] = copy.deepcopy(pool[donor_idx][0][local_i])
                before_mutation = copy.deepcopy(candidate)
                self.corega.mutate(candidate)
                active = set(group)
                for service in range(len(candidate)):
                    if service not in active:
                        candidate[service] = before_mutation[service]
                self._repair(candidate)
                proposals.population.append(candidate)
                origins.append(group_id)
                self.context_reconstructions += 1
        before = self.corega.evaluation_count
        self.corega.calculatePopulationFitnessObjectives(proposals)
        self.context_evaluations += self.corega.evaluation_count - before
        for group_id, group in enumerate(groups):
            entries = self.subpopulations[group_id] + [
                (copy.deepcopy([proposals.population[i][s] for s in group]),
                 copy.deepcopy(proposals.fitness[i]))
                for i, origin in enumerate(origins) if origin == group_id]
            idx = environmental_select_nsga3(
                [fit for _, fit in entries],
                min(self.subpopulation_size, len(entries)),
                self.referenceVectors, self.corega.rndEVOL)
            self.subpopulations[group_id] = [copy.deepcopy(entries[i]) for i in idx]
        combined = current.populationUnion(current, proposals)
        idx = environmental_select_nsga3(combined.fitness, self.populationSize,
                                         self.referenceVectors, self.corega.rndEVOL)
        self._set_population_from_indices(combined, idx)
        self._update_archive(self.corega.populationPt)
        self.cooperative_rounds += 1

    def next_step_cost(self):
        next_step = self._step_count + 1
        cost = self.populationSize
        if next_step % self.refinement_interval == 0:
            cost += max(1, int(round(self.populationSize * self.refinement_fraction)))
        if next_step % self.cooperative_interval == 0:
            n_groups = min(self.groups, self.corega.system.serviceNumber)
            pool_count = sum(len(pool) for pool in self.subpopulations) if self.subpopulations is not None else n_groups * min(self.subpopulation_size, self.populationSize)
            cost += pool_count + n_groups * self.proposals_per_group
        return cost

    def run_budget(self, search_budget):
        budget = int(search_budget)
        if budget < 1:
            raise ValueError('search_budget must be positive')
        initial = self.corega.evaluation_count
        while self.corega.evaluation_count - initial + self.next_step_cost() <= budget:
            used = self.corega.evaluation_count - initial
            predicted = self.next_step_cost()
            self.step(progress=used / budget)
            actual = self.corega.evaluation_count - initial - used
            if actual != predicted:
                raise AssertionError(f'evaluation preflight mismatch: {predicted} vs {actual}')
        actual = self.corega.evaluation_count - initial
        return {'initial_evaluations': initial, 'chaotic_initial_evaluations': self.chaotic_initial_evaluations,
                'search_evaluations': actual, 'budget': budget, 'unused_budget': budget-actual,
                'overshoot': 0, 'cooperative_context_evaluations': self.context_evaluations,
                'pool_refresh_evaluations': self.pool_refresh_evaluations,
                'context_reconstructions': self.context_reconstructions,
                'cooperative_rounds': self.cooperative_rounds, 'steps': self._step_count,
                'next_step_cost': self.next_step_cost()}
