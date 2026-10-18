"""Budget-accounted, sequential cooperative subpopulation prototype.

Each group owns a persistent pool of group-specific genes and complete-context
fitness. All proposals use a frozen context snapshot; synchronization occurs once
per round. No claim of GPU acceleration or completed chaotic initialization.
"""
import copy
import numpy as np
import POPULATION as pop
from algorithms.q1_stage2c import ADAPTIVE_SELECTIVE_MOPSO_NSGA3_STAGE2C
from core.q1_common import environmental_select_nsga3, obj_vector

class CCVNSGAIII_MOPSO_CPU_P4(ADAPTIVE_SELECTIVE_MOPSO_NSGA3_STAGE2C):
    name = 'ccv_nsgaiii_mopso_cpu_phase4'
    def __init__(self, system, populationSeed, evolutionSeed, conf_, groups=4,
                 cooperative_interval=5, subpopulation_size=4, proposals_per_group=2):
        super().__init__(system, populationSeed, evolutionSeed, conf_)
        self.groups = int(groups)
        self.cooperative_interval = int(cooperative_interval)
        self.subpopulation_size = int(subpopulation_size)
        self.proposals_per_group = int(proposals_per_group)
        if min(self.groups, self.cooperative_interval, self.subpopulation_size, self.proposals_per_group) < 1:
            raise ValueError('cooperative settings must be positive')
        self.context_evaluations = 0
        self.context_reconstructions = 0
        self.cooperative_rounds = 0
        self.subpopulations = None
        self.last_groups = []

    def _partition(self, services):
        return [g.tolist() for g in np.array_split(np.arange(services), min(self.groups, services)) if len(g)]

    def _init_subpopulations(self, groups, snapshot, fits):
        pools = []
        for group in groups:
            indices = list(range(len(snapshot)))
            self.corega.rndEVOL.shuffle(indices)
            entries = []
            for idx in indices[:min(self.subpopulation_size, len(indices))]:
                entries.append((copy.deepcopy([snapshot[idx][s] for s in group]), copy.deepcopy(fits[idx])))
            pools.append(entries)
        return pools

    def _repair(self, candidate):
        cloud = self.corega.system.cloudDeviceId
        for row in candidate:
            row[cloud] = 1
        for fog in range(self.corega.system.fogNumber):
            self.corega.notEnoughResourceFogDeviceWithRepair(candidate, fog)

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
        proposals = pop.POPULATION(0)
        proposals.population = []
        origins = []
        for group_id, group in enumerate(groups):
            pool = self.subpopulations[group_id]
            for _ in range(self.proposals_per_group):
                context_idx = self.corega.rndEVOL.randint(0, len(snapshot)-1)
                donor_idx = self.corega.rndEVOL.randint(0, len(pool)-1)
                candidate = copy.deepcopy(snapshot[context_idx])
                for local_i, service in enumerate(group):
                    candidate[service] = copy.deepcopy(pool[donor_idx][0][local_i])
                # Variation is restricted to the active group: mutate full candidate,
                # then restore all inactive services from the frozen context.
                before = copy.deepcopy(candidate)
                self.corega.mutate(candidate)
                active = set(group)
                for service in range(len(candidate)):
                    if service not in active:
                        candidate[service] = before[service]
                self._repair(candidate)
                proposals.population.append(candidate)
                origins.append(group_id)
                self.context_reconstructions += 1
        before_nfe = self.corega.evaluation_count
        self.corega.calculatePopulationFitnessObjectives(proposals)
        self.context_evaluations += self.corega.evaluation_count - before_nfe
        # Each pool evolves independently; only its own proposals enter its pool.
        for group_id, group in enumerate(groups):
            pool = self.subpopulations[group_id]
            candidate_entries = pool + [
                (copy.deepcopy([proposals.population[i][s] for s in group]), copy.deepcopy(proposals.fitness[i]))
                for i, origin in enumerate(origins) if origin == group_id
            ]
            indices = environmental_select_nsga3(
                [fit for _, fit in candidate_entries],
                min(self.subpopulation_size, len(candidate_entries)),
                self.referenceVectors, self.corega.rndEVOL
            )
            self.subpopulations[group_id] = [copy.deepcopy(candidate_entries[i]) for i in indices]
        combined = current.populationUnion(current, proposals)
        indices = environmental_select_nsga3(combined.fitness, self.populationSize, self.referenceVectors, self.corega.rndEVOL)
        self._set_population_from_indices(combined, indices)
        self._update_archive(self.corega.populationPt)
        self.cooperative_rounds += 1

    def step(self, progress=0.0):
        super().step(progress=progress)
        if self._step_count % self.cooperative_interval == 0:
            self._cooperative_round()

    def run_budget(self, search_budget):
        """Complete-step budget; explicitly reports any overshoot."""
        budget = int(search_budget)
        if budget < 1:
            raise ValueError('search_budget must be positive')
        initial = self.corega.evaluation_count
        while self.corega.evaluation_count - initial < budget:
            used = self.corega.evaluation_count - initial
            self.step(progress=used / budget)
        actual = self.corega.evaluation_count - initial
        return {'initial_evaluations': initial, 'search_evaluations': actual,
                'budget': budget, 'overshoot': actual-budget,
                'cooperative_context_evaluations': self.context_evaluations,
                'context_reconstructions': self.context_reconstructions,
                'cooperative_rounds': self.cooperative_rounds, 'steps': self._step_count}
