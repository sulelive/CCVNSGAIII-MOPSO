"""Phase 8: group-local repair with immutable inactive-group context and audited NFE."""
import copy
from q1_cooperative_phase7 import CCVNSGAIII_MOPSO_CPU_P7
from q1_cooperative_phase5 import CCVNSGAIII_MOPSO_CPU_P5
from core.q1_common import environmental_select_nsga3
import POPULATION as pop

class CCVNSGAIII_MOPSO_CPU_P8(CCVNSGAIII_MOPSO_CPU_P7):
    name = 'ccv_nsgaiii_mopso_cpu_phase8'

    def __init__(self, *args, **kwargs):
        self.group_repair_calls = 0
        self.inactive_change_count = 0
        self.active_removed_count = 0
        super().__init__(*args, **kwargs)

    def _repair_group(self, candidate, group, canonical):
        """Repair fog capacity by removing only active-group replicas.

        Since canonical is feasible, inactive-group load alone cannot exceed capacity.
        """
        active = set(group)
        system = self.corega.system
        cloud = system.cloudDeviceId
        for service in range(system.serviceNumber):
            if service not in active and candidate[service] != canonical[service]:
                self.inactive_change_count += 1
                raise AssertionError('inactive group changed before repair')
            candidate[service][cloud] = 1
        for fog in range(system.fogNumber):
            used = sum(system.serviceResources[s] * candidate[s][fog] for s in range(system.serviceNumber))
            removable = [s for s in group if candidate[s][fog]]
            while used > system.fogResources[fog]:
                if not removable:
                    raise AssertionError('canonical context infeasible or inactive capacity exceeded')
                idx = self.corega.rndEVOL.randint(0, len(removable)-1)
                service = removable.pop(idx)
                candidate[service][fog] = 0
                used -= system.serviceResources[service]
                self.active_removed_count += 1
        self.group_repair_calls += 1
        if any(candidate[s] != canonical[s] for s in range(system.serviceNumber) if s not in active):
            self.inactive_change_count += 1
            raise AssertionError('inactive group changed during repair')

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
                self._repair_group(candidate, group, canonical)
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
                self._repair_group(candidate, group, canonical)
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

