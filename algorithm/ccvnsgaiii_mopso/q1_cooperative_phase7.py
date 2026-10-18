"""Phase 7: single-pass chaotic initialization at the inherited construction boundary.

Constructor-scoped GAcore.generatePopulation replacement; not thread-safe for concurrent
construction in one interpreter. No baseline fairness claim without common checkpoint policy.
"""
import hashlib
import GAcore as gacore
from q1_cooperative_phase6 import CCVNSGAIII_MOPSO_CPU_P6

class CCVNSGAIII_MOPSO_CPU_P7(CCVNSGAIII_MOPSO_CPU_P6):
    name = 'ccv_nsgaiii_mopso_cpu_phase7'
    def __init__(self, system, populationSeed, evolutionSeed, conf_, **kwargs):
        if 'chaos_seed' not in kwargs:
            digest = hashlib.sha256(f'paper3-chaos:{populationSeed}:{evolutionSeed}'.encode()).digest()
            kwargs['chaos_seed'] = (int.from_bytes(digest[:8], 'big') + 0.5) / 2**64
        seed = float(kwargs['chaos_seed'])
        if not 0 < seed < 1:
            raise ValueError('chaos_seed must be strictly between 0 and 1')
        original = gacore.GAcore.generatePopulation
        def single_pass(core, population):
            z = seed
            cloud = core.system.cloudDeviceId
            for individual in range(core.populationSize):
                chromosome = [[0] * core.system.fogNumber for _ in range(core.system.serviceNumber)]
                for row in chromosome:
                    for fog in range(core.system.fogNumber):
                        z = 4.0 * z * (1.0 - z)
                        row[fog] = int(z >= 0.5)
                    row[cloud] = 1
                for fog in range(core.system.fogNumber):
                    core.notEnoughResourceFogDeviceWithRepair(chromosome, fog)
                population.population[individual] = chromosome
            core.calculatePopulationFitnessObjectives(population)
            core._paper3_single_pass_chaos_seed = seed
        gacore.GAcore.generatePopulation = single_pass
        try:
            super().__init__(system, populationSeed, evolutionSeed, conf_, **kwargs)
        finally:
            gacore.GAcore.generatePopulation = original
        if self.initialization_nfe != self.populationSize:
            raise AssertionError('single-pass initialization must cost exactly population size')

    def _chaotic_initialize(self):
        # Stage2C base constructor has already initialized and evaluated the chaotic
        # population once. Preserve its archive and avoid a second evaluation.
        if getattr(self.corega, '_paper3_single_pass_chaos_seed', None) != self.chaos_seed:
            raise AssertionError('single-pass chaotic constructor was not applied')
        self.chaotic_initial_evaluations = self.corega.evaluation_count

    def run_total_budget(self, total_budget):
        result = super().run_total_budget(total_budget)
        assert result['inherited_initialization_nfe'] == 0
        assert result['chaotic_initialization_nfe'] == self.populationSize
        return result
