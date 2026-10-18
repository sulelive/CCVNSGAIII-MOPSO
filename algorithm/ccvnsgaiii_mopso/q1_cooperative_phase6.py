"""Phase 6: run-seeded chaos and all-inclusive NFE accounting.

This is a conservative extension: initialization overhead is INCLUDED, not hidden.
No partial optimization step or unmeasured fitness evaluation is performed.
"""
import hashlib
from q1_cooperative_phase5 import CCVNSGAIII_MOPSO_CPU_P5

class CCVNSGAIII_MOPSO_CPU_P6(CCVNSGAIII_MOPSO_CPU_P5):
    name = 'ccv_nsgaiii_mopso_cpu_phase6'

    def __init__(self, system, populationSeed, evolutionSeed, conf_, **kwargs):
        if 'chaos_seed' not in kwargs:
            digest = hashlib.sha256(f'paper3-chaos:{populationSeed}:{evolutionSeed}'.encode()).digest()
            # Strictly inside (0,1); reproducible and independent of global RNG.
            kwargs['chaos_seed'] = (int.from_bytes(digest[:8], 'big') + 0.5) / 2**64
        super().__init__(system, populationSeed, evolutionSeed, conf_, **kwargs)
        self.initialization_nfe = self.corega.evaluation_count
        if self.initialization_nfe < self.chaotic_initial_evaluations:
            raise AssertionError('initialization counter inconsistency')

    def run_total_budget(self, total_budget):
        """Enforce a hard cap on ALL objective calls, including inherited + chaotic initialization."""
        total_budget = int(total_budget)
        if total_budget < self.initialization_nfe:
            raise ValueError(f'total budget {total_budget} is below initialization cost {self.initialization_nfe}')
        search_allowance = total_budget - self.initialization_nfe
        start = self.corega.evaluation_count
        while self.corega.evaluation_count + self.next_step_cost() <= total_budget:
            expected = self.next_step_cost()
            before = self.corega.evaluation_count
            self.step(progress=(before-start)/max(1, search_allowance))
            if self.corega.evaluation_count - before != expected:
                raise AssertionError("preflight NFE mismatch")
            actual = self.corega.evaluation_count - start
            if actual < 0 or actual > search_allowance:
                raise AssertionError('total evaluation budget exceeded')
        search_nfe = self.corega.evaluation_count - start
        assert self.corega.evaluation_count == self.initialization_nfe + search_nfe
        return dict(total_budget=total_budget, total_nfe=self.corega.evaluation_count,
                    initialization_nfe=self.initialization_nfe,
                    inherited_initialization_nfe=self.initialization_nfe-self.chaotic_initial_evaluations,
                    chaotic_initialization_nfe=self.chaotic_initial_evaluations,
                    search_nfe=search_nfe, unused_nfe=total_budget-self.corega.evaluation_count,
                    overshoot=0, steps=self._step_count, cooperative_rounds=self.cooperative_rounds,
                    context_nfe=self.context_evaluations, pool_refresh_nfe=self.pool_refresh_evaluations,
                    chaos_seed=self.chaos_seed, next_step_cost=self.next_step_cost())
