PAPER 2 Q1 — STAGE 2A
=====================
Purpose
-------
Control the principal six-algorithm comparison by a common SEARCH-PHASE objective-evaluation budget, while preserving the historical original NSGAIII-MOPSO implementation and the supplied adaptive extension.

Key methodological decision
---------------------------
The common budget excludes constructor/initialization re-evaluations, which are recorded separately. Search-phase calls to GAcore.calculateFitnessObjectives are counted exactly. Each algorithm stops after the first COMPLETE algorithmic step that reaches or exceeds the target. The achieved count and overshoot are saved, so budget compliance is transparent.

Why complete steps?
-------------------
Stopping halfway through a generation would leave some algorithms in an invalid algorithmic state. At large budgets, the one-step overshoot is small relative to the total budget and is explicitly reported.

Primary pilot command (100 services)
------------------------------------
python runners/q1_budget_runner.py --apps 5 --population 200 --budget 40000 --checkpoints 5000 10000 20000 40000 --runs 1 --out outputs/budget_pilot

Then compute indicators:
python runners/q1_metric_postprocess.py --root outputs/budget_pilot --apps 5 --ref 1.10

Ablation pilot:
python runners/q1_ablation_runner.py --apps 5 --population 200 --budget 40000 --runs 1 --out outputs/ablation_pilot

Smoke test first (recommended in Colab):
python runners/q1_budget_runner.py --apps 1 --population 8 --budget 80 --checkpoints 40 80 --runs 1 --out outputs/stage2a_smoke

Interpretation of objectives
----------------------------
Internal minimization coordinates:
  spread       = Service Spread (lower better)
  makespan     = Communication Latency Proxy (lower better)
  resources    = unused-resource fraction (lower better)
Manuscript Resource Utilization = 1 - resources (higher better).

Ablation variants
-----------------
Original: historical NSGAIII-MOPSO baseline (separate class)
A: simplex vectors; no adaptation; closeness-only score
B: simplex + adaptive movement; closeness-only score
C: simplex + occupancy-aware score; no adaptive movement
Proposed: simplex + adaptive movement + occupancy-aware score

Important
---------
Do not start the 20-run 100/200/400/800 study until the 100-service single-run budget pilot has been inspected for budget overshoot, runtime, finite objectives, Pareto fronts, and metric postprocessing.
