PAPER 2 Q1 REVISION — STAGE 1 VALIDATION FRAMEWORK
===================================================

Purpose
-------
This package is a pre-flight validation framework. It must pass before the expensive
Q1 experiment matrix is launched.

Algorithms included
-------------------
1. NSGA-II (standard weak-Pareto + crowding environmental selection)
2. NSGA-III (simplex reference directions + reference-direction niching)
3. MOEA/D (Tchebycheff decomposition + neighborhood replacement)
4. RVEA (reference-vector association + angle-penalized distance + periodic vector adaptation)
5. Historical Original NSGAIII-MOPSO (preserved source, unchanged)
6. Proposed Adaptive Reference-Vector Guided NSGAIII-MOPSO (preserved source, unchanged)

Important methodological policy
-------------------------------
- Historical baseline/proposed source files are preserved unchanged.
- New standalone comparators use standard weak Pareto dominance in their own selection logic.
- Final Pareto extraction for every algorithm uses standard weak Pareto dominance.
- Internal objective `resources` is the unused-resource fraction; manuscript-facing
  Resource Utilization is reported as 1 - resources.
- Internal `makespan` is reported as Communication Latency Proxy.
- Exact objective-evaluation calls are counted by instrumentation.

Colab folder setup
------------------
Upload/copy the entire PAPER2_Q1_STAGE1 folder into Google Drive, then:

%cd /content/drive/MyDrive/PAPER2_Q1_STAGE1

Quick import/smoke validation (20 services, pop 12, 1 generation):
python runners/q1_validation_runner.py --apps 1 --population 12 --generations 1

Same-seed reproducibility validation:
python runners/q1_validation_runner.py --apps 1 --population 12 --generations 1 --repro-check

After these pass, run a 100-service pilot:
python runners/q1_validation_runner.py --apps 5 --population 200 --generations 2 --repro-check

Do NOT launch 20-run × 100/200/400/800 experiments until the validation summary has
been reviewed, particularly objective-evaluation counts and same-seed reproducibility.
