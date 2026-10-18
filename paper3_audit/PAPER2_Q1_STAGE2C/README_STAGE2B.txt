PAPER2 Q1 — STAGE 2B: ADAPTIVE V2 DEVELOPMENT CANDIDATE
========================================================

Purpose
-------
Stage 2A showed that Adaptive-V1 was not competitive with NSGA-II, NSGA-III, or RVEA
under a common objective-evaluation budget. Stage 2B therefore develops and validates a
scientifically corrected V2 while preserving Original and V1 unchanged.

V2 corrections
--------------
1. Standard weak Pareto dominance for internal sorting, archive, and pBest logic.
2. Ideal-nadir normalization before reference-vector association and adaptation.
3. Archive leaders sampled from sparsely occupied reference regions.
4. Pareto-aware pBest update; mutually nondominated ties use the inherited equal-weight
   normalized scalar score as a convergence tie-break.
5. Controlled vector adaptation: default 10% warm-up, adaptation every 2 V2 steps,
   alpha=0.35, anchor weight=0.50.
6. Parameterized occupancy score:
      Q = lambda/(1+d) + (1-lambda)/(1+rho)
   with lambda=0.50 by default.
7. Objective-vector deduplication before HV/IGD+/Spacing/reference-front construction.

IMPORTANT
---------
V2 is a DEVELOPMENT CANDIDATE. Do not use its current results in the manuscript as
confirmatory evidence until it passes the new-seed validation gate and the algorithm is frozen.
Do not overwrite Original or Adaptive-V1 files.

Colab diagnostic (known development seed)
-----------------------------------------
python runners/q1_stage2b_runner.py \
  --mode diagnostic \
  --apps 5 --population 200 --budget 40000 \
  --checkpoints 5000 10000 20000 40000 \
  --runs 1 --master-seed 20260901 \
  --out outputs/stage2b_diagnostic

python runners/q1_metric_postprocess_dedup.py \
  --root outputs/stage2b_diagnostic --apps 5 --ref 1.10

Acceptance gate A (diagnostic)
------------------------------
V2 should show a clear improving HV/IGD+ trajectory and materially improve over V1.
This seed is for debugging only, not parameter selection for the final paper.

Five NEW matched-seed validation
--------------------------------
Use a different master seed from the historical diagnostic, e.g. 20260914:

python runners/q1_stage2b_runner.py \
  --mode validation \
  --apps 5 --population 200 --budget 40000 \
  --checkpoints 5000 10000 20000 40000 \
  --runs 5 --master-seed 20260914 \
  --out outputs/stage2b_validation5

python runners/q1_metric_postprocess_dedup.py \
  --root outputs/stage2b_validation5 --apps 5 --ref 1.10

Acceptance gate B (new seeds)
-----------------------------
Before freezing V2:
- clear improvement over V1 should first be demonstrated;
- V2 should exhibit sustained convergence rather than a flat trajectory;
- competitiveness against NSGA-II/NSGA-III/RVEA should be evaluated across the five
  unseen matched seeds, not judged from the diagnostic seed;
- runtime and unique nondominated-set size must remain practical.

Only after passing this gate should the 20-run confirmatory Q1 study begin.
