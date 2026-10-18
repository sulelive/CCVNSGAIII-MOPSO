PAPER 2 Q1 — STAGE 2C
Adaptive Reference-Vector Guided Selective MOPSO Refinement

Purpose
-------
Stage 2B showed that population-wide hybrid MOPSO updates stagnated in HV/IGD+.
Stage 2C therefore makes NSGA-III the primary optimizer and invokes MOPSO only as
periodic selective exploitation in sparsely occupied adaptive-reference regions.

Preserved provenance
--------------------
Original, Adaptive V1, and Adaptive V2 files are retained unchanged.
New candidate: adaptive_selective_mopso_stage2c

Stage-2C mechanism
------------------
1. Execute a complete NSGA-III generation.
2. Adapt reference vectors after warm-up using the current nondominated geometry,
   regularized toward the original simplex anchors.
3. Perform standard NSGA-III reference-direction environmental selection.
4. Maintain a deduplicated external nondominated archive.
5. Every refinement_interval generations, select refinement_fraction of the
   population from sparsely occupied reference regions.
6. Apply one archive-guided binary MOPSO refinement to those solutions only.
7. Merge refinements with the NSGA-III population and accept them only through
   reference-direction environmental selection.

Important binary-PSO correction
--------------------------------
The historical sigmoid rule gives P(bit=1)=0.5 when velocity=0, which can randomize
an entire placement during a PSO move. Stage 2C uses a current-bit memory logit plus
the social velocity. At zero social displacement, the current bit is retained with
high probability instead of becoming a coin flip. This is a deliberate Stage-2C
architectural correction and must be documented if retained in the final method.

Default development settings (not yet frozen manuscript parameters)
--------------------------------------------------------------------
refinement interval = 5 generations
refinement fraction = 0.20
adaptation strength = 0.20
anchor weight = 0.50
adaptation warm-up = 0.10 of NFE budget
position-memory logit = 2.0

First gate
----------
100 services, population 200, one matched seed, 10,000 search evaluations.
Compare only NSGA-III, Adaptive V2, and Stage 2C.
If Stage 2C remains near V2 quality, reject/rework before expensive runs.
If it approaches NSGA-III with healthy HV/IGD+ convergence, extend to 40,000 NFE,
then five unseen matched seeds before freezing.

Metrics
-------
Use runners/q1_metric_postprocess_dedup.py. It deduplicates objective vectors before
HV, IGD+, spacing, and pooled empirical reference-front construction.
