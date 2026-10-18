# Paper 3 Experimental Protocol

## Comparison algorithms

The Paper 3 comparison set is:

1. NSGA-III
2. NSGAIII-MOPSO
3. ARV-NSGAIII-MOPSO
4. CCVNSGAIII-MOPSO

MOEA/D and RVEA are not included in the Paper 3 comparison set.

## Large-scale scenarios

The principal Paper 3 scalability scenarios are:

- 200 services
- 500 services
- 1000 services

## Proposed method

The proposed method combines cooperative co-evolution with the
reference-vector NSGA-III/MOPSO lineage.

The final canonical class distributed in this release is:

`CCVNSGAIII_MOPSO_CPU_P8`

## Endpoint rule

If an evaluation budget terminates inside an incomplete algorithmic
step, the result must be reported from the last fully committed
population.

The requested cap and the last committed NFE should be recorded
separately where they differ.

## Reproducibility

Random seeds, population size, objective-evaluation budget, scenario,
algorithm configuration and endpoint accounting should be reported
with every publication-scale experiment.
