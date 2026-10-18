# CCVNSGAIII-MOPSO

**Cooperative Co-evolutionary NSGA-III-MOPSO for Large-Scale
QoS-Aware Fog Service Placement**

CCVNSGAIII-MOPSO is a many-objective optimization framework developed
for large-scale service-placement problems in fog computing.

The method combines cooperative co-evolution, NSGA-III reference-vector
environmental selection and MOPSO-based search within the adaptive
NSGAIII-MOPSO algorithmic lineage.

## Canonical implementation

The primary implementation distributed in this repository is:

`CCVNSGAIII_MOPSO_CPU_P8`

Source:

`algorithm/ccvnsgaiii_mopso/q1_cooperative_phase8.py`

Canonical SHA-256:

`ecf2e2672ced1d8a9a92aafea000d3c464cfbaeb2dcbec45a14b2a62c441a98f`

The source has been preserved byte-for-byte from the audited Phase-30
implementation.

## Algorithm lineage

The final implementation inherits through:

`P8 -> P7 -> P6 -> P5 -> P4 -> ADAPTIVE_SELECTIVE_MOPSO_NSGA3_STAGE2C -> NSGA3_Q1`

The required inherited implementation is therefore included with the
release.

## Main research objectives

The fog-placement formulation evaluates three principal QoS/resource
dimensions:

- latency;
- service spread; and
- resource utilization.

The implementation internally uses a minimization-oriented objective
representation where required.

## Repository structure

    CCVNSGAIII-MOPSO/
    |
    +-- algorithm/
    |   +-- ccvnsgaiii_mopso/
    |   |   +-- q1_cooperative_phase4.py
    |   |   +-- q1_cooperative_phase5.py
    |   |   +-- q1_cooperative_phase6.py
    |   |   +-- q1_cooperative_phase7.py
    |   |   +-- q1_cooperative_phase8.py
    |   |
    |   +-- paper2_q1_stage2c/
    |
    +-- paper3_audit/
    |   +-- PAPER2_Q1_STAGE2C/
    |
    +-- examples/
    |   +-- verify_import.py
    |
    +-- tests/
    |   +-- test_phase8.py
    |
    +-- docs/
    |   +-- SOURCE_PROVENANCE.md
    |   +-- REPRODUCIBILITY.md
    |   +-- EXPERIMENT_PROTOCOL.md
    |   +-- PHASE8_CPU_REPORT.md
    |
    +-- runtime.py
    +-- requirements.txt
    +-- CITATION.cff
    +-- LICENSE
    +-- SHA256SUMS.txt

## Installation

Clone or download the repository.

Create a Python environment and install the dependencies:

    pip install -r requirements.txt

## Quick verification

From the repository root:

    python examples/verify_import.py

The command should report:

    CCVNSGAIII-MOPSO successfully imported.

and display the full inheritance chain.

## Phase-8 validation

The repository includes the original Phase-8 validation script and
scenario data.

The final release ZIP is validated after extraction in an isolated
temporary directory before publication.

## Paper 3 comparison scope

The intended Paper 3 comparison set is:

- NSGA-III
- NSGAIII-MOPSO
- ARV-NSGAIII-MOPSO
- CCVNSGAIII-MOPSO

MOEA/D and RVEA are intentionally excluded from the Paper 3
experimental comparison.

## Reproducibility and integrity

See:

- `docs/REPRODUCIBILITY.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/SOURCE_PROVENANCE.md`
- `SHA256SUMS.txt`

The canonical Phase-8 source should retain its documented SHA-256
checksum.

## Citation

Citation metadata is provided in `CITATION.cff`.

When the associated Paper 3 article receives its final journal
citation or DOI, the repository citation metadata should be updated
accordingly.

## License

This repository is released under the MIT License. See `LICENSE`.

## Research integrity note

Later Phase-39 optimization branches are not substituted for the
canonical Phase-30 implementation in this release. This prevents later
engineering changes from being silently represented as the exact
implementation used in the audited Phase-30 workflow.
