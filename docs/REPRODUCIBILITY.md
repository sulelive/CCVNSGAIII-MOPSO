# Reproducibility Guide

## 1. Purpose

This repository provides the checksum-verified Phase-30 implementation
of CCVNSGAIII-MOPSO used in the Paper 3 research workflow for
large-scale QoS-aware fog service placement.

## 2. Canonical algorithm

Primary class:

`CCVNSGAIII_MOPSO_CPU_P8`

Primary source:

`algorithm/ccvnsgaiii_mopso/q1_cooperative_phase8.py`

Canonical SHA-256:

`ecf2e2672ced1d8a9a92aafea000d3c464cfbaeb2dcbec45a14b2a62c441a98f`

The canonical algorithm source must not be edited when reproducing the
reported implementation.

## 3. Inheritance chain

The implementation uses the following inheritance chain:

1. CCVNSGAIII_MOPSO_CPU_P8
2. CCVNSGAIII_MOPSO_CPU_P7
3. CCVNSGAIII_MOPSO_CPU_P6
4. CCVNSGAIII_MOPSO_CPU_P5
5. CCVNSGAIII_MOPSO_CPU_P4
6. ADAPTIVE_SELECTIVE_MOPSO_NSGA3_STAGE2C
7. NSGA3_Q1
8. _BaseEA

## 4. Environment

Python 3.13 was used during the audited Colab workflow.

Install the public dependencies with:

`pip install -r requirements.txt`

## 5. Verify installation

Run:

`python examples/verify_import.py`

A successful run prints the canonical class and inheritance chain.

## 6. Run the original Phase-8 validation

From the repository root, configure the module path to include:

- `algorithm/ccvnsgaiii_mopso`
- `paper3_audit/PAPER2_Q1_STAGE2C`

Then execute:

`python tests/test_phase8.py`

The validated test produces deterministic checks for seeds 731001,
731002 and 731003.

## 7. Objective convention

The research implementation uses minimization-oriented objective
vectors internally. Resource utilization may therefore be represented
through an unused-resource objective, with reported resource
utilization obtained from the corresponding transformed quantity.

Users should preserve the objective convention of the frozen
experiment when reproducing published results.

## 8. Evaluation accounting

Publication experiments must use the last fully committed population
when a requested objective-evaluation cap falls inside an incomplete
algorithmic step. An incomplete state must not be promoted to the
reported endpoint.

## 9. Source integrity

`SHA256SUMS.txt` contains repository-level file checksums.

The canonical P8 source checksum is independently documented in
`docs/SOURCE_PROVENANCE.md`.

## 10. Later optimized branches

Later Phase-39 optimization branches are not substituted for the
canonical Phase-30 algorithm in this release.

This distinction is intentional so that later engineering
optimizations are not silently represented as the implementation used
in the earlier audited experimental protocol.
