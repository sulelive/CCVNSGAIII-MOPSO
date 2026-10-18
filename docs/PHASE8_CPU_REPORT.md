# Phase 8 — cooperative group isolation and repair audit

## Implemented
- Group-local capacity repair removes only active-group fog replicas; inactive service rows are checked against the frozen canonical context before and after repair.
- Mandatory cloud assignments are maintained. A feasible canonical context guarantees that inactive-group load alone does not exceed capacity.
- Both historical pool refresh and proposal reconstruction use the group-local repair; the normal inherited global repair remains available for initialization and non-cooperative operations.
- Instrumentation records group-repair calls, active replicas removed, and inactive changes (zero required).
- Preserves Phase 7 single-pass chaotic initialization and all-inclusive complete-step budget cap.

## CPU smoke test results
| Seed | Budget | Actual NFE | Unused | Group repairs | Inactive changes |
|---|---:|---:|---:|---:|---:|
| 731001 | 30 | 12 | 18 | 0 | 0 |
| 731002 | 82 | 76 | 6 | 20 | 0 |
| 731003 | 192 | 174 | 18 | 100 | 0 |

All cases were repeated with identical same-seed nondominated signatures. The tests also assert capacity feasibility, mandatory cloud assignment, and evaluation accounting. This is a small CPU correctness smoke test, not a performance comparison.

## Important limitations
- The current runner uses complete optimization steps and leaves unused budget; it does **not** yet provide exact equal-NFE checkpoints across the proposed method and all baselines.
- Pool refresh uses a common canonical context and evaluates complete solutions. This does not prove broad performance benefit.
- The inherited construction-time population generator replacement is temporarily global and not safe for concurrent algorithm construction in one Python interpreter. Run isolated processes.
- No 20-run comparative study, 500/1000-service scalability run, or GPU benchmark has been performed.

## Next stage
Implement a shared evaluation-ledger/checkpoint runner for all algorithms and an exact-budget partial-step strategy (or explicitly compare reachable common checkpoints). Audit all algorithms' initialization, archive, and refinement calls before computing metrics.
