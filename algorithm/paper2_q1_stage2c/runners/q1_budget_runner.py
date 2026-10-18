# -*- coding: utf-8 -*-
"""Q1 Stage-2A: matched-seed, objective-evaluation-budget experiment runner.

Budget definition: SEARCH-PHASE objective-function calls after algorithm construction.
Initialization calls are recorded separately and are not charged to the common search budget.
Algorithms stop at the first complete algorithmic step that reaches/exceeds the requested
budget. Exact achieved evaluations and overshoot are always reported.
"""
from __future__ import print_function
import os, sys, csv, json, time, random, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "algorithms"))

import CONFIG as config
import SYSTEMMODEL as systemmodel
import GAcore
from core.q1_instrumentation import install_evaluation_counter
from core.q1_common import nondominated_indices, save_front_csv
from algorithms.q1_steppable import STEPPABLE_ALGORITHMS

install_evaluation_counter(GAcore.GAcore)


def build_config(args):
    c = config.CONFIG()
    c.topologyGraphml = args.topology
    c.applicationJson = os.path.join(ROOT, "scenario", "threeapps.json")
    c.userJson = os.path.join(ROOT, "scenario", "user.json")
    c.numberOfReplicatedApps = args.apps
    c.populationSize = args.population
    # construction only; stepping is controlled by this runner
    c.numberGenerations = 1
    return c


def objective_rows(fitness, nd):
    rows = []
    for i in nd:
        f = fitness[i]
        rows.append((float(f["spread"]), float(f["makespan"]), float(f["resources"])))
    return rows


def write_snapshot(path, fitness, nd, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","spread","communication_latency_proxy","unused_resource_fraction","resource_utilization","total_objective_evaluations","search_evaluations","step"])
        for i in nd:
            x = fitness[i]
            w.writerow([i, x["spread"], x["makespan"], x["resources"], 1.0-float(x["resources"]), metadata["total_evals"], metadata["search_evals"], metadata["step"]])


def run_one(name, args, seeds, run_id):
    c = build_config(args)
    model_seed, pop_seed, evol_seed = seeds
    sm = systemmodel.SYSTEMMODEL(c); sm.FGCSmodel3(model_seed)
    alg = STEPPABLE_ALGORITHMS[name](sm, pop_seed, evol_seed, c)
    core = alg.corega
    initial_evals = int(getattr(core, "evaluation_count", 0))
    start_total = initial_evals
    target = int(args.budget)
    checkpoints = sorted(set([int(x) for x in args.checkpoints if int(x) > 0 and int(x) <= target] + [target]))
    next_cp = 0
    step = 0
    t0 = time.perf_counter()
    rows = []
    alg_dir = os.path.join(args.out, "apps_%d" % args.apps, name, "run_%02d" % run_id)
    while True:
        before = int(getattr(core, "evaluation_count", 0))
        search_before = before - start_total
        if search_before >= target:
            break
        progress = min(1.0, float(search_before) / float(max(1, target)))
        alg.step(progress=progress)
        step += 1
        total_evals = int(getattr(core, "evaluation_count", 0))
        search_evals = total_evals - start_total
        fit = core.populationPt.fitness
        nd = nondominated_indices(fit)
        elapsed = time.perf_counter() - t0
        while next_cp < len(checkpoints) and search_evals >= checkpoints[next_cp]:
            cp = checkpoints[next_cp]
            meta = {"total_evals": total_evals, "search_evals": search_evals, "step": step}
            write_snapshot(os.path.join(alg_dir, "checkpoint_%07d.csv" % cp), fit, nd, meta)
            rows.append({
                "run":run_id, "algorithm":name, "checkpoint_target":cp,
                "search_evaluations":search_evals, "total_objective_evaluations":total_evals,
                "initialization_evaluations":initial_evals, "overshoot":search_evals-cp,
                "step":step, "runtime_seconds":elapsed, "nondominated_count":len(nd),
                "services":sm.serviceNumber, "fog_nodes":sm.fogNumber,
                "model_seed":model_seed, "population_seed":pop_seed, "evolution_seed":evol_seed,
            })
            next_cp += 1
        if search_evals >= target:
            break
    # final front, regardless of checkpoint naming
    fit = core.populationPt.fitness; nd = nondominated_indices(fit)
    save_front_csv(os.path.join(alg_dir, "final_pareto.csv"), fit, nd)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="nxgenerated.100")
    ap.add_argument("--apps", type=int, default=5)
    ap.add_argument("--population", type=int, default=200)
    ap.add_argument("--budget", type=int, default=40000, help="Common SEARCH-PHASE objective-evaluation budget.")
    ap.add_argument("--checkpoints", type=int, nargs="*", default=[5000,10000,20000,40000])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--master-seed", type=int, default=20260901)
    ap.add_argument("--algorithms", nargs="*", default=list(STEPPABLE_ALGORITHMS.keys()))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "budget_pilot"))
    args = ap.parse_args()
    bad = [a for a in args.algorithms if a not in STEPPABLE_ALGORITHMS]
    if bad: raise ValueError("Unknown algorithms: %s" % bad)
    os.makedirs(args.out, exist_ok=True)
    master = random.Random(args.master_seed)
    seeds = [(master.randint(1,10**9), master.randint(1,10**9), master.randint(1,10**9)) for _ in range(args.runs)]
    allrows = []
    for name in args.algorithms:
        for run_id, triplet in enumerate(seeds, 1):
            print("[BUDGET] %s run %d/%d seeds=%s" % (name, run_id, args.runs, triplet))
            rr = run_one(name, args, triplet, run_id); allrows.extend(rr)
            z = rr[-1]
            print("  achieved search evals=%d target=%d overshoot=%d steps=%d nd=%d runtime=%.3fs" % (
                z["search_evaluations"], args.budget, z["search_evaluations"]-args.budget,
                z["step"], z["nondominated_count"], z["runtime_seconds"]))
    path = os.path.join(args.out, "budget_summary_apps_%d.csv" % args.apps)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    with open(os.path.join(args.out, "budget_config_apps_%d.json" % args.apps), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.out, "seeds_apps_%d.json" % args.apps), "w") as f:
        json.dump([{"run":i+1,"model_seed":s[0],"population_seed":s[1],"evolution_seed":s[2]} for i,s in enumerate(seeds)], f, indent=2)
    print("Summary:", path)

if __name__ == "__main__":
    main()
