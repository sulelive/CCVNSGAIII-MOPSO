# -*- coding: utf-8 -*-
"""Stage-1 validation runner for all six Q1 algorithms.

Purpose: smoke-test imports, feasibility, objective orientation, seed reproducibility,
objective-evaluation counting, and Pareto export BEFORE expensive 20-run experiments.
"""
from __future__ import print_function
import os, sys, csv, json, time, copy, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "algorithms"))

import CONFIG as config
import SYSTEMMODEL as systemmodel
import GAcore
from core.q1_instrumentation import install_evaluation_counter
from core.q1_common import nondominated_indices, reporting_row, save_front_csv
from algorithms.q1_comparators import NSGA2_Q1, NSGA3_Q1, MOEAD_Q1, RVEA_Q1
import HYBRID_MOPSO_NSGA3 as original_hybrid
import ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR as adaptive_hybrid

install_evaluation_counter(GAcore.GAcore)

ALGS = {
    "nsga2": NSGA2_Q1,
    "nsga3": NSGA3_Q1,
    "moead": MOEAD_Q1,
    "rvea": RVEA_Q1,
    "original_nsgaiii_mopso": original_hybrid.HYBRID_MOPSO_NSGA3,
    "adaptive_nsgaiii_mopso": adaptive_hybrid.ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR,
}


def build_config(args):
    c = config.CONFIG()
    c.topologyGraphml = args.topology
    c.applicationJson = os.path.join(ROOT, "scenario", "threeapps.json")
    c.userJson = os.path.join(ROOT, "scenario", "user.json")
    c.numberOfReplicatedApps = args.apps
    c.populationSize = args.population
    c.numberGenerations = args.generations
    return c


def run_once(name, args, seeds, tag):
    c = build_config(args)
    model_seed, pop_seed, evol_seed = seeds
    sm = systemmodel.SYSTEMMODEL(c); sm.FGCSmodel3(model_seed)
    cls = ALGS[name]
    alg = cls(sm, pop_seed, evol_seed, c)
    t0 = time.perf_counter(); alg.evolve(); runtime = time.perf_counter() - t0
    core = alg.corega
    fit = core.populationPt.fitness
    nd = nondominated_indices(fit)
    # representative only for validation diagnostics
    rep_idx = min(range(len(fit)), key=lambda i: float(fit[i].get("total", float("inf"))))
    rr = reporting_row(fit[rep_idx])
    outdir = os.path.join(args.out, name, tag)
    save_front_csv(os.path.join(outdir, "pareto.csv"), fit, nd)
    return {
        "algorithm": name, "tag": tag, "services": sm.serviceNumber,
        "fog_nodes": sm.fogNumber, "population": c.populationSize, "generations": c.numberGenerations,
        "model_seed": model_seed, "population_seed": pop_seed, "evolution_seed": evol_seed,
        "objective_evaluations": int(getattr(core, "evaluation_count", -1)),
        "runtime_seconds": runtime, "nondominated_count": len(nd),
        "spread": rr["service_spread"], "communication_latency_proxy": rr["communication_latency_proxy"],
        "resource_utilization": rr["resource_utilization"],
        "signature": [(round(float(fit[i]["spread"]), 12), round(float(fit[i]["makespan"]), 12), round(float(fit[i]["resources"]), 12)) for i in nd]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="nxgenerated.100")
    ap.add_argument("--apps", type=int, default=1, help="1=20 services; 5=100; 10=200")
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "validation"))
    ap.add_argument("--repro-check", action="store_true", help="Run each algorithm twice with same seeds")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = (731001, 731002, 731003)
    rows = []
    for name in ALGS:
        print("[VALIDATE]", name)
        a = run_once(name, args, seeds, "A"); rows.append(a)
        if args.repro_check:
            b = run_once(name, args, seeds, "B"); rows.append(b)
            ok = (a["signature"] == b["signature"])
            print("  reproducible:", ok)
            if not ok: raise RuntimeError("Same-seed reproducibility failed for %s" % name)
        print("  evals=%s nd=%s runtime=%.3fs" % (a["objective_evaluations"], a["nondominated_count"], a["runtime_seconds"]))
    # signatures are too large for summary CSV
    csvrows = []
    for r in rows:
        z = dict(r); z.pop("signature", None); csvrows.append(z)
    path = os.path.join(args.out, "validation_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0].keys())); w.writeheader(); w.writerows(csvrows)
    with open(os.path.join(args.out, "validation_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Validation summary:", path)

if __name__ == "__main__": main()
