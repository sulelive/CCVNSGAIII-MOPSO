# -*- coding: utf-8 -*-
"""Stage-2C matched-seed evaluation-budget runner.
Default development comparison: NSGA-III vs Adaptive-V2 vs Stage-2C.
"""
from __future__ import print_function
import os, sys, csv, json, time, random, argparse
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,".."))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"core")); sys.path.insert(0,os.path.join(ROOT,"algorithms"))
import CONFIG as config
import SYSTEMMODEL as systemmodel
import GAcore
from core.q1_instrumentation import install_evaluation_counter
from core.q1_common import nondominated_indices, save_front_csv
from algorithms.q1_steppable import STEPPABLE_ALGORITHMS
from algorithms.q1_adaptive_v2 import ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR_V2
from algorithms.q1_stage2c import ADAPTIVE_SELECTIVE_MOPSO_NSGA3_STAGE2C
install_evaluation_counter(GAcore.GAcore)
ALGORITHMS=dict(STEPPABLE_ALGORITHMS)
ALGORITHMS["adaptive_nsgaiii_mopso_v2"]=ADAPTIVE_HYBRID_MOPSO_NSGA3_VECTOR_V2
ALGORITHMS["adaptive_selective_mopso_stage2c"]=ADAPTIVE_SELECTIVE_MOPSO_NSGA3_STAGE2C

def build_config(args):
    c=config.CONFIG(); c.topologyGraphml=args.topology
    c.applicationJson=os.path.join(ROOT,"scenario","threeapps.json"); c.userJson=os.path.join(ROOT,"scenario","user.json")
    c.numberOfReplicatedApps=args.apps; c.populationSize=args.population; c.numberGenerations=1
    return c

def write_snapshot(path, fitness, nd, md):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["index","spread","communication_latency_proxy","unused_resource_fraction","resource_utilization","total_objective_evaluations","search_evaluations","step"])
        for i in nd:
            x=fitness[i]; w.writerow([i,x["spread"],x["makespan"],x["resources"],1.0-float(x["resources"]),md["total_evals"],md["search_evals"],md["step"]])

def run_one(name,args,seeds,run_id):
    c=build_config(args); ms,ps,es=seeds
    sm=systemmodel.SYSTEMMODEL(c); sm.FGCSmodel3(ms)
    alg=ALGORITHMS[name](sm,ps,es,c)
    if name=="adaptive_selective_mopso_stage2c":
        alg.refinement_interval=args.refinement_interval; alg.refinement_fraction=args.refinement_fraction
        alg.adaptation_strength=args.alpha; alg.anchor_weight=args.anchor_weight
        alg.adaptation_warmup=args.adaptation_warmup; alg.position_memory=args.position_memory
    core=alg.corega; init=int(getattr(core,"evaluation_count",0)); start=init; target=int(args.budget)
    cps=sorted(set([int(x) for x in args.checkpoints if 0<int(x)<=target]+[target])); next_cp=0; step=0; rows=[]; t0=time.perf_counter()
    algdir=os.path.join(args.out,"apps_%d"%args.apps,name,"run_%02d"%run_id)
    while True:
        search_before=int(getattr(core,"evaluation_count",0))-start
        if search_before>=target: break
        progress=min(1.0,float(search_before)/float(max(1,target)))
        alg.step(progress=progress); step+=1
        total=int(getattr(core,"evaluation_count",0)); search=total-start
        fit=core.populationPt.fitness; nd=nondominated_indices(fit); elapsed=time.perf_counter()-t0
        while next_cp<len(cps) and search>=cps[next_cp]:
            cp=cps[next_cp]; write_snapshot(os.path.join(algdir,"checkpoint_%07d.csv"%cp),fit,nd,{"total_evals":total,"search_evals":search,"step":step})
            rows.append({"run":run_id,"algorithm":name,"checkpoint_target":cp,"search_evaluations":search,"total_objective_evaluations":total,"initialization_evaluations":init,"overshoot":search-cp,"step":step,"runtime_seconds":elapsed,"nondominated_count":len(nd),"services":sm.serviceNumber,"fog_nodes":sm.fogNumber,"model_seed":ms,"population_seed":ps,"evolution_seed":es})
            next_cp+=1
        if search>=target: break
    fit=core.populationPt.fitness; nd=nondominated_indices(fit); save_front_csv(os.path.join(algdir,"final_pareto.csv"),fit,nd)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--topology",default="nxgenerated.100"); ap.add_argument("--apps",type=int,default=5); ap.add_argument("--population",type=int,default=200)
    ap.add_argument("--budget",type=int,default=10000); ap.add_argument("--checkpoints",type=int,nargs="*",default=[2500,5000,7500,10000]); ap.add_argument("--runs",type=int,default=1); ap.add_argument("--master-seed",type=int,default=20260915)
    ap.add_argument("--algorithms",nargs="*",default=["nsga3","adaptive_nsgaiii_mopso_v2","adaptive_selective_mopso_stage2c"]); ap.add_argument("--out",default=os.path.join(ROOT,"outputs","stage2c_diagnostic10k"))
    ap.add_argument("--refinement-interval",type=int,default=5); ap.add_argument("--refinement-fraction",type=float,default=0.20); ap.add_argument("--alpha",type=float,default=0.20); ap.add_argument("--anchor-weight",type=float,default=0.50); ap.add_argument("--adaptation-warmup",type=float,default=0.10); ap.add_argument("--position-memory",type=float,default=2.0)
    args=ap.parse_args(); bad=[a for a in args.algorithms if a not in ALGORITHMS]
    if bad: raise ValueError("Unknown algorithms: %s"%bad)
    os.makedirs(args.out,exist_ok=True); master=random.Random(args.master_seed); seeds=[(master.randint(1,10**9),master.randint(1,10**9),master.randint(1,10**9)) for _ in range(args.runs)]; allrows=[]
    for name in args.algorithms:
        for rid,s in enumerate(seeds,1):
            print("[STAGE2C] %s run %d/%d seeds=%s"%(name,rid,args.runs,s)); rr=run_one(name,args,s,rid); allrows.extend(rr); z=rr[-1]
            print("  evals=%d target=%d overshoot=%d steps=%d nd=%d runtime=%.3fs"%(z["search_evaluations"],args.budget,z["search_evaluations"]-args.budget,z["step"],z["nondominated_count"],z["runtime_seconds"]))
    path=os.path.join(args.out,"stage2c_summary_apps_%d.csv"%args.apps)
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    with open(os.path.join(args.out,"stage2c_config_apps_%d.json"%args.apps),"w") as f: json.dump(vars(args),f,indent=2)
    with open(os.path.join(args.out,"stage2c_seeds_apps_%d.json"%args.apps),"w") as f: json.dump([{"run":i+1,"model_seed":s[0],"population_seed":s[1],"evolution_seed":s[2]} for i,s in enumerate(seeds)],f,indent=2)
    print("Summary:",path)
if __name__=="__main__": main()
