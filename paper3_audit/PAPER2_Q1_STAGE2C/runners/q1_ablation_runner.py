# -*- coding: utf-8 -*-
"""Small ablation runner using the same search-evaluation budget policy."""
from __future__ import print_function
import os, sys, csv, time, random, argparse
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,"..")); sys.path.insert(0,ROOT)
import CONFIG as config, SYSTEMMODEL as systemmodel, GAcore
from core.q1_instrumentation import install_evaluation_counter
from core.q1_common import nondominated_indices, save_front_csv
from algorithms.q1_ablation import AblationA_SimplexOnly, AblationB_SimplexAdaptive, AblationC_SimplexOccupancy, AblationProposed
from algorithms.q1_steppable import SteppableOriginalHybrid
install_evaluation_counter(GAcore.GAcore)
ALGS={"original":SteppableOriginalHybrid,"A_simplex":AblationA_SimplexOnly,"B_simplex_adaptive":AblationB_SimplexAdaptive,"C_simplex_occupancy":AblationC_SimplexOccupancy,"proposed":AblationProposed}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--apps",type=int,default=5); ap.add_argument("--population",type=int,default=200); ap.add_argument("--budget",type=int,default=40000); ap.add_argument("--runs",type=int,default=1); ap.add_argument("--topology",default="nxgenerated.100"); ap.add_argument("--out",default=os.path.join(ROOT,"outputs","ablation_pilot")); args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    master=random.Random(20260901); seeds=[(master.randint(1,10**9),master.randint(1,10**9),master.randint(1,10**9)) for _ in range(args.runs)]; rows=[]
    for name,cls in ALGS.items():
      for run_id,s in enumerate(seeds,1):
        c=config.CONFIG(); c.topologyGraphml=args.topology; c.applicationJson=os.path.join(ROOT,"scenario","threeapps.json"); c.userJson=os.path.join(ROOT,"scenario","user.json"); c.numberOfReplicatedApps=args.apps; c.populationSize=args.population; c.numberGenerations=1
        sm=systemmodel.SYSTEMMODEL(c); sm.FGCSmodel3(s[0]); alg=cls(sm,s[1],s[2],c); core=alg.corega; init=int(getattr(core,"evaluation_count",0)); t0=time.perf_counter(); steps=0
        while int(getattr(core,"evaluation_count",0))-init < args.budget:
            search=int(getattr(core,"evaluation_count",0))-init; alg.step(progress=min(1.0,float(search)/max(1,args.budget))); steps+=1
        runtime=time.perf_counter()-t0; search=int(getattr(core,"evaluation_count",0))-init; fit=core.populationPt.fitness; nd=nondominated_indices(fit); d=os.path.join(args.out,"apps_%d"%args.apps,name,"run_%02d"%run_id); save_front_csv(os.path.join(d,"final_pareto.csv"),fit,nd)
        rows.append({"variant":name,"run":run_id,"search_evaluations":search,"overshoot":search-args.budget,"initialization_evaluations":init,"steps":steps,"runtime_seconds":runtime,"nondominated_count":len(nd),"services":sm.serviceNumber})
        print(name,run_id,"evals",search,"overshoot",search-args.budget,"nd",len(nd))
    p=os.path.join(args.out,"ablation_summary_apps_%d.csv"%args.apps); f=open(p,"w",newline=""); w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows); f.close(); print("Summary:",p)
if __name__=="__main__": main()
