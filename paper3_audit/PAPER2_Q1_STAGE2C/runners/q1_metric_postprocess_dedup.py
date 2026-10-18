# -*- coding: utf-8 -*-
"""Stage-2B metric postprocessor with objective-vector deduplication.

Deduplication is applied BEFORE normalization, pooled reference-front construction,
HV, IGD+, and Spacing. Raw nondominated counts remain available in runner summaries.
"""
from __future__ import print_function
import os,sys,csv,glob,argparse
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,"..")); sys.path.insert(0,ROOT)
from core.q1_common import nondominated_indices
from core.q1_metrics import normalize_with_bounds,hypervolume_3d_min,igd_plus,spacing


def read_front(path):
    rows=[]
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            rows.append([float(r["spread"]),float(r["communication_latency_proxy"]),float(r["unused_resource_fraction"])])
    return np.asarray(rows,dtype=float)


def unique_rows(F):
    if len(F)==0: return F
    # 12 decimals protects against textual round-trip noise without merging meaningful points.
    _,idx=np.unique(np.round(F,12),axis=0,return_index=True)
    return F[np.sort(idx)]


def nd_points(F):
    F=unique_rows(F)
    fits=[{"spread":x[0],"makespan":x[1],"resources":x[2]} for x in F]
    return F[nondominated_indices(fits)]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--apps",type=int,default=5); ap.add_argument("--ref",type=float,default=1.10)
    args=ap.parse_args(); base=os.path.join(args.root,"apps_%d"%args.apps)
    paths=glob.glob(os.path.join(base,"*","run_*","checkpoint_*.csv"))
    if not paths: raise RuntimeError("No checkpoint files under %s"%base)
    bycp={}; fronts={}; rawcounts={}
    for p in paths:
        cp=int(os.path.basename(p).split("_")[1].split(".")[0]); bycp.setdefault(cp,[]).append(p)
        raw=read_front(p); rawcounts[p]=len(raw); fronts[p]=unique_rows(raw)
    all_pool=np.vstack([F for F in fronts.values() if len(F)]); lo=np.min(all_pool,axis=0); hi=np.max(all_pool,axis=0)
    final_cp=max(bycp); final_pool=np.vstack([fronts[p] for p in bycp[final_cp] if len(fronts[p])])
    ref_front=nd_points(normalize_with_bounds(final_pool,lo,hi))
    out=[]
    for cp,plist in sorted(bycp.items()):
        for p in plist:
            F=fronts[p]; Fn=normalize_with_bounds(F,lo,hi); alg=p.split(os.sep)[-3]; run=os.path.basename(os.path.dirname(p))
            out.append({"checkpoint":cp,"algorithm":alg,"run":run,"hv":hypervolume_3d_min(Fn,(args.ref,args.ref,args.ref)),
                        "igd_plus":igd_plus(Fn,ref_front),"spacing":spacing(Fn),"raw_nondominated_count":rawcounts[p],
                        "unique_nondominated_count":len(F),"normalization_lo_spread":lo[0],"normalization_lo_latency":lo[1],
                        "normalization_lo_unused":lo[2],"normalization_hi_spread":hi[0],"normalization_hi_latency":hi[1],
                        "normalization_hi_unused":hi[2],"reference_front_size":len(ref_front),"final_reference_checkpoint":final_cp,"hv_reference":args.ref})
    path=os.path.join(args.root,"metrics_dedup_apps_%d.csv"%args.apps)
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("Metrics:",path); print("Common normalization lo:",lo); print("Common normalization hi:",hi); print("Final empirical reference front size:",len(ref_front),"at checkpoint",final_cp)

if __name__=="__main__": main()
