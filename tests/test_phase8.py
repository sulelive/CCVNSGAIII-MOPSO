import sys, json, math
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'paper3_audit'/'PAPER2_Q1_STAGE2C'
sys.path[:0]=[str(ROOT),str(ROOT/'core'),str(ROOT/'algorithms'),str(HERE)]
import CONFIG, SYSTEMMODEL, GAcore
from core.q1_instrumentation import install_evaluation_counter
from core.q1_common import nondominated_indices,obj_vector
from q1_cooperative_phase8 import CCVNSGAIII_MOPSO_CPU_P8
install_evaluation_counter(GAcore.GAcore)
ORIGINAL=GAcore.GAcore.generatePopulation

def run(seed,budget):
    c=CONFIG.CONFIG(); c.topologyGraphml='nxgenerated.100'; c.applicationJson=str(ROOT/'scenario'/'threeapps.json'); c.userJson=str(ROOT/'scenario'/'user.json'); c.numberOfReplicatedApps=1; c.populationSize=12; c.numberGenerations=1
    m=SYSTEMMODEL.SYSTEMMODEL(c); m.FGCSmodel3(seed)
    a=CCVNSGAIII_MOPSO_CPU_P8(m,seed+1,seed+2,c,groups=4,cooperative_interval=1,subpopulation_size=3,proposals_per_group=2)
    assert GAcore.GAcore.generatePopulation is ORIGINAL
    r=a.run_total_budget(budget)
    assert r['initialization_nfe']==12 and r['inherited_initialization_nfe']==0
    assert r['total_nfe']<=budget and r['total_nfe']==r['initialization_nfe']+r['search_nfe']
    assert r['pool_refresh_nfe']==r['cooperative_rounds']*12
    assert r['context_nfe']==r['cooperative_rounds']*20
    for x,f in zip(a.corega.populationPt.population,a.corega.populationPt.fitness):
        assert all(row[m.cloudDeviceId]==1 for row in x)
        assert all(sum(m.serviceResources[s]*x[s][fog] for s in range(m.serviceNumber))<=m.fogResources[fog] for fog in range(m.fogNumber))
        assert all(math.isfinite(float(v)) for v in obj_vector(f))
    assert a.inactive_change_count == 0
    assert a.group_repair_calls == r['cooperative_rounds'] * (12+8)
    sig=sorted(tuple(round(float(v),12) for v in obj_vector(a.corega.populationPt.fitness[i])) for i in nondominated_indices(a.corega.populationPt.fitness))
    return {'accounting':r,'signature':sig,'group_repair_calls':a.group_repair_calls,'inactive_changes':a.inactive_change_count,'active_removed':a.active_removed_count}

if __name__=='__main__':
    cases=[]
    for seed,budget in [(731001,30),(731002,82),(731003,192)]:
        x=run(seed,budget); y=run(seed,budget); assert x==y
        cases.append(x)
        print('PASS',seed,'budget',budget,'total',x['accounting']['total_nfe'],'unused',x['accounting']['unused_nfe'])
    assert cases[0]['accounting']['chaos_seed']!=cases[1]['accounting']['chaos_seed']
    (HERE/'test_phase8_results.json').write_text(json.dumps({'same_seed_reproducible':True,'cases':cases},indent=2))
    print('PASS: single-pass chaos, no inherited initialization, deterministic signatures and feasibility')
