"""Three frozen rounds of sampler mechanism search, including a final holdout."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .benchmark_v2 import reference_details, score_result, write_csv, write_json
from .inference_v3 import CONFIGS, infer_v3
from .stress_cases import DEVELOPMENT_CASES, STRESS_CASES, StressCase

HELDOUT_CASES=(
    StressCase('heldout_shift_curve4',dimension=4,shift=.4,curve=.25,rotation_seed=7341,joint_phase=True,noise=.18),
    StressCase('heldout_four_aged8',dimension=8,four_modes=True,shift=-.35,age=2,rotation_seed=7342,noise=.16,transient=True),
    StressCase('heldout_curve16',dimension=16,curve=.7,shift=-.25,rotation_seed=7343,noise=.3,joint_phase=True),
    StressCase('heldout_resolved8',dimension=8,curve=.45,resolved=True,rotation_seed=7344,noise=.18,age=1.5),
    StressCase('heldout_harmless4',dimension=4,harmless=True,curve=.3,shift=.5,rotation_seed=7345,noise=.35,age=4,transient=True),
    StressCase('heldout_narrow16',dimension=16,shift=.3,rotation_seed=7346,noise=.09,age=.75,joint_phase=True),
)

def _hashes(root):
    paths=['src/transcif/research/inference_v3.py','src/transcif/research/sampler_search_v3.py',
           'src/transcif/research/inference_v2.py','src/transcif/research/stress_cases.py',
           'src/transcif/research/benchmark_v2.py','docs/SAMPLER_SEARCH_V3_PROTOCOL.md',
           'scripts/run_sampler_search_v3.py','src/transcif/research/inference.py',
           'src/transcif/research/weighted_metrics.py','src/transcif/research/representations.py',
           'src/transcif/research/gates12.py','src/transcif/research/toy.py',
           'src/transcif/research/__init__.py','src/transcif/__init__.py',
           'tests/test_research_inference_v3.py']
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}

def _selection(rows,n):
    """Frozen equal-case/budget risk: mean CDF + .25 worst cell + .05 failure rate."""
    ranked=[]
    for method in sorted(set(r['method'] for r in rows)-{'sobol','prior'}):
        selected=[r for r in rows if r['method']==method]
        cells={}
        for r in selected: cells.setdefault((r['case'],r['budget']),[]).append(r['cdf_distance'])
        means=[np.mean(x) for x in cells.values()]
        risk=float(np.mean(means)+.25*max(means)+.05*np.mean([r['cdf_distance']>.25 for r in selected]))
        ranked.append({'method':method,'selection_risk':risk})
    ranked.sort(key=lambda x:(x['selection_risk'],x['method']))
    return [x['method'] for x in ranked[:n]],ranked

def run_round(root,output,round_id,cases,seeds,methods):
    if output.exists(): raise ValueError(f'Immutable output already exists: {output}')
    output.mkdir(parents=True)
    protocol={'round':round_id,'selection_status':'final heldout' if round_id==3 else 'development (previously viewed problems)',
              'cases':[x.metadata() for x in cases],'seeds':list(seeds),'budgets':[128,512],
              'methods':[asdict(c) for c in CONFIGS if c.name in methods],
              'selection_rule':'mean cell CDF + .25 worst cell CDF + .05 fraction CDF>.25',
              'sha256':_hashes(root),'frozen_before_first_evaluation':True}
    write_json(output/'protocol.json',protocol)
    for p in protocol['sha256']:
        dst=output/'source_snapshot'/p; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes((root/p).read_bytes())
    rows=[]; diagnostics=[]; audits={}
    for case in cases:
        reference=reference_details(case); audits[case.name]=reference['audit']
        np.savez_compressed(output/f'reference_{case.name}.npz',**{k:reference[k] for k in ('thresholds','cdf','mean','truths','endpoints')})
        for budget in (128,512):
            for seed in seeds:
                for method in methods:
                    count=[0]
                    def evaluate(z):
                        count[0]+=len(z)
                        if count[0]>budget: raise AssertionError('Forward row budget exceeded')
                        return case.evaluate(z)
                    result=infer_v3(evaluate,case.dimension,budget,seed,method)
                    assert result.evaluations==count[0]<=budget
                    row={'round':round_id,'case':case.name,'dimension':case.dimension,'noise':case.noise,
                         'curve':case.curve,'age':case.age,'rotated':case.rotation_seed is not None,
                         'four_modes':case.four_modes,'budget':budget,'seed':seed,'method':method,
                         **score_result(result,reference),'actual_forward_rows':count[0]}
                    row['cdf_failure']=int(row['cdf_distance']>.25)
                    rows.append(row)
                    diagnostics.append({'case':case.name,'budget':budget,'seed':seed,'method':method,**result.diagnostics})
        write_csv(output/'repeats.csv',rows); write_json(output/'diagnostics.json',diagnostics)
        write_json(output/'reference_audit.json',audits)
        print(f'round {round_id}: {case.name} completed; {len(rows)} runs',flush=True)
    aggregate=[]
    for case in cases:
        for budget in (128,512):
            for method in methods:
                cell=[r for r in rows if r['case']==case.name and r['budget']==budget and r['method']==method]
                aggregate.append({'case':case.name,'budget':budget,'method':method,
                    **{k:float(np.mean([r[k] for r in cell])) for k in ('cdf_distance','crps_excess_to_reference','cdf_failure','ess','actual_forward_rows','inference_seconds')}})
    write_csv(output/'aggregate.csv',aggregate)
    comparisons=[]
    for case in cases:
        for budget in (128,512):
            baseline=[r for r in rows if r['case']==case.name and r['budget']==budget and r['method']=='sobol']
            for method in methods:
                if method in ('sobol','prior'): continue
                cand=[r for r in rows if r['case']==case.name and r['budget']==budget and r['method']==method]
                diff=np.array([c['cdf_distance']-b['cdf_distance'] for c,b in zip(cand,baseline)])
                comparisons.append({'case':case.name,'budget':budget,'method':method,'cdf_difference':float(diff.mean()),
                    'ci_low':float(diff.mean()-1.96*diff.std(ddof=1)/np.sqrt(len(diff))),
                    'ci_high':float(diff.mean()+1.96*diff.std(ddof=1)/np.sqrt(len(diff))),
                    'reference_refinement_mean':audits[case.name]['cdf_refinement_mean'],
                    'reference_refinement_max':audits[case.name]['cdf_refinement_max']})
    write_csv(output/'comparisons.csv',comparisons)
    macro=[]
    for budget in (128,512):
        for method in methods:
            selected=[r for r in aggregate if r['budget']==budget and r['method']==method]
            macro.append({'budget':budget,'method':method,**{k:float(np.mean([r[k] for r in selected])) for k in ('cdf_distance','crps_excess_to_reference','cdf_failure','actual_forward_rows')}})
    write_csv(output/'macro.csv',macro)
    assert _hashes(root)==protocol['sha256'],'Code changed during frozen round'
    write_json(output/'summary.json',{'round':round_id,'runs':len(rows),'all_actual_rows_verified':True,'macro':macro,'hashes_unchanged':True})
    return rows

def run(root,output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    r1=run_round(root,output/'round1',1,DEVELOPMENT_CASES,range(6000,6010),[c.name for c in CONFIGS])
    top,ranking=_selection(r1,3); write_json(output/'round1_selection.json',{'chosen':top,'ranking':ranking})
    r2=run_round(root,output/'round2',2,STRESS_CASES,range(7000,7010),['sobol','prior']+top)
    winner,ranking=_selection(r2,1); write_json(output/'round2_selection.json',{'chosen':winner,'ranking':ranking})
    r3=run_round(root,output/'round3',3,HELDOUT_CASES,range(8000,8020),['sobol','prior']+winner)
    write_json(output/'complete.json',{'total_runs':len(r1)+len(r2)+len(r3),'frozen_final_method':winner,
        'round_runs':[len(r1),len(r2),len(r3)],'holdout_not_used_for_selection':True,
        'claim':'Mechanism search only; no novelty or strong-baseline win assumed. Read heldout comparisons and reference refinement.'})
    return winner
