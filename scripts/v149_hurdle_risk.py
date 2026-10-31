"""Source-only occurrence/severity controls for monthly risk-curve calibration."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from strict_tie_quantile import StrictTieQuantileRegressor, audit_gradient_contract
from critical_composition_response import DELTAS
from v139_physical_reserve_calibration import sha
from v142_pace_lgee_monthly_holdout import record
from v146_critical_composition_calibration import summarize

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'results/v147-conditional-curve-scale-r1'
C = ROOT / 'results/v146-critical-composition-calibration-r1'
BASE = ROOT / 'results/v145-recomposed-risk-baselines-r1'
OUT = ROOT / 'results/v149-hurdle-risk-r1'
PROTOCOL = ROOT / 'docs/V149_HURDLE_RISK_PROTOCOL.md'
FAMILIES = ('horizontal_composition_joint', 'vertical_hull_joint')
MODELS = ('ridge_all', 'hurdle_mean', 'positive_severity', 'strict_q90')

def weights(regions):
    counts = pd.Series(regions).value_counts()
    w = np.asarray([1 / counts[r] for r in regions])
    return w / w.mean()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit_gradient_contract()
    paths = [PROTOCOL, Path(__file__), ROOT/'scripts/strict_tie_quantile.py',
        ROOT/'scripts/v146_critical_composition_calibration.py', ROOT/'scripts/v142_pace_lgee_monthly_holdout.py',
        P/'summary.json', P/'fit_inverse_labels.csv', P/'scale_features.csv', P/'region_roles.csv',
        C/'summary.json', C/'curves.npz', C/'curve_index.csv', C/'calibration_inverse_scores.csv',
        C/'target_rows.csv', BASE/'target_rows.csv', BASE/'summary.json',
        ROOT/'results/v144-canonical-profile-cache-r1/canonical_monthly_panel.csv']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for directory in (P, C):
        summary = json.loads((directory/'summary.json').read_text())
        if summary['assertion_failures']:
            raise ValueError('upstream audit failed')
        for name, digest in summary['output_hashes'].items():
            if sha(directory/name) != digest:
                raise ValueError(f'upstream output changed: {name}')
    labels = pd.read_csv(P/'fit_inverse_labels.csv')
    features = pd.read_csv(P/'scale_features.csv', float_precision='round_trip')
    roles = pd.read_csv(P/'region_roles.csv')
    inverse = pd.read_csv(C/'calibration_inverse_scores.csv')
    ci = pd.read_csv(C/'curve_index.csv')
    curve_file = np.load(C/'curves.npz', allow_pickle=False)
    if list(curve_file['methods']) != list(FAMILIES) or not np.array_equal(curve_file['deltas'], DELTAS):
        raise ValueError('curve convention changed')
    curves = curve_file['curves']
    panel = pd.read_csv(paths[-1])
    monthly = panel.loc[panel.tau.eq(600)].set_index('profile_id')
    columns = json.loads((P/'summary.json').read_text())['feature_columns']
    if any('factor_box' in x or 'floor' in x or 'nominal_K' in x for x in columns):
        raise ValueError('forbidden model feature')
    outputs, diagnostics, probabilities, scales, radii, scores_saved = [], [], [], [], [], []
    for target in ('PACE', 'LGEE', 'SC'):
        for seed in (41, 83):
            roster = roles.loc[roles.target_region.eq(target) & roles.seed.eq(seed)]
            fit = set(roster.loc[roster.role.eq('fit'), 'region'])
            cal = set(roster.loc[roster.role.eq('calibration'), 'region'])
            if len(fit)!=30 or len(cal)!=12 or fit & cal or target in fit|cal:
                raise ValueError('region role violation')
            x = features.loc[features.target_region.eq(target) & features.seed.eq(seed)].set_index('profile_id')[columns]
            if not x.index.is_unique or not np.isfinite(x.to_numpy()).all():
                raise ValueError('invalid feature table')
            fold_index = ci.loc[ci.target_region.eq(target) & ci.seed.eq(seed)].set_index('profile_id')
            query_ids = fold_index.index.to_numpy(int)
            for fi, family in enumerate(FAMILIES):
                train = labels.loc[labels.target_region.eq(target) & labels.seed.eq(seed) & labels.family.eq(family)].set_index('profile_id')
                if set(train.index) != set(monthly.loc[monthly.region.isin(fit)].index) or not train.supported.all():
                    raise ValueError('fit label support/denominator mismatch')
                calibration = inverse.loc[inverse.target_region.eq(target) & inverse.seed.eq(seed) & inverse.method.eq(family)].set_index('profile_id')
                if set(calibration.index) != set(monthly.loc[monthly.region.isin(cal)].index):
                    raise ValueError('calibration label denominator mismatch')
                w = weights(train.region.to_numpy())
                scaler = StandardScaler().fit(x.loc[train.index], sample_weight=w)
                tx, qx = scaler.transform(x.loc[train.index]), scaler.transform(x.loc[query_ids])
                delta = train.training_delta.to_numpy(float)
                event = delta > 0
                prevalence = float(np.average(event, weights=w))
                if event.all() or not event.any():
                    probability = np.full(len(query_ids), prevalence)
                else:
                    classifier = LogisticRegression(C=1, max_iter=2000)
                    classifier.fit(tx, event, sample_weight=w)
                    probability = classifier.predict_proba(qx)[:,1]
                if event.any():
                    positive = Ridge(alpha=10).fit(tx[event], np.log1p(256*delta[event]),
                        sample_weight=weights(train.region.to_numpy()[event]))
                    severity = np.maximum(0, np.expm1(np.clip(positive.predict(qx), -30, 20))/256)
                else:
                    severity = np.zeros(len(query_ids))
                ordinary = Ridge(alpha=10).fit(tx, np.log1p(256*delta), sample_weight=w)
                qmodel = StrictTieQuantileRegressor(loss='quantile', alpha=.9, n_estimators=150,
                    learning_rate=.05, max_depth=2, min_samples_leaf=15, min_impurity_decrease=1e-10, random_state=20261028)
                qmodel.fit(x.loc[train.index], delta, sample_weight=w)
                predicted = {'ridge_all':np.maximum(0,np.expm1(np.clip(ordinary.predict(qx),-30,20))/256),
                    'hurdle_mean':probability*severity, 'positive_severity':severity,
                    'strict_q90':qmodel.predict(x.loc[query_ids])}
                p = pd.Series(probability,index=query_ids)
                ycal = calibration.grid_index.gt(0).to_numpy()
                cp = p.loc[calibration.index].to_numpy()
                cw = weights(calibration.region.to_numpy())
                brier = float(np.average((cp-ycal)**2,weights=cw))
                constant_brier = float(np.average((prevalence-ycal)**2,weights=cw))
                diagnostics.append({'target_region':target,'seed':seed,'family':family,'fit_months':len(train),
                    'positive_fit_months':int(event.sum()),'calibration_months':len(calibration),'positive_calibration_months':int(ycal.sum()),
                    'fit_prevalence':prevalence,'calibration_prevalence':float(np.average(ycal,weights=cw)),
                    'brier':brier,'constant_brier':constant_brier,
                    'average_precision':float(average_precision_score(ycal,cp,sample_weight=cw)) if ycal.any() else None})
                for j,qid in enumerate(query_ids):
                    probabilities.append({'target_region':target,'seed':seed,'family':family,'profile_id':int(qid),
                        'role':fold_index.loc[qid,'role'],'probability':float(probability[j]),'positive_severity':float(severity[j])})
                for model_name, raw_scale in predicted.items():
                    method = f'{"horizontal" if fi==0 else "vertical"}_{model_name}'
                    scale = pd.Series(np.clip(raw_scale,1/256,1), index=query_ids)
                    cal_scale = scale.loc[calibration.index].to_numpy()
                    score = np.where(calibration.supported,calibration.delta.to_numpy()/cal_scale,np.inf)
                    pool = pd.DataFrame({'region':calibration.region.to_numpy(),'score':score}).groupby('region').score.max()
                    if set(pool.index)!=cal:
                        raise ValueError('calibration region score mismatch')
                    for qid,value in scale.items():
                        scales.append({'target_region':target,'seed':seed,'method':method,'profile_id':int(qid),'scale':float(value)})
                    for j,item in enumerate(calibration.itertuples()):
                        scores_saved.append({'target_region':target,'seed':seed,'method':method,'profile_id':int(item.Index),
                            'region':item.region,'scale':float(cal_scale[j]),'score':float(score[j]) if np.isfinite(score[j]) else None,
                            'supported':bool(item.supported)})
                    for alpha in (.1,.2):
                        rank = math.ceil(13*(1-alpha))
                        radius = float(pool.sort_values().iloc[rank-1])
                        radii.append({'target_region':target,'seed':seed,'method':method,'alpha':alpha,'rank':rank,
                            'radius':radius if np.isfinite(radius) else None,'infinite_radius':not np.isfinite(radius)})
                        for row in panel.loc[panel.region.eq(target)].itertuples(index=False):
                            product = radius*scale.loc[row.profile_id]
                            fallback = not np.isfinite(product) or product>1
                            if fallback:
                                lo,hi=int(row.physical_lower),int(row.physical_upper)
                            else:
                                grid = min(int(np.searchsorted(DELTAS,product,side='left')),256)
                                a,b=curves[int(fold_index.loc[row.profile_id,'curve_id']),fi,(600,900).index(row.tau),grid]
                                lo=max(int(row.physical_lower),math.floor(a*row.hours+1e-9))
                                hi=min(int(row.physical_upper),math.ceil(b*row.hours-1e-9))
                            if not row.physical_lower<=lo<=hi<=row.physical_upper:
                                raise ValueError('physical containment failure')
                            outputs.append({**record(row,seed,method,alpha,lo,hi),'physical_fallback':fallback,
                                'factor_floor':row.factor_box_K_upper-row.factor_box_K_lower,'calibration_scope':'joint_two_thresholds'})
            print(f'completed hurdle controls {target}/{seed}',flush=True)
    rows=pd.DataFrame(outputs)
    keys=['region','month','tau','seed','method','alpha']
    if len(rows)!=66*2*2*2*8 or rows.duplicated(keys).any():
        raise ValueError('target denominator failure')
    for key,g in rows.groupby(['region','tau','seed','method','alpha']):
        if len(g)!=(18 if key[0]=='PACE' else 24):raise ValueError('target cell denominator')
    for _,g in rows.groupby(keys[:-1]):
        wide,narrow=g.sort_values('alpha').itertuples(index=False)
        if wide.lower>narrow.lower or wide.upper<narrow.upper:raise ValueError('alpha nesting')
    archived=pd.read_csv(BASE/'target_rows.csv').assign(calibration_scope='per_threshold_archived_v145')
    combined=pd.concat([rows,pd.read_csv(C/'target_rows.csv'),archived],ignore_index=True)
    frontier,joint=summarize(combined)
    directional=rows.assign(lower_covered=rows.lower.le(rows.factor_box_K_lower),upper_covered=rows.upper.ge(rows.factor_box_K_upper)).groupby(
        ['region','seed','tau','method','alpha']).agg(lower_coverage=('lower_covered','mean'),upper_coverage=('upper_covered','mean')).reset_index()
    if any(sha(ROOT/name)!=digest for name,digest in hashes.items()):raise ValueError('frozen input/code changed')
    OUT.mkdir(parents=True)
    for name,frame in [('target_rows',rows),('comparison_frontier',frontier),('joint_coverage',joint),('individual_coverage',directional),
        ('occurrence_diagnostics',pd.DataFrame(diagnostics)),('probabilities',pd.DataFrame(probabilities)),('scales',pd.DataFrame(scales)),
        ('calibration_scores',pd.DataFrame(scores_saved)),('calibration_radii',pd.DataFrame(radii)),('region_roles',roles)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    summary={'status':'SOURCE_OCCURRENCE_SEVERITY_DEVELOPMENT','scored_rows':len(rows),'assertion_failures':0,
        'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in sorted(OUT.iterdir())},
        'limitations':['Opened development targets; monthly tail-count proxy endpoints, not hourly forecasts.',
            'Hurdle/Ridge/strict quantile are baseline controls, not original algorithms.',
            'Fit teachers exclude each own region; 29-to-30 teacher change remains an approximation.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(pd.DataFrame(diagnostics).to_string(index=False))
    print('assertion failures: 0')

if __name__=='__main__':main()
