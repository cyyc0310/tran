#!/usr/bin/env python3
"""Source-only LOO selection of donor neighborhood size for V194 CIF shapes."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v194_china_scenario_shape_transfer as v194

OUT = ROOT / "results/china-projected-cif-v195-adaptive-donors-r1"
K_CANDIDATES = (3, 5, 10, 15, 20, 29)
FIXED_K = K_CANDIDATES
METHODS = ["constant_anchor", "all30_mean_shape", "all30_shape_law"] + [f"fixed{k}_{kind}" for k in FIXED_K for kind in ("mean", "law")] + ["adaptive_mean", "adaptive_law"]


def loo_select(donors):
    """Select k by equal-weight leave-one-donor-out source shape MAE."""
    means = np.asarray([d[2] for d in donors])
    shapes = np.stack([d[3] for d in donors])
    risk = {k: [] for k in K_CANDIDATES}
    for i in range(len(donors)):
        keep = np.arange(len(donors)) != i
        peer_means = means[keep]
        peer_shapes = shapes[keep]
        order = np.argsort(np.abs(peer_means - means[i]), kind="stable")
        ordered = peer_shapes[order]
        sums = np.cumsum(ordered, axis=0)
        for k in K_CANDIDATES:
            pseudo_pred = sums[k - 1] / k
            risk[k].append(float(np.mean(np.abs(pseudo_pred - shapes[i]))))
    risk_mean = {k: float(np.mean(values)) for k, values in risk.items()}
    selected = min(K_CANDIDATES, key=lambda k: (risk_mean[k], k))
    return selected, risk_mean


def predict(target_mu, donors, selected_k):
    shapes = np.stack([d[3] for d in donors])
    means = np.asarray([d[2] for d in donors])
    order = np.argsort(np.abs(means - target_mu), kind="stable")
    result = {"constant_anchor": np.full(shapes.shape[1], target_mu), "all30_mean_shape": target_mu + shapes.mean(axis=0), "all30_shape_law": target_mu + shapes}
    for k in FIXED_K:
        part = shapes[order[:k]]
        result[f"fixed{k}_mean"] = target_mu + part.mean(axis=0)
        result[f"fixed{k}_law"] = target_mu + part
    part = shapes[order[:selected_k]]
    result["adaptive_mean"] = target_mu + part.mean(axis=0)
    result["adaptive_law"] = target_mu + part
    return result


def crps_empirical(particles, y):
    x = np.sort(np.asarray(particles, dtype=np.float64), axis=0)
    first = np.mean(np.abs(x - y), axis=0)
    n = x.shape[0]
    weights = (2 * np.arange(1, n + 1) - n - 1).reshape(-1, 1)
    return float(np.mean(first - np.sum(weights * x, axis=0) / (n * n)))


def digest(arr):
    return hashlib.sha256(np.asarray(arr, dtype="<f8", order="C").tobytes()).hexdigest()


def score_law(particles, y):
    lo, hi = np.quantile(particles, [0.05, 0.95], axis=0, method="linear")
    width = hi - lo
    outside = np.maximum(lo-y, 0) + np.maximum(y-hi, 0)
    return {"crps": crps_empirical(particles, y), "coverage90": float(np.mean((y >= lo) & (y <= hi))), "width90": float(np.mean(width)), "winkler90": float(np.mean(width + 20 * outside))}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    data = v194.read_data()
    tracks = ["same_scenario", "leave_scenario_out"]
    manifest_path = OUT / "prediction_manifest.jsonl"
    mhash = hashlib.sha256()
    n_queries = 0
    with manifest_path.open("w") as f:
        for track in tracks:
            for scenario in v194.SCENARIOS:
                for year in v194.YEARS:
                    vals, bounds = data[(scenario, year)]
                    for target_idx in range(1, len(v194.EXPECTED_PROVINCES)):
                        province = v194.EXPECTED_PROVINCES[target_idx]
                        for month_idx, (lo, hi) in enumerate(bounds):
                            mu = float(vals[lo:hi, target_idx].mean())
                            donors = v194.make_donors(data, track, scenario, year, target_idx, bounds, month_idx, mu)
                            selected, risks = loo_select(donors)
                            preds = predict(mu, donors, selected)
                            row = {"track": track, "scenario": scenario, "year": year, "province": province, "month": month_idx+1, "selected_k": selected, "source_loo_mae": risks, "donor_sha256": digest(np.stack([d[3] for d in donors])), "predictions_sha256": {m: digest(preds[m]) for m in METHODS}}
                            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                            f.write(line); mhash.update(line.encode()); n_queries += 1
                    print(f"sealed {track}/{scenario}/{year}; {n_queries} queries", flush=True)
    freeze = {"status":"SEALED_BEFORE_SCORING", "protocol":"V195_SOURCE_CV_ADAPTIVE_DONORS_PROTOCOL.md", "protocol_sha256":hashlib.sha256((ROOT/'docs/V195_SOURCE_CV_ADAPTIVE_DONORS_PROTOCOL.md').read_bytes()).hexdigest(), "v194_protocol_sha256":hashlib.sha256((ROOT/'docs/V194_CHINA_SCENARIO_SHAPE_TRANSFER_PROTOCOL.md').read_bytes()).hexdigest(), "v194_amendment_sha256":hashlib.sha256((ROOT/'docs/V194_DATA_COMPLETENESS_AMENDMENT.md').read_bytes()).hexdigest(), "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "candidate_k":K_CANDIDATES, "query_count":n_queries, "manifest_sha256":mhash.hexdigest(), "scope":"simulated China CIF outputs; retrospective oracle monthly means; no real-hour validation"}
    # Use the already audited input receipt digests, preserving their Figshare MD5 provenance.
    freeze["workbook_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(v194.DATA.glob("CEF data for Scenario *.xlsx"))}
    (OUT/"prediction_seal.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2)+"\n")
    print("V195 predictions sealed; scoring starts", flush=True)

    reader=manifest_path.open(); rows=[]
    for track in tracks:
        for scenario in v194.SCENARIOS:
            for year in v194.YEARS:
                vals,bounds=data[(scenario,year)]
                for target_idx in range(1,len(v194.EXPECTED_PROVINCES)):
                    province=v194.EXPECTED_PROVINCES[target_idx]
                    for month_idx,(lo,hi) in enumerate(bounds):
                        mu=float(vals[lo:hi,target_idx].mean())
                        donors=v194.make_donors(data,track,scenario,year,target_idx,bounds,month_idx,mu)
                        selected,risks=loo_select(donors);preds=predict(mu,donors,selected)
                        sealed=json.loads(next(reader))
                        if sealed["selected_k"]!=selected or sealed["source_loo_mae"]!={str(k):risks[k] for k in K_CANDIDATES}:
                            raise AssertionError("LOO selection differs from seal")
                        if sealed["donor_sha256"]!=digest(np.stack([d[3] for d in donors])) or sealed["predictions_sha256"]!={m:digest(preds[m]) for m in METHODS}:
                            raise AssertionError("prediction differs from seal")
                        y=vals[lo:hi,target_idx].copy()
                        perm=y[np.random.default_rng(194_700_000+target_idx*100+month_idx).permutation(y.size)]
                        if not np.isclose(float(perm.mean()),mu,atol=1e-12,rtol=0): raise AssertionError("permuted target mean changed")
                        if any(not np.array_equal(preds[m],predict(mu,donors,selected)[m]) for m in METHODS): raise AssertionError("label permutation isolation failed")
                        for method,pred in preds.items():
                            point=pred.mean(axis=0) if pred.ndim==2 else pred
                            e={"track":track,"scenario":scenario,"year":year,"province":province,"month":month_idx+1,"method":method,"selected_k":selected,"source_loo_mae_at_selected_k":risks[selected],"level_mae":abs(float(point.mean())-mu),"shape_mae":float(np.mean(np.abs(point-mu-(y-mu)))),"shape_rmse":float(np.sqrt(np.mean((point-y)**2))),"hour_mae":float(np.mean(np.abs(point-y))),"hour_rmse":float(np.sqrt(np.mean((point-y)**2))),"crps":None,"coverage90":None,"width90":None,"winkler90":None}
                            if pred.ndim==2:e.update(score_law(pred,y))
                            rows.append(e)
                print(f"scored {track}/{scenario}/{year}",flush=True)
    if reader.readline():raise AssertionError("extra manifest rows")
    reader.close()
    import pandas as pd
    df=pd.DataFrame(rows);df.to_csv(OUT/"monthly_scores.csv",index=False)
    metrics=["level_mae","shape_mae","shape_rmse","hour_mae","hour_rmse","crps","coverage90","width90","winkler90"]
    df.groupby(["track","method"],dropna=False)[metrics].mean().reset_index().to_csv(OUT/"summary_macro.csv",index=False)
    df.groupby(["track","scenario","method"],dropna=False)[metrics].mean().reset_index().to_csv(OUT/"summary_scenario.csv",index=False)
    df[df.method.str.startswith("adaptive")].groupby(["track","selected_k"]).size().rename("queries").reset_index().to_csv(OUT/"selected_k_counts.csv",index=False)
    print("V195 complete",OUT,flush=True)
    print(pd.read_csv(OUT/"summary_macro.csv").to_string(index=False),flush=True)

if __name__=="__main__":main()
