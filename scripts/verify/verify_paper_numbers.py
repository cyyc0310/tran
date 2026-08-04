"""Cross-check every number in docs/paper/2026-07-26-zeroshot-config-cif-paper.md
(including the TransCIF-ZS+ test-time-calibration variant) against results/*.json.
Prints PASS/FAIL per claim."""
import json
import numpy as np

R = "results/"

def load(name):
    with open(R + name) as f:
        return json.load(f)

checks = []

def check(label, paper_val, computed, tol=0.011):
    if isinstance(paper_val, str) or isinstance(computed, str):
        ok = str(paper_val) == str(computed)
    else:
        ok = abs(float(paper_val) - float(computed)) <= tol * max(1.0, abs(float(paper_val)))
    checks.append((ok, label, paper_val, computed))

# ---------------- Table 1 (unified_eval_full) ----------------
ue = load("unified_eval_full.json")
rows = ue["results"] if isinstance(ue, dict) and "results" in ue else ue
print("unified_eval keys sample:", list(rows[0].keys()) if isinstance(rows, list) else list(rows.keys()))

# aggregate over seeds per region
from collections import defaultdict
grouped = defaultdict(list)
for r in rows:
    grouped[r["target"]].append(r)
by_region = {}
for k, rs_list in grouped.items():
    by_region[k] = {
        "mean_rs": float(np.mean([r["mean_rs"] for r in rs_list])),
        "persist_mae": float(np.mean([r["persistence"]["mae"] for r in rs_list])),
        "patchtst_mae": float(np.mean([r["patchtst_sup"]["mae"] for r in rs_list])),
        "zs_mae_mean": float(np.mean([r["transcif_zs"]["mae"] for r in rs_list])),
        "zs_mae_std": float(np.std([r["transcif_zs"]["mae"] for r in rs_list])),
        "zp_mae_mean": float(np.mean([r["transcif_zs_plus"]["mae"] for r in rs_list])),
        "zp_mae_std": float(np.std([r["transcif_zs_plus"]["mae"] for r in rs_list])),
        "n_seeds": len(rs_list),
    }

paper_table1 = {
    # region: (rs, persist, patchtst, tc_mean, tc_std, rho, rho_p,
    #          zp_mean, zp_std, rho_plus, rho_p_plus)
    "US_FPL": (0.06, 13.4, 13.9, 35.7, 5.7, 2.58, 2.67, 12.9, 0.0, 0.93, 0.96),
    "US_PJM": (0.07, 15.6, 12.8, 39.8, 6.6, 3.11, 2.55, 14.1, 0.0, 1.10, 0.90),
    "US_ISNE": (0.14, 16.0, 13.0, 38.4, 3.3, 2.96, 2.41, 15.4, 0.0, 1.18, 0.96),
    "US_MISO": (0.18, 55.6, 41.2, 54.7, 2.3, 1.33, 0.98, 46.7, 0.1, 1.13, 0.84),
    "QLD1": (0.18, 29.1, 30.3, 51.1, 10.1, 1.69, 1.75, 27.0, 0.1, 0.89, 0.93),
    "US_NYIS": (0.26, 14.6, 12.0, 16.8, 1.0, 1.40, 1.15, 13.5, 0.0, 1.13, 0.93),
    "UK_14_SE_England": (0.28, 50.7, 47.5, 58.7, 0.2, 1.24, 1.16, 50.4, 0.0, 1.06, 1.00),
    "UK_07_South_Wales": (0.29, 76.2, 74.1, 73.0, 1.1, 0.99, 0.96, 71.6, 0.4, 0.97, 0.94),
    "NSW1": (0.29, 53.7, 46.3, 49.5, 0.7, 1.07, 0.92, 46.4, 0.3, 1.00, 0.86),
    "UK_12_South_England": (0.31, 57.4, 53.1, 65.0, 0.5, 1.22, 1.13, 54.9, 0.1, 1.03, 0.96),
    "UK_11_SW_England": (0.32, 53.0, 51.2, 53.3, 1.6, 1.04, 1.01, 49.4, 0.1, 0.96, 0.93),
    "US_ERCO": (0.32, 64.8, 48.1, 59.1, 0.2, 1.23, 0.91, 59.8, 0.2, 1.24, 0.92),
    "VIC1": (0.34, 116.8, 91.6, 104.1, 2.4, 1.14, 0.89, 98.2, 0.3, 1.07, 0.84),
    "US_CISO": (0.40, 27.4, 26.5, 40.5, 1.9, 1.53, 1.48, 25.3, 0.0, 0.95, 0.92),
    "UK_09_E_Midlands": (0.42, 91.8, 78.2, 96.6, 0.9, 1.24, 1.05, 84.4, 0.4, 1.08, 0.92),
    "UK_13_London": (0.42, 52.7, 52.7, 60.4, 0.1, 1.15, 1.14, 52.5, 0.2, 1.00, 0.99),
    "UK_17_Wales": (0.46, 73.0, 65.9, 69.5, 0.4, 1.06, 0.95, 67.9, 0.3, 1.03, 0.93),
    "UK_15_England": (0.52, 48.6, 44.0, 50.6, 0.6, 1.15, 1.04, 47.5, 0.2, 1.08, 0.98),
    "UK_18_GB": (0.56, 46.0, 41.2, 47.7, 0.7, 1.16, 1.04, 44.8, 0.2, 1.09, 0.97),
    "UK_08_W_Midlands": (0.58, 81.5, 66.9, 81.2, 1.3, 1.21, 1.00, 77.5, 0.1, 1.16, 0.95),
    "UK_10_E_England": (0.60, 61.1, 50.6, 58.3, 0.7, 1.15, 0.95, 56.9, 0.3, 1.12, 0.93),
    "UK_05_Yorkshire": (0.60, 50.0, 39.4, 47.2, 0.4, 1.20, 0.94, 45.8, 0.3, 1.16, 0.92),
    "SA1": (0.69, 68.1, 51.6, 64.3, 0.4, 1.25, 0.95, 60.5, 0.1, 1.17, 0.89),
    "UK_06_NW_Merseyside": (0.78, 54.4, 43.9, 51.8, 0.8, 1.18, 0.95, 51.5, 0.0, 1.17, 0.95),
    "US_BPAT": (0.78, 6.3, 6.0, 8.4, 0.1, 1.41, 1.34, 6.1, 0.0, 1.03, 0.98),
    "UK_03_NW_England": (0.80, 29.3, 21.5, 29.2, 0.6, 1.35, 1.00, 27.8, 0.1, 1.29, 0.95),
    "UK_16_Scotland": (0.88, 35.3, 28.2, 38.5, 1.4, 1.36, 1.09, 34.0, 0.2, 1.21, 0.96),
    "UK_02_S_Scotland": (0.90, 22.1, 17.9, 22.7, 0.8, 1.26, 1.03, 21.3, 0.1, 1.19, 0.97),
    "UK_01_N_Scotland": (0.91, 35.0, 32.8, 44.1, 2.2, 1.34, 1.26, 34.4, 0.3, 1.05, 0.98),
}

def find_region(key):
    for k in by_region:
        if k == key or k.startswith(key.split("_")[0] + "_" + key.split("_")[1] if key.startswith("UK") else key):
            if key.startswith("UK"):
                # match UK_NN prefix
                if k[:5] == key[:5]:
                    return by_region[k]
            else:
                return by_region[k]
    return by_region.get(key)

rhos, rhops = [], []
rhos_plus, rhops_plus = [], []
for reg, (rs, per, pt, tcm, tcs, rho, rhop, zpm, zps, rho_pl, rhop_pl) in paper_table1.items():
    d = find_region(reg)
    if d is None:
        checks.append((False, f"T1 {reg}: region missing in JSON", "-", "-"))
        continue
    # discover keys once
    if reg == "US_FPL":
        print("region entry keys:", list(d.keys()))
    per_j = d.get("persist_mae") or d.get("persistence_mae") or d.get("persistence")
    pt_j = d.get("patchtst_mae") or d.get("patchtst") or d.get("supervised_mae")
    tc_j = d.get("zs_mae_mean") or d.get("zero_shot_mae") or d.get("zs_mean") or d.get("transcif_mae")
    tc_s = d.get("zs_mae_std") or d.get("zs_std")
    rs_j = d.get("mean_rs") or d.get("rs_mean")
    if per_j: check(f"T1 {reg} persist", per, per_j, 0.05)
    if pt_j: check(f"T1 {reg} PatchTST", pt, pt_j, 0.05)
    if tc_j:
        check(f"T1 {reg} TransCIF", tcm, tc_j, 0.05)
        if pt_j: check(f"T1 {reg} rho", rho, tc_j / pt_j, 0.02)
        if per_j: check(f"T1 {reg} rho_P", rhop, tc_j / per_j, 0.02)
        rhos.append(tc_j / pt_j if pt_j else None)
        rhops.append(tc_j / per_j if per_j else None)
    if tc_s is not None: check(f"T1 {reg} std", tcs, tc_s, 0.15)
    if rs_j: check(f"T1 {reg} mean_rs", rs, rs_j, 0.05)
    zp_j = d.get("zp_mae_mean")
    if zp_j:
        check(f"T1 {reg} ZS+", zpm, zp_j, 0.05)
        if pt_j: check(f"T1 {reg} rho+", rho_pl, zp_j / pt_j, 0.02)
        if per_j: check(f"T1 {reg} rho_P+", rhop_pl, zp_j / per_j, 0.02)
        rhos_plus.append(zp_j / pt_j if pt_j else None)
        rhops_plus.append(zp_j / per_j if per_j else None)
    zp_s = d.get("zp_mae_std")
    if zp_s is not None: check(f"T1 {reg} ZS+ std", zps, zp_s, 0.15)

rhos = [r for r in rhos if r]
rhops = [r for r in rhops if r]
if rhos:
    check("T1 median rho = 1.24", 1.24, float(np.median(rhos)), 0.01)
    check("T1 mean rho = 1.41", 1.41, float(np.mean(rhos)), 0.01)
    check("T1 median rho_P = 1.04", 1.04, float(np.median(rhops)), 0.01)
    check("T1 wins vs persistence = 12/29", 12, int(sum(1 for r in rhops if r < 1.0)), 0)
    check("T1 within 1.25x = 17/29", 17, int(sum(1 for r in rhos if r <= 1.25)), 0)
    check("T1 within 1.5x = 24/29", 24, int(sum(1 for r in rhos if r <= 1.5)), 0)
    check("T1 matched +-5% additional = 4", 4, int(sum(1 for r in rhops if 1.0 <= r <= 1.05)), 0)
rhos_plus = [r for r in rhos_plus if r]
rhops_plus = [r for r in rhops_plus if r]
if rhos_plus:
    check("T1 ZS+ median rho+ = 1.08", 1.08, float(np.median(rhos_plus)), 0.01)
    check("T1 ZS+ mean rho+ = 1.09", 1.09, float(np.mean(rhos_plus)), 0.01)
    check("T1 ZS+ median rho_P+ = 0.94", 0.94, float(np.median(rhops_plus)), 0.01)
    check("T1 ZS+ wins vs persistence = 29/29", 29, int(sum(1 for r in rhops_plus if r < 1.0)), 0)
    check("T1 ZS+ wins vs PatchTST = 6/29", 6, int(sum(1 for r in rhos_plus if r < 1.0)), 0)
    check("T1 ZS+ within 1.25x = 28/29", 28, int(sum(1 for r in rhos_plus if r <= 1.25)), 0)
    check("T1 ZS+ within 1.5x = 29/29", 29, int(sum(1 for r in rhos_plus if r <= 1.5)), 0)
    check("T1 ZS+ worst rho+ = 1.29", 1.29, float(max(rhos_plus)), 0.01)
    check("T1 ZS+ worst rho_P+ = 0.995", 0.995, float(max(rhops_plus)), 0.01)

# ---------------- Theorem 1 ----------------
def peek(name):
    d = load(name)
    print(f"\n===== {name} =====")
    if isinstance(d, list):
        print(f"list of {len(d)}; first entry:")
        print(json.dumps(d[0], indent=1)[:1200])
    else:
        print("keys:", list(d.keys()))
        print(json.dumps({k: v for k, v in d.items() if not isinstance(v, list)}, indent=1)[:1800])
    return d

t1 = peek("theorem1_validation.json")
t2 = peek("theorem2_transfer_bound.json")
ab = peek("ablation_full.json")
cf = peek("conformal_prediction.json")
to = peek("temporal_ood.json")
dp = peek("deployment_warmup.json")
cc = peek("carboncast_analysis.json")

# ---------------- Theorem 1 checks ----------------
check("Thm1 identity max residual 1.3e-4", 1.3e-4, max(r["identity_max_residual"] for r in t1), 0.05)
fracs = [r["term1_fraction"] for r in t1]
check("Thm1 term1 fraction mean 71.3%", 0.713, float(np.mean(fracs)))
check("Thm1 term1 fraction min 36%", 0.36, float(min(fracs)), 0.03)
check("Thm1 term1 fraction max 97.6%", 0.976, float(max(fracs)), 0.01)
lt_eps = np.array([r["L_T"] * r["mean_rs_error_abs"] for r in t1])
cif_mae = np.array([r["mean_cif_error_abs"] for r in t1])
corr = float(np.corrcoef(lt_eps, cif_mae)[0, 1])
check("Thm1 corr(LT*eps, MAE)=0.914", 0.914, corr)
check("Thm1 R2=0.835", 0.835, corr ** 2)
lts = {r["region"]: r["L_T"] for r in t1}
check("Thm1 L_T VIC1 = 1160", 1160, lts.get("VIC1", 0), 0.01)
check("Thm1 min L_T = 208", 208, min(lts.values()), 0.01)
check("Thm1 5.6x ratio", 5.6, 1160 / 208, 0.01)

# ---------------- Theorem 2 checks ----------------
st = t2["statistics"]
check("Thm2 corr min_dist -0.17", -0.17, st["corr_min_dist"], 0.03)
check("Thm2 p min_dist 0.37", 0.37, st["p_min_dist"], 0.02)
check("Thm2 corr centroid 0.29", 0.29, st["corr_centroid_dist"], 0.02)
check("Thm2 corr density -0.07", -0.07, st["corr_density"], 0.05)
check("Thm2 corr effective 0.58", 0.58, st["corr_effective_dist"], 0.01)
check("Thm2 p effective 0.001", 0.001, st["p_effective_dist"], 0.05)
check("Thm2 quadratic R2 0.662", 0.662, st["r2_quadratic_rs"], 0.01)
a, b, _ = st["quadratic_coeffs"]
check("Thm2 vertex 0.582", 0.582, -b / (2 * a), 0.01)

# ---------------- Ablation checks ----------------
ab_groups = defaultdict(list)
for r in ab:
    ab_groups[r["config"]].append(r["mae"])
print("\nablation configs:", {k: round(float(np.mean(v)), 2) for k, v in ab_groups.items()})
ab_mean = {k: float(np.mean(v)) for k, v in ab_groups.items()}
full = ab_mean.get("Full model")
check("Abl Full 53.6", 53.6, full, 0.01)
paper_abl = {  # substring match: (paper MAE, paper delta %)
    "weighted": (63.5, 18.5), "gate": (58.7, 9.7), "bias": (58.5, 9.3),
    "decomp": (55.0, 2.7), "irect": (47.5, -11.3),
}
for key, (pm, pd) in paper_abl.items():
    match = [k for k in ab_mean if key.lower() in k.lower() and "zs" not in k.lower()]
    if not match:
        checks.append((False, f"Abl config '{key}' not found", pm, list(ab_mean.keys())))
        continue
    m = ab_mean[match[0]]
    check(f"Abl {match[0]} MAE {pm}", pm, m, 0.01)
    check(f"Abl {match[0]} delta {pd}%", pd, (m - full) / full * 100, 0.05)

# ZS+ calibration ablation (paper Table 2b: 9 regions x 5 seeds;
# deltas are per-region means vs. full ZS+, aggregate MAE is mean over regions)
zs_reg = defaultdict(lambda: defaultdict(list))
for r in ab:
    if r["config"] in ("ZS+ (full calib)", "ZS+ w/o anchor", "ZS+ w/o residual",
                       "ZS+ w/o self-val", "ZS+ w/o rolling selection",
                       "ZS+ legacy 2-branch fusion", "Raw ZS (no calib)"):
        zs_reg[r["config"]][r["target"]].append(r["mae"])
zs_rm = {c: {t: float(np.mean(v)) for t, v in d.items()} for c, d in zs_reg.items()}
if zs_rm:
    zp_full = zs_rm["ZS+ (full calib)"]
    paper_zs_abl = {  # config: (paper MAE, paper delta %)
        "ZS+ (full calib)": (44.9, 0.0),
        "ZS+ w/o self-val": (49.1, 12.7),
        "ZS+ w/o rolling selection": (45.5, 1.0),
        "ZS+ legacy 2-branch fusion": (48.7, 6.3),
        "ZS+ w/o residual": (45.0, 0.8),
        "ZS+ w/o anchor": (45.1, 0.6),
        "Raw ZS (no calib)": (53.4, 39.8),
    }
    for cfg, (pm, pd) in paper_zs_abl.items():
        if cfg not in zs_rm:
            checks.append((False, f"Abl config '{cfg}' not found", pm, list(zs_rm.keys())))
            continue
        agg = float(np.mean(list(zs_rm[cfg].values())))
        delta = float(np.mean([(zs_rm[cfg][t] - zp_full[t]) / zp_full[t] for t in zp_full])) * 100
        check(f"Abl {cfg} MAE {pm}", pm, agg, 0.01)
        check(f"Abl {cfg} delta {pd}%", pd, delta, 0.05)
else:
    checks.append((False, "ZS+ ablation configs missing from ablation_full.json", "-", "-"))

# ---------------- Conformal checks (on ZS+ point forecasts) ----------------
print("\nconformal entry keys:", list(cf[0].keys()))
cov90 = [r["coverage_90_per_h"] for r in cf]
cov95 = [r["coverage_95_per_h"] for r in cf]
check("Conf mean 90 cov 0.952", 0.952, float(np.mean(cov90)))
check("Conf mean 95 cov 0.975", 0.975, float(np.mean(cov95)))
valid = sum(1 for c in cov90 if c >= 0.90)
check("Conf 25/29 valid @90", 25, valid, 0)
misses = [r["region"] for r in cf if r["coverage_90_per_h"] < 0.90]
print("coverage misses:", misses)
check("Conf misses = FPL/PJM/QLD1/UK_16",
      "US_FPL,US_PJM,QLD1,UK_16_Scotland",
      ",".join(sorted(misses, key=lambda m: (not m.startswith("US"), m))))
width_ratio = float(np.mean([r["width_90_per_h"] / r["point_mae"] for r in cf]))
check("Conf width/MAE 6.8", 6.8, width_ratio, 0.01)
crps = float(np.mean([r["crps_90"] for r in cf]))
check("Conf CRPS 95.2", 95.2, crps, 0.01)
check("Conf ZS+ point MAE 41.5", 41.5, float(np.mean([r["point_mae"] for r in cf])), 0.01)
check("Conf raw-ZS point MAE 52.0", 52.0, float(np.mean([r["point_mae_raw_zs"] for r in cf])), 0.01)

# ---------------- Temporal OOD checks ----------------
r_std = float(np.mean([r["ratio_Standard (80/20)"] for r in to]))
r_75 = float(np.mean([r["ratio_9-month (75/25)"] for r in to]))
r_50 = float(np.mean([r["ratio_6-month (50/50)"] for r in to]))
check("OOD 80/20 = 1.091", 1.091, r_std)
check("OOD 75/25 = 1.114", 1.114, r_75)
check("OOD 50/50 = 1.222", 1.222, r_50)
check("OOD +2%", 2, (r_75 / r_std - 1) * 100, 0.3)
check("OOD +12%", 12, (r_50 / r_std - 1) * 100, 0.05)
rp_std = float(np.mean([r["ratio_plus_Standard (80/20)"] for r in to]))
rp_75 = float(np.mean([r["ratio_plus_9-month (75/25)"] for r in to]))
rp_50 = float(np.mean([r["ratio_plus_6-month (50/50)"] for r in to]))
check("OOD ZS+ 80/20 = 0.82", 0.82, rp_std)
check("OOD ZS+ 75/25 = 0.82", 0.82, rp_75)
check("OOD ZS+ 50/50 = 0.82", 0.82, rp_50)

# ---------------- Deployment checks ----------------
print("\ncrossover_days:", dp["crossover_days"])
paper_cross = {"QLD1": 30, "NSW1": ">270", "VIC1": 60, "SA1": 30,
               "UK_07": ">270", "UK_01": 30, "US_ERCO": 180, "US_BPAT": 30}
for reg, pv in paper_cross.items():
    jk = [k for k in dp["crossover_days"] if k.startswith(reg)]
    jv = dp["crossover_days"][jk[0]] if jk else None
    check(f"Deploy crossover {reg}", str(pv), str(jv) if jv is not None else "None")

# ZS+ vs ZS warm-up means over the 8 regions
# (paper: 57.0 vs 59.9 @30d, 52.5 vs 57.4 @60d, 51.0 vs 56.9 @90d)
for day, (p_zsp, p_zs) in {30: (57.0, 59.9), 60: (52.5, 57.4), 90: (51.0, 56.9)}.items():
    zsp_v, zs_v = [], []
    for series in dp["zero_shot_warmup"].values():
        e = next((x for x in series if x["days"] == day), None)
        if e and e.get("zs_mae") is not None:
            zs_v.append(e["zs_mae"])
            zsp_v.append(e["zsp_mae"])
    check(f"Deploy ZS+ mean @{day}d = {p_zsp}", p_zsp, float(np.mean(zsp_v)), 0.01)
    check(f"Deploy ZS mean @{day}d = {p_zs}", p_zs, float(np.mean(zs_v)), 0.01)

# ---------------- CarbonCast checks ----------------
paper_t3 = {  # region: (cc_sup, cc_zs, tc_zs, tc_zsp)
    "US_FPL": (20.8, 24.9, 42.4, 12.8), "US_MISO": (46.2, 46.2, 56.6, 46.6),
    "QLD1": (35.6, 85.2, 53.7, 27.1), "NSW1": (61.4, 97.9, 48.6, 46.2),
    "VIC1": (110.1, 107.2, 102.1, 98.1), "SA1": (58.6, 60.6, 64.3, 60.5),
    "UK_07": (85.3, 74.2, 73.5, 70.9), "UK_01": (42.9, 43.8, 45.5, 34.1),
    "US_BPAT": (6.7, 9.8, 8.6, 6.1),
}
degr, tc_wins, tcp_wins, tcp_ratios, overlaps = [], 0, 0, [], []
for r in cc:
    reg = r["region"]
    key = next((k for k in paper_t3 if reg.startswith(k)), None)
    if key:
        ps, pz, pt_, ptp = paper_t3[key]
        check(f"T3 {key} CC-Sup", ps, r["cc_supervised"]["mae"], 0.01)
        check(f"T3 {key} CC-ZS", pz, r["cc_zeroshot"]["mae"], 0.01)
        check(f"T3 {key} TC-ZS", pt_, r["transcif_zeroshot"]["mae"], 0.01)
        check(f"T3 {key} TC-ZS+", ptp, r["transcif_zs_plus"]["mae"], 0.01)
    degr.append(r["cc_zeroshot"]["mae"] / r["cc_supervised"]["mae"])
    if r["transcif_zeroshot"]["mae"] < r["cc_zeroshot"]["mae"]:
        tc_wins += 1
    if r["transcif_zs_plus"]["mae"] < r["cc_zeroshot"]["mae"]:
        tcp_wins += 1
    tcp_ratios.append(r["transcif_zs_plus"]["mae"] / r["cc_zeroshot"]["mae"])
    overlaps.append(r["norm_mismatch"]["range_overlap"])
check("T3 CC degr mean 1.28", 1.28, float(np.mean(degr)), 0.01)
check("T3 CC degr min 0.87", 0.87, float(min(degr)), 0.01)
check("T3 CC degr max 2.39", 2.39, float(max(degr)), 0.01)
check("T3 TC wins 5/9", 5, tc_wins, 0)
check("T3 TC-ZS+ wins 8/9", 8, tcp_wins, 0)
check("T3 TC-ZS+/CC-ZS mean ratio 0.73", 0.73, float(np.mean(tcp_ratios)), 0.01)
check("T3 norm overlap 95.8%", 0.958, float(np.mean(overlaps)), 0.01)

# ---------------- Report ----------------
print("\n" + "=" * 70)
fails = [c for c in checks if not c[0]]
for ok, label, pv, cv in checks:
    if not ok:
        print(f"FAIL  {label}: paper={pv} computed={cv}")
print(f"\n{len(checks) - len(fails)}/{len(checks)} checks passed, {len(fails)} FAILED")
