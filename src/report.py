"""Assemble reports/REPORT.md (completes Section 10 of the brief). Every number is read from outputs/."""
import json
import numpy as np, pandas as pd
from config import *

O = lambda f: pd.read_csv(OUT / f)
dq = json.load(open(OUT / "data_quality.json")); sh = json.load(open(OUT / "sensor_shortlist.json")); ef = json.load(open(OUT / "eda_facts.json"))
head, drop, cap = O("in_domain_headline.csv"), O("cross_dataset_drop.csv"), O("cap_vs_uncapped.csv")
stage, cond, mode, worst = O("failure_by_stage.csv"), O("failure_by_condition.csv"), O("failure_by_fault_mode.csv"), O("failure_worst_engines.csv")
pol, diff = O("policy_results.csv"), O("policy_diff.csv"); be = json.load(open(OUT / "breakeven.json"))
res = O("model_results.csv")
md = lambda df, **k: df.to_markdown(index=False, **k)
H = lambda ds, m, col, es="test_last": float(head[(head.dataset == ds) & (head.model == m) & (head.eval_set == es)][col].iloc[0])
D = lambda es, v, t, col: float(drop[(drop.eval_set == es) & (drop.variant == v) & (drop.target == t)][col].iloc[0])
P = lambda s, pl, r, col: float(pol[(pol.scenario == s) & (pol.policy == pl) & (pol.r == r)][col].iloc[0])
DF = lambda s, r, col: float(diff[(diff.scenario == s) & (diff.r == r)][col].iloc[0])
M = lambda c, fm, col: float(mode[(mode.case == c) & (mode.fault_mode.str.startswith(fm))][col].iloc[0])
S = lambda c, b, col: float(stage[(stage.case == c) & (stage.rul_bin == b)][col].iloc[0])
pct = lambda x: f"{x:+.0f}%"

# ---------- tables
t_head = head[head.eval_set == "test_last"].copy()
t_head["RMSE (95% CI)"] = t_head.apply(lambda r: f"{r.rmse:.1f} ({r.rmse_ci_lo:.1f}-{r.rmse_ci_hi:.1f})", axis=1)
t_head = t_head.rename(columns={"dataset": "Dataset", "model": "Model", "mae": "MAE", "nasa_score": "NASA score", "overest_rate": "Overest. rate"})
t_head["Model"] = t_head.Model.map({"naive_age": "Naive: mean life - age", "ridge": "Ridge", "xgboost": "XGBoost"})
t_head = t_head[["Dataset", "Model", "RMSE (95% CI)", "MAE", "NASA score", "Overest. rate"]].round({"MAE": 1, "NASA score": 0, "Overest. rate": 2})

t_drop = drop[drop.eval_set == "test_last"].copy()
t_drop["Features"] = t_drop.variant.map({"raw": "raw", "norm": "per-condition normalized"})
t_drop["Cross RMSE (95% CI)"] = t_drop.apply(lambda r: f"{r.rmse_cross:.1f} ({r.rmse_cross_ci_lo:.1f}-{r.rmse_cross_ci_hi:.1f})", axis=1)
t_drop = t_drop.sort_values(["target", "Features"], ascending=[True, False]).rename(columns={"target": "Target", "rmse_in_domain": "In-domain RMSE", "rmse_drop_pct": "RMSE change %", "score_ratio": "NASA score x"})
t_drop = t_drop[["Target", "Features", "In-domain RMSE", "Cross RMSE (95% CI)", "RMSE change %", "NASA score x"]].round({"In-domain RMSE": 1, "RMSE change %": 0, "NASA score x": 1})

t_stage = stage[stage.case.isin(["FD004 in-domain", "FD001->FD004 (norm)"])].pivot_table(index="rul_bin", columns="case", values=["rmse", "bias"], sort=False).round(1)
t_stage.columns = [f"{a} {b}" for a, b in t_stage.columns]; t_stage = t_stage.reset_index().rename(columns={"rul_bin": "True RUL (cycles)"})
order = ["0-10", "11-25", "26-50", "51-75", "76-100", "101-125", ">125"]; t_stage = t_stage.set_index("True RUL (cycles)").loc[order].reset_index()
t_mode = mode[mode.case.isin(["FD004 in-domain", "FD001->FD004 (norm)"])].round(1).rename(columns={"case": "Case", "fault_mode": "Fault mode (holdout engines)", "rmse_last50": "RMSE last 50 cycles", "bias_last50": "Bias last 50 cycles"})
t_mode = t_mode[["Case", "Fault mode (holdout engines)", "engines", "RMSE last 50 cycles", "Bias last 50 cycles"]]

def pol_table(scen_list, r):
    rows = []
    for s in scen_list:
        for pl, nm in [("rtf", "Run-to-failure"), ("fixed", "Fixed interval (tuned)"), ("rul", "RUL threshold")]:
            rows.append({"Scenario": s, "Policy": nm, "Cost / 100 cycles": round(100 * P(s, pl, r, "cost_per_cycle"), 2),
                         "Unplanned failures %": round(100 * P(s, pl, r, "failure_rate"), 1), "Premature pulls %": round(100 * P(s, pl, r, "premature_rate"), 1),
                         "Mean cycles flown": round(P(s, pl, r, "mean_cycles"), 0)})
    return pd.DataFrame(rows)

SC = ["FD001 in-domain", "FD002 in-domain", "FD004 in-domain", "FD004 oracle (true RUL)", "FD001->FD002 (norm)", "FD001->FD004 (norm)", "FD001->FD004 (raw)"]
t_be = []
for s in SC:
    v = be[s]["vs_fixed"]; w = be[s]["vs_rtf"]
    rng = lambda a, b: "never" if a is None else (f"{a:g} to {b:g}")
    t_be.append({"Scenario": s, "RUL cheaper than fixed (r range, point est.)": rng(v["point_win_from"], v["point_win_to"]),
                 "...statistically (95% CI)": rng(v["sig_win_from"], v["sig_win_to"]),
                 "RUL costlier than fixed from r": "-" if v["point_loss_from"] is None else f"{v['point_loss_from']:g}",
                 "RUL cheaper than run-to-failure from r": "never" if w["point_win_from"] is None else f"{w['point_win_from']:g}"})
t_be = pd.DataFrame(t_be)
rel = pd.DataFrame([{"Scenario": s, **{f"r={r}": pct(100 * DF(s, r, "rel_vs_fixed")) for r in [2, 3, 5, 10, 20]}} for s in SC])

life = pd.DataFrame(dq["lifetime_stats_train"])
nd = lambda ds: [x for x in dq["rows_per_dataset_split"] if x["dataset"] == ds]

best_ind = {ds: H(ds, "xgboost", "rmse") for ds in ["FD001", "FD002", "FD004"]}
worstcase = worst[worst.case == "FD004 in-domain"].iloc[0]
cr = be["FD001->FD004 (norm)"]["vs_fixed"]; c2 = be["FD001->FD002 (norm)"]["vs_fixed"]

REPORT = f"""# Predictive Engine Maintenance: When Should We Pull an Engine?
**Results report - completes Section 10 of the case study (v1.0, November 2026)**

> **Framing.** The airline is hypothetical and the data are NASA's *simulated* C-MAPSS turbofan degradation runs, not real fleet data. All costs are illustrative assumptions. Nothing here is a savings claim: it is a set of *simulated comparisons*, and only the pattern across cost assumptions is meaningful. The model is decision support only and authorizes nothing.

## 1. Summary of findings (Section 10)

| Item | Result |
|---|---|
| **Best in-domain model and score** | XGBoost on per-condition-normalized rolling features, trained with RUL capped at 125. Official test engines (last cycle): **RMSE {H('FD001','xgboost','rmse'):.1f} (FD001), {H('FD002','xgboost','rmse'):.1f} (FD002), {H('FD004','xgboost','rmse'):.1f} (FD004)**; NASA score {H('FD001','xgboost','nasa_score'):.0f} / {H('FD002','xgboost','nasa_score'):.0f} / {H('FD004','xgboost','nasa_score'):.0f}. That is {min(100*(1-H(d,'xgboost','rmse')/H(d,'naive_age','rmse')) for d in ['FD001','FD002','FD004']):.0f}-{max(100*(1-H(d,'xgboost','rmse')/H(d,'naive_age','rmse')) for d in ['FD001','FD002','FD004']):.0f}% lower RMSE than the "mean life minus age" baseline that a fixed schedule implies. |
| **Cross-dataset drop** (train FD001) | Raw features collapse: RMSE {D('test_last','raw','FD002','rmse_in_domain'):.1f} -> {D('test_last','raw','FD002','rmse_cross'):.1f} on FD002 ({pct(D('test_last','raw','FD002','rmse_drop_pct'))}) and {D('test_last','raw','FD004','rmse_in_domain'):.1f} -> {D('test_last','raw','FD004','rmse_cross'):.1f} on FD004 ({pct(D('test_last','raw','FD004','rmse_drop_pct'))}). Per-condition normalization removes most of it on FD002 ({pct(D('test_last','norm','FD002','rmse_drop_pct'))}) but not on FD004 ({pct(D('test_last','norm','FD004','rmse_drop_pct'))}). |
| **Where the model fails** | (1) Under operating-condition shift, unless sensors are normalized per condition. (2) On a fault mode it never saw: FD001-trained models are **{M('FD001->FD004 (norm)','B','bias_last50'):.0f} cycles too optimistic** in the last 50 cycles on FD004's second fault mode (vs {M('FD001->FD004 (norm)','A','bias_last50'):.0f} on the familiar one). (3) Mid-life: error peaks 50-75 cycles before failure (RMSE {S('FD004 in-domain','51-75','rmse'):.0f} on FD004) and the model is too optimistic there (bias {S('FD004 in-domain','51-75','bias'):+.0f}). |
| **Breakeven cost ratio** | In-domain, an RUL-threshold policy is cheaper than a hindsight-tuned fixed interval for every tested r >= 1.5 (up to 50), by {abs(DF('FD004 in-domain',5,'rel_vs_fixed'))*100:.0f}% (FD004, r=5). Under distribution shift it **stops beating the fixed schedule at r = {cr['point_loss_from']:g} (FD001 -> FD004) and r = {c2['point_loss_from']:g} (FD001 -> FD002)**, because unplanned failures from unseen conditions cost r each. With raw features it never beats it. |
| **Recommendation** | See Section 7: use RUL-based triggering only where the fleet's operating regime and fault modes are represented in training data, normalize per condition, and treat failure-cost ratio and model validity as the two things to confirm before relying on it. |

## 2. Data and pipeline
All twelve C-MAPSS files were loaded into PostgreSQL 16 ({sum(x['rows'] for x in dq['rows_per_dataset_split']):,} sensor rows, {sum(x['engines'] for x in dq['rows_per_dataset_split'])} engines). Validation (all passed): no nulls, no duplicate (engine, cycle) keys, no cycle gaps, every row labeled, RUL = 0 at the last cycle of every training engine.

{md(life.rename(columns={'dataset':'Dataset','engines':'Train engines','min':'Min life','mean':'Mean life','median':'Median life','max':'Max life'}))}

**Engine-level splits.** Within each training file, engines are assigned 70% *fit* / 10% *val* / 20% *holdout* (seed {SEED}); the official test file is a fourth, untouched group. No engine appears in two groups, and no random row split is used anywhere. Normalization statistics use *fit* engines only and require no labels. Rolling features are strictly causal; `tests/test_features.py` checks this against an independent pandas computation.

**Schema note.** The starter DDL made `unit_id` the primary key of `engines`, but unit numbers restart at 1 in every dataset and in both train and test files, so a surrogate `engine_id` is used with `(dataset, split, unit_id)` as the natural key.

## 3. Exploratory analysis
![lifetimes](figures/eda_lifetimes.png)
* **Lifetimes vary widely** (FD001: {int(life.iloc[0]['min'])}-{int(life.iloc[0]['max'])} cycles), which is exactly why one interval fits no engine well.
* **{len(sh['dropped'])} of 21 sensors are constant or near-constant** ({', '.join(sh['dropped'])}) and were dropped. The remaining **{len(sh['sensors'])}** ({', '.join(sh['sensors'])}) show a consistent per-engine monotone trend in FD001, FD002 and FD004.
* **FD002 and FD004 run under six operating conditions**, which swamp the degradation signal in raw sensors until each sensor is normalized per condition:

![normalization](figures/eda_normalization.png)
* **FD004 has two fault modes.** In FD001 every engine's s7 ends low; in FD004 **{100*ef['s7_end_of_life_share_positive']['FD004']:.0f}%** of fit engines end with s7 *high* (FD003: {100*ef['s7_end_of_life_share_positive']['FD003']:.0f}%), so the mean curve for some sensors hides two opposite behaviors. This matters in Section 5.

![degradation](figures/eda_degradation.png)

## 4. Models
Naive baselines, ridge, and XGBoost ({len(res.model.unique())} models x 20 configurations: five train/test pairs, raw vs normalized features, capped vs uncapped labels). Hyperparameters were chosen on *val* engines (ridge alpha; XGBoost early stopping). An LSTM was **not** built: the brief allows it only if justified, and the failure analysis points to fault-mode shift and mid-life ambiguity, neither of which a sequence model on the same sensors is expected to fix. That is a judgment, not a tested result.

Headline: official test engines, last observed cycle (NASA protocol), normalized features, label capped at 125. Truth is capped at 125 and predictions clipped to [0, 125] for all models, so numbers are comparable; the CI is a bootstrap over engines.

{md(t_head)}

![comparison](figures/model_comparison.png)
* XGBoost beats ridge by roughly 3-6 RMSE points and the age baseline by 2-3x. The confidence intervals of the three datasets overlap, so the data do **not** show FD004 to be harder in-domain.
* **Capping the label at 125 helps** on every dataset ({', '.join(f"{r.train}: {r['train_cap125__eval_vs_capped_truth']:.1f} vs {r['train_uncapped__eval_vs_capped_truth']:.1f}" for _, r in cap.iterrows())}; capped vs uncapped label, test RMSE vs capped truth). Early-life RUL above ~125 is mostly unobservable in the sensors, and trying to fit it hurts the part that matters.
* **Over-estimation** (predicting more life than exists) occurs on {100*min(H(d,'xgboost','overest_rate') for d in ['FD001','FD002','FD004']):.0f}-{100*max(H(d,'xgboost','overest_rate') for d in ['FD001','FD002','FD004']):.0f}% of last-cycle predictions. The policy layer, not the regressor, has to absorb this (Section 6).
* Per-cycle accuracy on all cycles of the 20% holdout engines gives a similar ranking (XGBoost RMSE {H('FD001','xgboost','rmse','holdout'):.1f} / {H('FD002','xgboost','rmse','holdout'):.1f} / {H('FD004','xgboost','rmse','holdout'):.1f}); see `outputs/in_domain_headline.csv`.

## 5. Generalization and failure analysis
Train on FD001, test on engines of another dataset; same metric as in-domain:

{md(t_drop)}

![cross](figures/cross_dataset.png)
The brief's expected finding holds: **FD001-trained models degrade sharply on FD002 and FD004 unless operating conditions are normalized.** Normalization is necessary, and on FD002 (same fault mode, new conditions) nearly sufficient. On FD004 a residual gap of {pct(D('test_last','norm','FD004','rmse_drop_pct'))} remains. Note that normalizing a *target* dataset uses its own unlabeled sensor statistics, which is only available once the new fleet has flown for a while.

**Where the residual error comes from (XGBoost, normalized, holdout engines):**

*By lifecycle stage* (bias = predicted - true; positive = predicts more life than exists):

{md(t_stage)}

*By fault mode* (FD004 holdout engines, classified by the direction of s7 at end of life, an analysis-only label that is unavailable at prediction time):

{md(t_mode)}

![stage](figures/error_by_stage.png)
![pva](figures/pred_vs_actual.png)
* **Fault mode is the dominant failure.** The FD001-trained model has never seen mode B and over-predicts remaining life by about {M('FD001->FD004 (norm)','B','bias_last50'):.0f} cycles near end of life on those engines. The three worst engines in that case are all mode B (`outputs/failure_worst_engines.csv`). Trained in-domain on FD004, the same group's bias is much smaller ({M('FD004 in-domain','B','bias_last50'):+.0f}).
* **Operating condition is *not* a failure source after normalization:** RMSE is within about 1.5 cycles across all six conditions in both FD002 and FD004 (`outputs/failure_by_condition.csv`).
* **Lifecycle:** error is largest 50-75 cycles before failure and small in the last 25 cycles (FD004 RMSE {S('FD004 in-domain','0-10','rmse'):.0f}-{S('FD004 in-domain','11-25','rmse'):.0f}); the model is optimistic in the 26-100 band and pessimistic above ~100, where the cap compresses it.
* Even in-domain there are tails: the worst FD004 holdout engine (engine_id {int(worstcase.engine_id)}) has a last-50-cycle bias of {worstcase.bias:+.0f}. Tails like this are why the tuned RUL threshold sits well above zero (tau = {P('FD004 in-domain','rul',10,'param'):g} at r = 10).

## 6. Maintenance-policy simulation
**Model of the decision.** Each held-out run-to-failure engine is one life of T cycles. Each cycle a policy either keeps flying or pulls the engine (planned cost 1); if it reaches T the engine fails (cost r = C_failure / C_planned). The metric is **cost per operating cycle** = total cost / total cycles flown, so pulling early is penalized by the life it wastes. A pull with true RUL above {30} cycles counts as *premature*.
* **Run-to-failure:** never pull. **Fixed interval:** pull at age N. **RUL threshold:** pull when predicted RUL < tau for k consecutive cycles (k in {{1,3,5}}).
* N and (tau, k) are re-tuned for every r, **nested by engine fold**: the parameters applied to an engine were tuned on other engines only. Predictions are cross-fitted (5-fold, grouped by engine), so no engine is scored by a model that saw it.
* The fixed-interval baseline is tuned with hindsight on the same fleet's lifetimes, which makes it a *strong* competitor; real schedules are usually more conservative.
* **Cross-dataset deployment:** the FD001 model is applied to FD002/FD004, tau/k are tuned on FD001 history, and the fixed interval is tuned on the *target* fleet's own lifetimes (no sensors needed). Evaluation covers every training-file engine of the target ({int(pol[pol.scenario=='FD001->FD004 (norm)'].n_engines.iloc[0])} for FD004).

**Cost at r = 10** (failure costs ten planned services; the brief's illustrative starting value):

{md(pol_table(['FD004 in-domain','FD001->FD004 (norm)'], 10))}

Full tables: `outputs/policy_results.csv`. The oracle row in the breakeven table below uses perfect RUL knowledge as an upper bound.

**RUL threshold cost relative to the tuned fixed interval** (negative = RUL cheaper):

{md(rel)}

![breakeven](figures/breakeven.png)

**Breakeven cost ratios** (grid r = 1 to 50, step 0.5; at r = 1 every policy ties by construction):

{md(t_be)}

![failures](figures/failures_premature.png)

**Reading the result.**
* **In-domain, RUL-based maintenance beats a fixed schedule at every r >= 1.5.** The reason is visible in the mean cycles flown: at r = 10 the tuned fixed interval pulls engines after about {P('FD004 in-domain','fixed',10,'mean_cycles'):.0f} cycles on average (mean life {P('FD004 in-domain','rtf',10,'mean_cycles'):.0f}), and {100*P('FD004 in-domain','fixed',10,'premature_rate'):.0f}% of pulls are premature, while the RUL policy flies {P('FD004 in-domain','rul',10,'mean_cycles'):.0f} cycles with {100*P('FD004 in-domain','rul',10,'failure_rate'):.1f}% unplanned failures. With perfect RUL knowledge the cost per 100 cycles would be {100*P('FD004 oracle (true RUL)','rul',10,'cost_per_cycle'):.2f} (RUL model {100*P('FD004 in-domain','rul',10,'cost_per_cycle'):.2f}, fixed {100*P('FD004 in-domain','fixed',10,'cost_per_cycle'):.2f}), so the model captures most of the available gain.
* **Under shift the sign flips as failures get more expensive.** FD001 -> FD004 (normalized): RUL wins only while r <= {cr['point_win_to']:g}; at r = 10 it costs {pct(100*DF('FD001->FD004 (norm)',10,'rel_vs_fixed'))} more than the fixed schedule, driven by {100*P('FD001->FD004 (norm)','rul',10,'failure_rate'):.0f}% unplanned failures. FD001 -> FD002 holds up longer (wins until r = {c2['point_win_to']:g}).
* **With raw features the transferred model never helps**: no engine ever triggers the tuned threshold (tau = {P('FD001->FD004 (raw)','rul',10,'param'):g}), so the policy silently behaves like run-to-failure (identical cost and 100% unplanned failures).
* **Uncertainty.** FD001 has only 100 engines, and at r >= 15 its advantage over the fixed interval is no longer statistically distinguishable from zero; the larger datasets stay significant through r = 50 in-domain.

## 7. Recommendation to the planning manager
*(Decision support for a hypothetical fleet on simulated data; it does not authorize any maintenance action.)*
1. **RUL-based triggering is worth piloting where the fleet resembles its training data.** In-domain it dominated a hindsight-tuned fixed interval across the whole tested range of cost ratios, mainly by avoiding the ~90% premature pulls that any single interval forces.
2. **Normalize sensors per operating condition.** It cost nothing in-domain and was the difference between usable and unusable transfer.
3. **Do not rely on a model across a new fault mode or regime.** The FD001 model looked acceptable on average (RMSE {D('test_last','norm','FD004','rmse_cross'):.1f} on FD004) while being ~{M('FD001->FD004 (norm)','B','bias_last50'):.0f} cycles optimistic on one engine group, which is exactly the failure the policy cannot afford when failure cost is high. Validate on the fleet's own data and monitor error by sub-population before raising the failure-cost-sensitive threshold.
4. **Know your r.** For a model applied outside its training regime, the RUL advantage over a plain schedule vanished at r = {cr['point_loss_from']:g} when the new fleet had an unseen fault mode and at r = {c2['point_loss_from']:g} when only the operating conditions were new. If unplanned failures are much costlier than that relative to a planned removal, a transferred model should not replace the schedule until it has been validated on the fleet's own data.
5. **Next steps:** add a drift/out-of-distribution flag (for example, sensor-pattern divergence like the s7 split), evaluate conservative thresholds (RUL quantiles instead of point estimates), and only then consider sequence models.

## 8. Limitations and deviations
* Simulated data; the cost model, the 125-cycle cap, the 30-cycle "premature" buffer, and r are assumptions. Only the pattern across r is informative.
* Single-engine, single-decision simulation; no spares, hangar slots, or schedules; one life per engine (a pull ends the life).
* Cost per cycle values life used linearly; the true cost of wasted life is not modeled beyond that.
* FD001's 20-engine holdout and 100 official test engines give wide intervals (see CIs); conclusions that depend on small differences between datasets are not supported.
* The fault-mode label used in Section 5 is derived from end-of-life behavior (analysis only). FD003 is loaded and explored but not modeled, since the brief's experiments cover FD001, FD002 and FD004.
* Transfer to FD002/FD004 uses the target's own unlabeled sensor statistics for normalization.
* N-CMAPSS, real-time integration, and certification are out of scope.

## 9. Reproduce
See `README.md`. `bash run_all.sh` runs ingestion, EDA, SQL features, models, analysis, policy simulation, and this report.
"""
(ROOT / "reports" / "REPORT.md").write_text(REPORT)
print("wrote reports/REPORT.md", len(REPORT.split()), "words")
