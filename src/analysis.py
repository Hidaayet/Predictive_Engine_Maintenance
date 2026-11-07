"""Model comparison, generalization + error analysis, and all figures for Section 10. Writes reports/*.md fragments and figures."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import text
from common import *
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
C = {"naive": "#9aa0a6", "linear": "#e8a33d", "xgb": "#1f6fb2", "fixed": "#c0504d", "rul": "#1f6fb2", "rtf": "#9aa0a6", "oracle": "#3a9d5d"}
eng = get_engine()
m = pd.read_csv(ROOT / "results" / "metrics.csv")
pol = pd.read_csv(ROOT / "results" / "policy_summary.csv"); be = pd.read_csv(ROOT / "results" / "breakeven.csv")
out = []

# ---------- (a) model comparison, in-domain, official test set ----------
t = m[(m.eval_set == "test_last") & (m.truth == "uncapped") & (m.src == m.tgt)].copy()
t["config"] = np.where(t.model == "naive", "naive", t.model + " / " + t.target + " / " + t.feats)
tab = t.pivot_table(index="config", columns="src", values=["rmse", "mae", "nasa_score", "overest_rate"]).round(2)
tab.to_csv(ROOT / "results" / "model_comparison_test_last.csv")
best = t.loc[t.groupby("src").rmse.idxmin()][["src", "config", "rmse", "mae", "nasa_score", "overest_rate"]]
out.append("### In-domain, official test engines (final cycle), scored against the true (uncapped) RUL\n")
for d in ["FD001", "FD002", "FD004"]:
    x = t[t.src == d].sort_values("rmse")[["config", "rmse", "mae", "nasa_score", "overest_rate"]]
    x.columns = ["config (model / label / features)", "RMSE", "MAE", "NASA score", "overestimation rate"]
    out.append(f"**{d}**\n\n" + x.round(2).to_markdown(index=False) + "\n")
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
for ax, d in zip(axs, ["FD001", "FD002", "FD004"]):
    x = t[t.src == d]; x = x[(x.model == "naive") | (x.target == "capped")].sort_values("rmse", ascending=False)
    ax.barh(x.config, x.rmse, color=[C[mm] for mm in x.model]); ax.set_title(f"{d} test RMSE (cycles)", fontsize=9)
    for i, v in enumerate(x.rmse): ax.text(v + .5, i, f"{v:.1f}", va="center", fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig_a_model_comparison.png"); plt.close(fig)

# ---------- (b) cross-dataset drop ----------
x = m[(m.eval_set == "test_last") & (m.truth == "uncapped") & (m.model.isin(["xgb"])) & (m.target == "capped")]
cd = x.pivot_table(index=["src", "tgt"], columns="feats", values=["rmse", "nasa_score", "overest_rate"]).round(2)
rows = []
for tgt in ["FD002", "FD004"]:
    ind = x[(x.src == tgt) & (x.tgt == tgt) & (x.feats == "norm")].iloc[0]
    for feats in ["raw", "norm"]:
        c = x[(x.src == "FD001") & (x.tgt == tgt) & (x.feats == feats)].iloc[0]
        rows.append(dict(test_set=tgt, features=feats, rmse_FD001_trained=c.rmse, rmse_in_domain=ind.rmse, rmse_increase_pct=100 * (c.rmse / ind.rmse - 1),
                         nasa_FD001_trained=c.nasa_score, nasa_in_domain=ind.nasa_score, overest_FD001_trained=c.overest_rate, overest_in_domain=ind.overest_rate))
for feats in ["raw", "norm"]:
    base = x[(x.src == "FD001") & (x.tgt == "FD001") & (x.feats == feats)].iloc[0]
    rows.append(dict(test_set="FD001 (reference)", features=feats, rmse_FD001_trained=base.rmse, rmse_in_domain=base.rmse, rmse_increase_pct=0, nasa_FD001_trained=base.nasa_score, nasa_in_domain=base.nasa_score, overest_FD001_trained=base.overest_rate, overest_in_domain=base.overest_rate))
cdrop = pd.DataFrame(rows); cdrop.to_csv(ROOT / "results" / "cross_dataset_drop.csv", index=False)
out.append("### Cross-dataset: XGBoost (capped label) trained on FD001 vs trained in-domain, official test final cycle\n")
out.append(cdrop.round(2).to_markdown(index=False) + "\n")
# also linear for context
lin = m[(m.eval_set == "test_last") & (m.truth == "uncapped") & (m.model == "linear") & (m.target == "capped") & (m.src == "FD001")][["tgt", "feats", "rmse"]]
out.append("Linear regression trained on FD001 for context (RMSE on the same test sets):\n\n" + lin.round(1).to_markdown(index=False) + "\n")
fig, ax = plt.subplots(figsize=(6.6, 3.4)); w = .2; xs = np.arange(3)
vals = {"in-domain (norm)": [x[(x.src == d) & (x.tgt == d) & (x.feats == "norm")].rmse.iloc[0] for d in ["FD001", "FD002", "FD004"]],
        "FD001-trained, normalised": [x[(x.src == "FD001") & (x.tgt == d) & (x.feats == "norm")].rmse.iloc[0] for d in ["FD001", "FD002", "FD004"]],
        "FD001-trained, raw": [x[(x.src == "FD001") & (x.tgt == d) & (x.feats == "raw")].rmse.iloc[0] for d in ["FD001", "FD002", "FD004"]]}
for i, (k, v) in enumerate(vals.items()):
    ax.bar(xs + (i - 1) * .27, v, .27, label=k, color=["#3a9d5d", "#1f6fb2", "#c0504d"][i])
    for xx, vv in zip(xs + (i - 1) * .27, v): ax.text(xx, vv + .6, f"{vv:.0f}", ha="center", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(["test FD001", "test FD002", "test FD004"]); ax.set_ylabel("RMSE (cycles)"); ax.legend(frameon=False, fontsize=7); ax.set_title("Cross-dataset performance drop (XGBoost, capped label)")
fig.tight_layout(); fig.savefig(FIG / "fig_b_cross_dataset_drop.png"); plt.close(fig)

# ---------- error analysis on all-cycle held-out predictions ----------
def allc(exp):
    return pd.read_sql(text("""SELECT p.unit_id, p.cycle, p.pred_rul, l.rul_uncapped, l.rul_capped, e.n_cycles, o.condition_id, p.dataset
        FROM predictions p JOIN rul_labels l ON (l.dataset,l.split,l.unit_id,l.cycle)=(p.dataset,p.split,p.unit_id,p.cycle)
        JOIN engines e ON (e.dataset,e.split,e.unit_id)=(p.dataset,p.split,p.unit_id)
        JOIN op_conditions o ON (o.dataset,o.split,o.unit_id,o.cycle)=(p.dataset,p.split,p.unit_id,p.cycle)
        WHERE p.experiment=:e AND p.eval_set='allcycles'"""), eng, params={"e": exp})
bins = [-1, 10, 30, 60, 125, 1000]; labels = ["0-10", "11-30", "31-60", "61-125", ">125"]
stage_rows = []; exps = {"FD001 in-domain": "FD001>FD001|xgb|capped|norm", "FD002 in-domain": "FD002>FD002|xgb|capped|norm", "FD004 in-domain": "FD004>FD004|xgb|capped|norm",
                        "FD001->FD002 (norm)": "FD001>FD002|xgb|capped|norm", "FD001->FD004 (norm)": "FD001>FD004|xgb|capped|norm"}
D = {}
for k, e in exps.items():
    d = allc(e); d["stage"] = pd.cut(d.rul_uncapped, bins, labels=labels); d["err"] = d.pred_rul - d.rul_capped; D[k] = d
    g = d.groupby("stage", observed=True).apply(lambda z: pd.Series(dict(n=len(z), rmse=np.sqrt((z.err ** 2).mean()), bias=z.err.mean(), overest_rate=(z.err > 0).mean())), include_groups=False)
    g.insert(0, "scenario", k); stage_rows.append(g.reset_index())
stage = pd.concat(stage_rows); stage.to_csv(ROOT / "results" / "error_by_stage.csv", index=False)
out.append("### Where the model fails: error by lifecycle stage (all cycles of held-out engines, error vs capped RUL; XGBoost, capped, normalised)\n")
out.append(stage.round(2).to_markdown(index=False) + "\n")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for k, d in D.items():
    g = stage[stage.scenario == k]; axs[0].plot(g.stage.astype(str), g.rmse, marker="o", label=k, ls="--" if "->" in k else "-")
    axs[1].plot(g.stage.astype(str), g.overest_rate, marker="o", label=k, ls="--" if "->" in k else "-")
axs[0].set_xlabel("True RUL (cycles to failure)"); axs[0].set_ylabel("RMSE"); axs[0].set_title("Error is smallest close to failure"); axs[0].legend(frameon=False, fontsize=7)
axs[1].set_xlabel("True RUL (cycles to failure)"); axs[1].set_ylabel("Overestimation rate"); axs[1].axhline(.5, color="k", lw=.5); axs[1].set_title("Share of predictions that are too optimistic")
fig.tight_layout(); fig.savefig(FIG / "fig_c_error_by_stage.png"); plt.close(fig)

# by operating condition (FD002 / FD004 in-domain)
cond = []
for k in ["FD002 in-domain", "FD004 in-domain", "FD001->FD002 (norm)", "FD001->FD004 (norm)"]:
    d = D[k]; g = d.groupby("condition_id").apply(lambda z: pd.Series(dict(rmse=np.sqrt((z.err ** 2).mean()), bias=z.err.mean())), include_groups=False); g.insert(0, "scenario", k); cond.append(g.reset_index())
cond = pd.concat(cond); cond.to_csv(ROOT / "results" / "error_by_condition.csv", index=False)
out.append("### Error by operating condition cluster\n\n" + cond.pivot(index="condition_id", columns="scenario", values="rmse").round(1).to_markdown() + "\n")

# worst engines on official test sets (final cycle), FD001-trained and in-domain
wl = pd.read_sql(text("""SELECT p.experiment, p.dataset, p.unit_id, p.pred_rul, l.rul_uncapped AS true_rul, e.n_cycles FROM predictions p
   JOIN rul_labels l ON (l.dataset,l.split,l.unit_id,l.cycle)=(p.dataset,p.split,p.unit_id,p.cycle) JOIN engines e ON (e.dataset,e.split,e.unit_id)=(p.dataset,p.split,p.unit_id)
   WHERE p.eval_set='test_last' AND p.experiment IN ('FD001>FD001|xgb|capped|norm','FD004>FD004|xgb|capped|norm')"""), eng)
wl["err"] = wl.pred_rul - wl.true_rul
fig_rows = []
for e_, g in wl.groupby("experiment"):
    top = g.reindex(g.err.abs().sort_values(ascending=False).index).head(5)[["dataset", "unit_id", "n_cycles", "true_rul", "pred_rul", "err"]]
    out.append(f"### Five worst test engines, {e_}\n\n" + top.round(1).to_markdown(index=False) + "\n")
    fig_rows.append(g)
ee = wl[wl.experiment.str.startswith("FD001>FD001")]
big = (ee.true_rul > 125); out.append(f"FD001 test engines with true RUL > 125 (beyond the cap): {int(big.sum())} of {len(ee)}; their mean error is {ee[big].err.mean():.1f} cycles versus {ee[~big].err.mean():.1f} for the others. The cap makes the model unable to predict long remaining lives.\n")

# ---------- (f) predicted vs actual ----------
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, (exp, ttl) in zip(axs, [("FD001>FD001|xgb|capped|norm", "FD001 in-domain"), ("FD004>FD004|xgb|capped|norm", "FD004 in-domain"), ("FD001>FD004|xgb|capped|norm", "FD001-trained on FD004")]):
    g = pd.read_sql(text("""SELECT p.pred_rul, l.rul_uncapped t FROM predictions p JOIN rul_labels l ON (l.dataset,l.split,l.unit_id,l.cycle)=(p.dataset,p.split,p.unit_id,p.cycle)
        WHERE p.experiment=:e AND p.eval_set='test_last'"""), eng, params={"e": exp})
    ax.scatter(g.t, g.pred_rul, s=9, alpha=.6, color="#1f6fb2"); lim = [0, 200]; ax.plot(lim, lim, "k", lw=.6); ax.axhline(125, color="#c0504d", lw=.5, ls=":")
    ax.set_xlim(lim); ax.set_ylim(0, 170); ax.set_title(ttl, fontsize=9); ax.set_xlabel("Actual RUL (final cycle of test engine)"); ax.set_ylabel("Predicted RUL")
fig.tight_layout(); fig.savefig(FIG / "fig_f_pred_vs_actual.png"); plt.close(fig)

# ---------- policy figures ----------
def curves(ax, scen, title, legend=True):
    g = pol[pol.scenario == scen].sort_values("cost_ratio")
    for k, lab in [("rtf", "run-to-failure"), ("fixed", "best fixed interval"), ("rul", "RUL threshold"), ("oracle", "perfect-RUL oracle")]:
        ax.plot(g.cost_ratio, g[k], label=lab, color=C[k], lw=2 if k in ("rul", "fixed") else 1.2, ls=":" if k == "oracle" else "-")
    ax.set_xlim(1, 20); ax.set_ylim(0, 40); ax.set_xlabel("Cost ratio r = C_failure / C_planned"); ax.set_title(title, fontsize=9)
    if legend: ax.legend(frameon=False, fontsize=7)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
curves(axs[0], "FD001 in-domain", "FD001 in-domain (OOF predictions)"); axs[0].set_ylabel("Simulated cost per 1000 engine-cycles (C_planned = 1)")
curves(axs[1], "FD004 in-domain", "FD004 in-domain", False)
curves(axs[2], "FD001->FD004 (normalised, tuned on FD001)", "FD001 model on FD004, threshold tuned on FD001", False)
fig.tight_layout(); fig.savefig(FIG / "fig_e_breakeven_main.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(7.2, 4))
for scen, col, ls in [("FD001 in-domain", "#1f6fb2", "-"), ("FD004 in-domain", "#3a9d5d", "-"), ("FD001->FD002 (normalised, tuned on FD001)", "#e8a33d", "--"), ("FD001->FD004 (normalised, tuned on FD001)", "#c0504d", "--"),
                      ("FD001->FD004 (raw sensors, tuned on target)", "#7b5ea7", ":")]:
    g = pol[pol.scenario == scen].sort_values("cost_ratio"); ax.plot(g.cost_ratio, g.gain_pct, label=scen, color=col, ls=ls, lw=1.8)
ax.axhline(0, color="k", lw=.8); ax.set_xscale("log"); ax.set_xticks([1.1, 1.5, 2, 3, 5, 10, 20, 50]); ax.set_xticklabels(["1.1", "1.5", "2", "3", "5", "10", "20", "50"])
ax.set_ylim(-100, 60); ax.set_xlabel("Cost ratio r = C_failure / C_planned (log scale)"); ax.set_ylabel("RUL policy saving vs best fixed interval (%)")
ax.set_title("Where RUL-based maintenance stops beating a fixed schedule\n(below 0 = fixed interval is cheaper)", fontsize=9); ax.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(FIG / "fig_e2_rul_gain_vs_ratio.png"); plt.close(fig)

# (d) failures & early services at r=10 + (e) bars
cr = pd.read_csv(ROOT / "results" / "cost_results.csv"); c10 = cr[(cr.cost_ratio == 10)]
tbl = c10[c10.policy.isin(["rtf", "fixed", "rul", "oracle"])].assign(per_engine=lambda d: d.cost_per_1000_cycles)
sc = ["FD001 in-domain", "FD002 in-domain", "FD004 in-domain", "FD001->FD002 (normalised, tuned on FD001)", "FD001->FD004 (normalised, tuned on FD001)"]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
w = .2
for i, p_ in enumerate(["rtf", "fixed", "rul", "oracle"]):
    v = [tbl[(tbl.scenario == s) & (tbl.policy == p_)].cost_per_1000_cycles.iloc[0] for s in sc]; axs[0].bar(np.arange(5) + (i - 1.5) * w, v, w, color=C[p_], label={"rtf": "run-to-failure", "fixed": "fixed", "rul": "RUL", "oracle": "oracle"}[p_])
axs[0].set_xticks(range(5)); axs[0].set_xticklabels(["FD001", "FD002", "FD004", "FD001→\nFD002", "FD001→\nFD004"], fontsize=8); axs[0].set_ylabel("Cost per 1000 cycles at r = 10"); axs[0].legend(frameon=False, fontsize=7); axs[0].set_ylim(0, 60)
axs[0].set_title("Simulated cost at r = 10", fontsize=9)
for i, p_ in enumerate(["fixed", "rul"]):
    ef = [100 * tbl[(tbl.scenario == s) & (tbl.policy == p_)].n_failures.iloc[0] / tbl[(tbl.scenario == s) & (tbl.policy == p_)].n_engines.iloc[0] for s in sc]
    axs[1].bar(np.arange(5) + (i - .5) * .35, ef, .35, color=C[p_], label={"fixed": "fixed", "rul": "RUL"}[p_])
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(["FD001", "FD002", "FD004", "FD001→\nFD002", "FD001→\nFD004"], fontsize=8); axs[1].set_ylabel("Unplanned failures (% of engines)"); axs[1].set_title("Failures at r = 10", fontsize=9); axs[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig_d_cost_and_failures.png"); plt.close(fig)

out.append("### Policy comparison at r = 10 (evaluated on held-out half of engines, mean of 40 repeated splits)\n")
t10 = c10[c10.policy.isin(["rtf", "fixed", "rul", "oracle"])][["scenario", "policy", "cost_per_1000_cycles", "n_failures", "n_early_services", "n_engines", "mean_wasted_rul", "param", "param_k"]].copy()
t10["failure_rate_pct"] = 100 * t10.n_failures / t10.n_engines
out.append(t10.round(2).to_markdown(index=False) + "\n")
out.append("### Breakeven table\n\n" + be.round(2).to_markdown(index=False) + "\n")
(ROOT / "reports" / "results_tables.md").write_text("\n".join(out))
print("\n".join(out))

# ---------- bootstrap CIs for headline test RMSE (engines resampled; N = 100/259/248 final-cycle rows) ----------
rng = np.random.default_rng(SEED); ci = []
tl = pd.read_sql(text("""SELECT p.experiment, p.pred_rul, l.rul_uncapped t FROM predictions p JOIN rul_labels l ON (l.dataset,l.split,l.unit_id,l.cycle)=(p.dataset,p.split,p.unit_id,p.cycle)
    WHERE p.eval_set='test_last' AND p.experiment LIKE '%|xgb|capped|norm' OR (p.eval_set='test_last' AND p.experiment LIKE '%|xgb|capped|raw')"""), eng)
for e_, g in tl.groupby("experiment"):
    e = (g.pred_rul - g.t).values; bs = [np.sqrt(np.mean(e[rng.integers(0, len(e), len(e))] ** 2)) for _ in range(2000)]
    ci.append(dict(experiment=e_, n=len(e), rmse=np.sqrt(np.mean(e ** 2)), lo95=np.percentile(bs, 2.5), hi95=np.percentile(bs, 97.5)))
ci = pd.DataFrame(ci); ci.to_csv(ROOT / "results" / "rmse_bootstrap_ci.csv", index=False)
(ROOT / "reports" / "results_tables.md").write_text((ROOT / "reports" / "results_tables.md").read_text() + "\n### Bootstrap 95% CI of test RMSE (XGBoost, capped label)\n\n" + ci.round(1).to_markdown(index=False) + "\n")
print(ci.round(1).to_string(index=False))
