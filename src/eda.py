"""Step 2: EDA. Lifetimes, constant/noisy sensors, degradation trends -> sensor shortlist + figures + report."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from common import *
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

eng = get_engine()
df = pd.read_sql("""
SELECT r.dataset, r.unit_id, r.cycle, o.condition_id, l.rul_uncapped, e.n_cycles, r.op_setting_1, r.op_setting_2, r.op_setting_3,
       r.s1,r.s2,r.s3,r.s4,r.s5,r.s6,r.s7,r.s8,r.s9,r.s10,r.s11,r.s12,r.s13,r.s14,r.s15,r.s16,r.s17,r.s18,r.s19,r.s20,r.s21
FROM sensor_readings r JOIN op_conditions o USING(dataset,split,unit_id,cycle)
JOIN rul_labels l USING(dataset,split,unit_id,cycle) JOIN engines e USING(dataset,split,unit_id)
WHERE r.split='train'""", eng)
ns = pd.read_sql("SELECT * FROM norm_stats", eng)
life = df.groupby(["dataset", "unit_id"]).n_cycles.first().reset_index()

# 1. lifetimes
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for ds, g in life.groupby("dataset"): ax.hist(g.n_cycles, bins=25, alpha=.55, label=f"{ds} (n={len(g)}, median {g.n_cycles.median():.0f})")
ax.set_xlabel("Engine lifetime (cycles to failure)"); ax.set_ylabel("Engines"); ax.legend(frameon=False); ax.set_title("Engine lifetimes (train engines)")
fig.tight_layout(); fig.savefig(FIG / "eda_lifetimes.png"); plt.close(fig)
lt = life.groupby("dataset").n_cycles.describe().round(1)

# 2. sensor diagnostics, per dataset, on condition-normalised values
rows = []
for ds in ["FD001", "FD002", "FD003", "FD004"]:
    d = df[df.dataset == ds].copy()
    mu = ns[ns.dataset == ds].pivot(index="condition_id", columns="sensor", values="mu")
    sd = ns[ns.dataset == ds].pivot(index="condition_id", columns="sensor", values="sd")
    for s in SENSORS:
        raw_sd = d[s].std()
        within_sd = float(sd[s].mean()) if (sd[s] != 1.0).any() else 0.0   # sd==1 flagged constants (COALESCE in SQL)
        z = (d[s] - d.condition_id.map(mu[s])) / d.condition_id.map(sd[s])
        # trend: Spearman of normalised value against engine age fraction, averaged over engines (sign-aware)
        d["_z"] = z; d["_age"] = d.cycle / d.n_cycles
        late = d[d._age > .8].groupby("unit_id")._z.mean(); early = d[d._age < .2].groupby("unit_id")._z.mean()
        sep = float((late - early).mean())     # mean shift in within-condition SDs, early vs late life
        rho = spearmanr(z, -d.rul_uncapped).correlation if within_sd > 0 else np.nan
        # noise: residual std after 5-cycle centered rolling mean, relative to total change early->late
        res = (d.assign(_z=z).groupby("unit_id")._z.apply(lambda x: (x - x.rolling(5, center=True, min_periods=1).mean()).std())).mean()
        rows.append(dict(dataset=ds, sensor=s, raw_std=raw_sd, constant=within_sd == 0, within_cond_sd=within_sd,
                         spearman_vs_age=rho, late_minus_early_sd=sep, noise_to_signal=(res / abs(sep)) if sep else np.inf))
sd_tab = pd.DataFrame(rows); sd_tab.to_csv(ROOT / "results" / "sensor_diagnostics.csv", index=False)

# shortlist: one COMMON list (needed so a model trained on one subset can score another).
# Rule: not constant; |Spearman rho| >= 0.30 on the base dataset FD001; same sign in FD002.
# FD003 is loaded and profiled but is not part of the planned experiments.
piv_rho = sd_tab.pivot(index="sensor", columns="dataset", values="spearman_vs_age")
piv_const = sd_tab.pivot(index="sensor", columns="dataset", values="constant")
ok_sign = np.sign(piv_rho["FD001"]) == np.sign(piv_rho["FD002"])
keep = [s for s in SENSORS if not piv_const.loc[s, ["FD001", "FD002", "FD004"]].any() and abs(piv_rho.loc[s, "FD001"]) >= 0.30 and ok_sign[s]]
print("shortlist:", keep)
pd.Series(keep).to_csv(ROOT / "results" / "sensor_shortlist.csv", index=False, header=["sensor"])

# heatmap of trend strength
fig, ax = plt.subplots(figsize=(6.2, 5.6))
im = ax.imshow(piv_rho.loc[SENSORS].values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(piv_rho.columns); ax.set_yticks(range(21)); ax.set_yticklabels(SENSORS)
for i, s in enumerate(SENSORS):
    ax.get_yticklabels()[i].set_fontweight("bold" if s in keep else "normal")
    for j, ds in enumerate(piv_rho.columns):
        v = piv_rho.loc[s, ds]; ax.text(j, i, "const" if piv_const.loc[s, ds] else f"{v:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, label="Spearman ρ (condition-normalised sensor vs cycles-elapsed-to-failure)"); ax.set_title("Degradation trend per sensor (bold = shortlisted)")
fig.tight_layout(); fig.savefig(FIG / "eda_sensor_trends.png"); plt.close(fig)

# 3. degradation trajectories for 4 shortlist sensors, FD001 raw and FD002 raw vs normalised
def traj(ax, d, s, norm, title):
    for u in d.unit_id.unique()[:25]:
        g = d[d.unit_id == u]
        ax.plot(g.rul_uncapped, g[s + "_n" if norm else s], lw=.5, alpha=.5, color="tab:blue")
    ax.invert_xaxis(); ax.set_title(title, fontsize=8); ax.set_xlabel("RUL (cycles)")
for ds in ["FD002"]:
    d = df[df.dataset == ds].copy()
    mu = ns[ns.dataset == ds].pivot(index="condition_id", columns="sensor", values="mu"); sdv = ns[ns.dataset == ds].pivot(index="condition_id", columns="sensor", values="sd")
    d["s11_n"] = (d.s11 - d.condition_id.map(mu.s11)) / d.condition_id.map(sdv.s11)
d1 = df[df.dataset == "FD001"].copy()
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
traj(axs[0], d1, "s11", False, "FD001 s11 (raw): clean monotone trend")
traj(axs[1], d, "s11", False, "FD002 s11 (raw): operating regime swamps trend")
traj(axs[2], d, "s11", True, "FD002 s11 (per-condition z-score): trend recovered")
fig.tight_layout(); fig.savefig(FIG / "eda_normalisation_effect.png"); plt.close(fig)

fig, axs = plt.subplots(2, 4, figsize=(11, 4.6))
d1n = d1.copy()
for ax, s in zip(axs.ravel(), ["s2", "s3", "s4", "s7", "s9", "s11", "s12", "s15"]):
    for u in d1n.unit_id.unique()[:30]:
        g = d1n[d1n.unit_id == u]; ax.plot(g.rul_uncapped, g[s], lw=.4, alpha=.5, color="tab:blue")
    ax.invert_xaxis(); ax.set_title(s, fontsize=8)
fig.suptitle("FD001: sensor value vs RUL for 30 engines (x: RUL, decreasing →)", fontsize=9); fig.tight_layout(); fig.savefig(FIG / "eda_fd001_trajectories.png"); plt.close(fig)

# report
const = sd_tab[sd_tab.constant].groupby("dataset").sensor.apply(lambda x: ", ".join(x))
rep = ["# EDA report (auto-generated by src/eda.py)\n",
       "## Engine lifetimes (train engines)\n", lt.to_markdown(), "\n",
       "![](figures/eda_lifetimes.png)\n",
       "## Constant sensors (zero variance within operating condition)\n", const.to_frame("constant sensors").to_markdown(), "\n",
       "FD002/FD004 sensors are *not* constant in raw form because six operating regimes shift the readings; constancy only shows after per-condition normalisation. "
       "In those datasets the raw sensors are dominated by regime, not wear.\n",
       "![](figures/eda_normalisation_effect.png)\n",
       "## Degradation trends\n", "![](figures/eda_sensor_trends.png)\n", "![](figures/eda_fd001_trajectories.png)\n",
       f"## Sensor shortlist ({len(keep)} sensors)\n", "Criteria: not constant; |Spearman ρ| ≥ 0.30 with elapsed life after per-condition normalisation in FD001; same sign in FD002. FD004 is not used in the sign rule because its two fault modes make pooled trends mixed. "
       "One common list is required so that models trained on one subset can be applied to another. Several kept sensors (s7, s12, s15, s20, s21) are weak when pooled over FD003/FD004 because the two fault modes degrade different sensors; this is flagged as a likely source of cross-dataset error.\n\n`" + ", ".join(keep) + "`\n",
       "Dropped sensors and why:\n", sd_tab[~sd_tab.sensor.isin(keep)].groupby("sensor").apply(lambda g: f"constant in {', '.join(g[g.constant].dataset)}" if g.constant.any() else f"weak trend in FD001 (|ρ|={abs(g[g.dataset=='FD001'].spearman_vs_age.iloc[0]):.2f})" if abs(g[g.dataset=='FD001'].spearman_vs_age.iloc[0])<0.30 else "trend sign flips between FD001 and FD002").to_frame("reason").to_markdown(), "\n",
       "## Issues found\n",
       "- Engine lifetimes vary widely (FD001: %d–%d cycles), so any fixed interval is wrong for many engines.\n" % (lt.loc['FD001','min'], lt.loc['FD001','max']),
       "- Test engines are truncated at a random point; only the final-cycle RUL is given, so test-set metrics use one row per engine.\n",
       "- Sensors are noisy; single-cycle readings are weak signals, which motivates rolling-window features.\n",
       "- FD003/FD004 contain two fault modes; no sensor flags the mode, so it is a latent source of error.\n"]
(ROOT / "reports" / "eda_report.md").write_text("\n".join(rep))
print(sd_tab[sd_tab.dataset=="FD001"].round(3).to_string())
print(piv_rho.round(2))
