"""Step 6: maintenance-policy simulation + cost-ratio sensitivity (breakeven).

Simulation unit: one engine lifecycle taken from a run-to-failure trajectory (train-split engines of C-MAPSS,
using out-of-fold or cross-dataset predictions so the model never saw the engine). Engine fails at its last
cycle L. After each cycle t the policy may order planned service (cost C_planned=1, engine renewed, RUL wasted = L-t);
if no service is ordered by cycle L the engine fails (cost C_failure = r * C_planned).

Because a planned service ends a lifecycle early, comparing raw cost per lifecycle would always favour servicing
immediately. We therefore compare the long-run cost RATE = total cost / total cycles flown (renewal-reward argument).

Policies:  run-to-failure | fixed interval (service at cycle T) | RUL threshold (service when predicted RUL < tau for k
consecutive cycles) | oracle (perfect RUL, service the cycle before failure; lower bound).
Fairness: T and (tau,k) are tuned on a random half of engines and evaluated on the OTHER half (40 repeated splits,
both directions). Fixed interval therefore gets its best possible T for the fleet, which is a strong benchmark.
"""
import numpy as np, pandas as pd, io
from sqlalchemy import text
from common import *

R_GRID = np.array([1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50], float)
T_GRID = np.arange(20, 561, 5)
TAU_GRID = np.arange(5, 126, 5); K_GRID = (1, 3, 5)
N_SPLITS = 40

def load_traj(eng, experiment):
    p = pd.read_sql(text("""SELECT p.unit_id, p.cycle, p.pred_rul, e.n_cycles FROM predictions p
        JOIN engines e ON e.dataset=p.dataset AND e.split=p.split AND e.unit_id=p.unit_id
        WHERE p.experiment=:e AND p.eval_set='allcycles' ORDER BY p.dataset, p.unit_id, p.cycle"""), eng, params={"e": experiment})
    return [(g.pred_rul.values, int(g.n_cycles.iloc[0])) for _, g in p.groupby("unit_id")]

def outcomes_fixed(traj):
    """returns arrays [N, len(T_GRID)] of cycles flown, failed flag, wasted RUL"""
    L = np.array([l for _, l in traj])[:, None]; T = T_GRID[None, :]
    serv = T < L
    return np.where(serv, T, L).astype(float), (~serv).astype(float), np.where(serv, L - T, 0).astype(float)

def outcomes_rul(traj):
    params = [(t, k) for k in K_GRID for t in TAU_GRID]
    N = len(traj); cyc = np.zeros((N, len(params))); fail = np.zeros_like(cyc); waste = np.zeros_like(cyc)
    for i, (p, L) in enumerate(traj):
        for j, (tau, k) in enumerate(params):
            below = p < tau; run = 0; trig = None
            for t in range(1, L):               # service possible at cycles 1..L-1
                run = run + 1 if below[t - 1] else 0
                if run >= k: trig = t; break
            if trig is None: cyc[i, j], fail[i, j] = L, 1
            else: cyc[i, j], waste[i, j] = trig, L - trig
    return cyc, fail, waste, params

def rate(cyc, fail, idx, r):
    """cost rate per 1000 cycles for each param column, over engines idx"""
    cost = (fail[idx] * r + (1 - fail[idx]) * 1.0).sum(0); return 1000 * cost / cyc[idx].sum(0)

def evaluate_scenario(traj, name, rng, src_tune=None):
    N = len(traj); L = np.array([l for _, l in traj], float)
    fx = outcomes_fixed(traj); ru = outcomes_rul(traj); params = ru[3]
    if src_tune is not None: ru_src = outcomes_rul(src_tune); 
    rows, summ = [], []
    splits = []
    for s in range(N_SPLITS // 2):
        perm = rng.permutation(N); a, b = perm[: N // 2], perm[N // 2:]; splits += [(a, b), (b, a)]
    per_split = {r: {"rtf": [], "fixed": [], "rul": [], "oracle": [], "fx_T": [], "ru_param": [], "fx_o": None} for r in R_GRID}
    detail = {r: {"fixed": [], "rul": []} for r in R_GRID}
    for r in R_GRID:
        for tune, ev in splits:
            gT = int(np.argmin(rate(fx[0], fx[1], tune, r)))
            if src_tune is None: gR = int(np.argmin(rate(ru[0], ru[1], tune, r)))
            else: gR = int(np.argmin(rate(ru_src[0], ru_src[1], np.arange(len(src_tune)), r)))
            c_rtf = 1000 * (r * len(ev)) / L[ev].sum()
            c_or = 1000 * len(ev) / (L[ev] - 1).sum()
            c_fx = rate(fx[0], fx[1], ev, r)[gT]; c_ru = rate(ru[0], ru[1], ev, r)[gR]
            ps = per_split[r]; ps["rtf"].append(c_rtf); ps["oracle"].append(c_or); ps["fixed"].append(c_fx); ps["rul"].append(c_ru)
            ps["fx_T"].append(T_GRID[gT]); ps["ru_param"].append(params[gR])
            for key, o, g in (("fixed", fx, gT), ("rul", ru, gR)):
                f = o[1][ev, g]; w = o[2][ev, g]
                detail[r][key].append((f.sum(), (1 - f).sum(), w[f == 0].mean() if (f == 0).any() else 0.0, f.sum() * r + (1 - f).sum(), len(ev)))
    for r in R_GRID:
        ps = per_split[r]
        for pol in ("rtf", "fixed", "rul", "oracle"):
            v = np.array(ps[pol]); row = dict(scenario=name, policy=pol, cost_ratio=r, cost_per_1000_cycles=v.mean(), cost_sd=v.std())
            if pol in ("fixed", "rul"):
                d = np.array(detail[r][pol]); row.update(n_failures=d[:, 0].mean(), n_early_services=d[:, 1].mean(), mean_wasted_rul=d[:, 2].mean(),
                                                         total_cost=d[:, 3].mean(), n_engines=d[:, 4].mean(),
                                                         param=np.median(ps["fx_T"]) if pol == "fixed" else np.median([p[0] for p in ps["ru_param"]]),
                                                         param_k=np.median([p[1] for p in ps["ru_param"]]) if pol == "rul" else np.nan)
            else:
                row.update(n_failures=len(traj) / 2 if pol == "rtf" else 0, n_early_services=0 if pol == "rtf" else len(traj) / 2, mean_wasted_rul=0 if pol == "rtf" else 1, total_cost=np.nan, n_engines=N / 2, param=np.nan, param_k=np.nan)
            rows.append(row)
        fxv, ruv = np.array(ps["fixed"]), np.array(ps["rul"])
        summ.append(dict(scenario=name, cost_ratio=r, gain_pct=100 * (1 - ruv.mean() / fxv.mean()), rul_wins=float(np.mean(ruv < fxv - 1e-9)),
                         fixed=fxv.mean(), rul=ruv.mean(), rtf=np.mean(ps["rtf"]), oracle=np.mean(ps["oracle"])))
    return pd.DataFrame(rows), pd.DataFrame(summ)

def first_r(df, cond):
    d = df[cond(df)]
    return float(d.cost_ratio.min()) if len(d) else np.nan

def upper_cross(g):
    """Smallest r at which RUL policy becomes worse than the tuned fixed interval (linear interpolation of gain_pct), NaN if never."""
    g = g.sort_values("cost_ratio"); x, y = g.cost_ratio.values, g.gain_pct.values
    for i in range(1, len(x)):
        if y[i] < 0 <= y[i - 1] and x[i - 1] > 1.0: return float(x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i]))
    return np.nan

SCENARIOS = [  # name, experiment, source experiment for transfer tuning (or None)
    ("FD001 in-domain",           "FD001>FD001|xgb|capped|norm", None),
    ("FD002 in-domain",           "FD002>FD002|xgb|capped|norm", None),
    ("FD004 in-domain",           "FD004>FD004|xgb|capped|norm", None),
    ("FD001->FD002 (normalised, tuned on target)", "FD001>FD002|xgb|capped|norm", None),
    ("FD001->FD004 (normalised, tuned on target)", "FD001>FD004|xgb|capped|norm", None),
    ("FD001->FD002 (normalised, tuned on FD001)",  "FD001>FD002|xgb|capped|norm", "FD001>FD001|xgb|capped|norm"),
    ("FD001->FD004 (normalised, tuned on FD001)",  "FD001>FD004|xgb|capped|norm", "FD001>FD001|xgb|capped|norm"),
    ("FD001->FD002 (raw sensors, tuned on target)", "FD001>FD002|xgb|capped|raw", None),
    ("FD001->FD004 (raw sensors, tuned on target)", "FD001>FD004|xgb|capped|raw", None),
]

def main():
    eng = get_engine(); rng = np.random.default_rng(SEED)
    allrows, allsum = [], []
    for name, exp, src in SCENARIOS:
        traj = load_traj(eng, exp); st = load_traj(eng, src) if src else None
        rows, summ = evaluate_scenario(traj, name, rng, st); allrows.append(rows); allsum.append(summ)
        print(name, len(traj), "engines", flush=True)
    cr = pd.concat(allrows); sm = pd.concat(allsum)
    cr.to_csv(ROOT / "results" / "cost_results.csv", index=False); sm.to_csv(ROOT / "results" / "policy_summary.csv", index=False)
    be = []
    for name, g in sm.groupby("scenario", sort=False):
        be.append(dict(scenario=name,
            r_fixed_beats_rtf=first_r(g, lambda d: d.fixed < d.rtf * 0.99),
            r_rul_beats_rtf=first_r(g, lambda d: d.rul < d.rtf * 0.99),
            r_rul_beats_fixed_5pct=first_r(g, lambda d: d.gain_pct >= 5),
            r_rul_worse_than_fixed=upper_cross(g),
            r_rul_beats_fixed_robust=first_r(g, lambda d: (d.gain_pct > 0) & (d.rul_wins >= 0.9)),
            gain_pct_at_r10=float(g[g.cost_ratio == 10].gain_pct.iloc[0]), rul_win_share_at_r10=float(g[g.cost_ratio == 10].rul_wins.iloc[0]),
            gain_pct_at_r5=float(g[g.cost_ratio == 5].gain_pct.iloc[0]),
            rul_vs_oracle_gap_pct_at_r10=float(100 * (g[g.cost_ratio == 10].rul.iloc[0] / g[g.cost_ratio == 10].oracle.iloc[0] - 1))))
    be = pd.DataFrame(be); be.to_csv(ROOT / "results" / "breakeven.csv", index=False)
    with eng.begin() as c:
        c.execute(text("TRUNCATE cost_results"))
        out = cr.rename(columns={"scenario": "scenario"})[["scenario", "policy", "cost_ratio", "param", "n_engines", "total_cost", "cost_per_1000_cycles", "n_failures", "n_early_services", "mean_wasted_rul"]]
        out = out.assign(n_engines=out.n_engines.round().astype('Int64'), n_failures=out.n_failures.round().astype('Int64'), n_early_services=out.n_early_services.round().astype('Int64'))
        buf = io.StringIO(); out.to_csv(buf, index=False, header=False); buf.seek(0)
        c.connection.cursor().copy_expert("COPY cost_results FROM STDIN WITH CSV", buf)
    pd.set_option("display.width", 250); print(be.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
