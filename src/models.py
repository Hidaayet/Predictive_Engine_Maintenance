"""Steps 4+5: baseline / linear / XGBoost models, in-domain and cross-dataset experiments.
All splits are BY ENGINE. Predictions go to the `predictions` table, metrics to results/metrics.csv.

Evaluation sets
  test_last : official held-out test engines, final observed cycle only (what the NASA score was designed for)
  allcycles : every cycle of train-split engines the model never saw (GroupKFold-by-engine out-of-fold for in-domain,
              or the whole other dataset for cross-dataset). Used for lifecycle-stage analysis and policy simulation.
"""
import io, sys, time, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sqlalchemy import text
from common import *

SENS = pd.read_csv(ROOT / "results" / "sensor_shortlist.csv").sensor.tolist()
FEATS = {
    "raw":  ["cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"{s}_raw{x}" for s in SENS for x in ("", "_rm10", "_rs10")],
    "norm": ["cycle"] + [f"{s}{x}" for s in SENS for x in ("_n", "_rm10", "_rs10", "_rm30")],
}
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=2.0, tree_method="hist", n_jobs=2, random_state=SEED)

def load():
    eng = get_engine()
    f = pd.read_sql("SELECT * FROM features", eng)
    l = pd.read_sql("SELECT dataset, split, unit_id, cycle, rul_uncapped, rul_capped FROM rul_labels", eng)
    return f.merge(l, on=["dataset", "split", "unit_id", "cycle"], how="left")

class Naive:
    """Fixed-lifetime baseline: RUL = mean training lifetime - cycles flown (floored at 0). This is what a fixed interval implies."""
    def fit(self, X, y, life=None): self.life = life; return self
    def predict(self, X): return np.maximum(self.life - X["cycle"].values, 0)

def make(model):
    if model == "linear": return make_pipeline(StandardScaler(), LinearRegression())
    if model == "xgb": return XGBRegressor(**XGB_PARAMS)
    return Naive()

def fit_predict(model, tr, te_list, feats, target):
    cols = FEATS[feats]
    m = make(model)
    if model == "naive":
        m.fit(None, None, life=tr.groupby("unit_id").cycle.max().mean()); return [m.predict(t[cols]) for t in te_list]
    m.fit(tr[cols], tr[target]); return [m.predict(t[cols]) for t in te_list]

def main():
    df = load()
    tr_all = df[df.split == "train"]; te_all = df[df.split == "test"]
    last = te_all.sort_values("cycle").groupby(["dataset", "unit_id"]).tail(1)   # final observed cycle of test engines
    DS = {d: dict(train=tr_all[tr_all.dataset == d], test_last=last[last.dataset == d]) for d in ["FD001", "FD002", "FD004"]}
    pred_rows, met_rows = [], []
    def record(exp, src, tgt, model, target, feats, eval_set, frame, pred):
        out = frame[["dataset", "split", "unit_id", "cycle"]].copy(); out["pred_rul"] = np.asarray(pred, float)
        out.insert(0, "eval_set", eval_set); out.insert(0, "model", model); out.insert(0, "experiment", exp); pred_rows.append(out)
        for truth in ("rul_uncapped", "rul_capped"):
            met_rows.append(dict(experiment=exp, src=src, tgt=tgt, model=model, target=target, feats=feats, eval_set=eval_set,
                                 truth=truth.split("_")[1], **metrics(frame[truth].values, pred)))
    t0 = time.time()
    configs = [(m, t, f) for m in ("naive", "linear", "xgb") for t in ("capped", "uncapped") for f in ("raw", "norm") if not (m == "naive" and (t == "uncapped" or f == "raw"))]
    for src in ["FD001", "FD002", "FD004"]:
        S = DS[src]["train"]
        for model, target, feats in configs:
            tcol = "rul_" + target
            # --- in-domain: out-of-fold (5-fold GroupKFold by engine) for all-cycle predictions
            oof = np.zeros(len(S)); gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            for tri, tei in gkf.split(S, groups=S.unit_id):
                oof[tei] = fit_predict(model, S.iloc[tri], [S.iloc[tei]], feats, tcol)[0]
            exp = f"{src}>{src}|{model}|{target}|{feats}"
            record(exp, src, src, model, target, feats, "allcycles", S, oof)
            # --- full fit on all train engines of src -> its official test set, plus other datasets (cross-dataset)
            targets = [(src, DS[src]["test_last"], "test_last")]
            if src == "FD001":
                for d in ("FD002", "FD004"): targets += [(d, DS[d]["test_last"], "test_last"), (d, DS[d]["train"], "allcycles")]
            preds = fit_predict(model, S, [t[1] for t in targets], feats, tcol)
            for (d, frame, es), p in zip(targets, preds):
                record(f"{src}>{d}|{model}|{target}|{feats}", src, d, model, target, feats, es, frame, p)
            print(f"{time.time()-t0:6.0f}s  {src} {model:6s} {target:8s} {feats}", flush=True)
    met = pd.DataFrame(met_rows); met.to_csv(ROOT / "results" / "metrics.csv", index=False)
    P = pd.concat(pred_rows); 
    eng = get_engine()
    with eng.begin() as c:
        c.execute(text("TRUNCATE predictions"))
        buf = io.StringIO(); P[["experiment", "model", "eval_set", "dataset", "split", "unit_id", "cycle", "pred_rul"]].to_csv(buf, index=False, header=False); buf.seek(0)
        c.connection.cursor().copy_expert("COPY predictions (experiment, model, eval_set, dataset, split, unit_id, cycle, pred_rul) FROM STDIN WITH CSV", buf)
    print("done", len(P), "prediction rows", f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
