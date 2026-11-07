"""Sanity tests: SQL rolling features are causal and match an independent pandas computation; folds are engine-disjoint."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from config import *

def main():
    c = connect()
    # 1. pandas re-computation of rolling features for 3 engines
    ids = pd.read_sql("SELECT engine_id FROM engines WHERE dataset='FD004' AND split='train' ORDER BY engine_id LIMIT 3", c).engine_id
    for eid in ids:
        sql_f = pd.read_sql(f"SELECT cycle, v_s11, m5_s11, sd5_s11, m15_s11 FROM features_norm WHERE engine_id={eid} ORDER BY cycle", c)
        v = sql_f.v_s11
        assert np.allclose(sql_f.m5_s11, v.rolling(5, min_periods=1).mean())
        assert np.allclose(sql_f.sd5_s11, v.rolling(5, min_periods=1).std().fillna(0))
        assert np.allclose(sql_f.m15_s11, v.rolling(15, min_periods=1).mean())
        # 2. causality: truncating the series at t must not change features at t
        t = len(v) // 2
        assert np.isclose(v.iloc[:t].rolling(5, min_periods=1).mean().iloc[-1], sql_f.m5_s11.iloc[t - 1])
    # 3. folds are engine-level: no engine in two folds, test fold only on official test
    assert pd.read_sql("SELECT count(*) n FROM (SELECT engine_id FROM engines GROUP BY 1 HAVING count(DISTINCT fold)>1) t", c).n[0] == 0
    assert pd.read_sql("SELECT count(*) n FROM engines WHERE (split='test') <> (fold='test')", c).n[0] == 0
    # 4. normalization stats were fit on fit-fold engines only
    chk = pd.read_sql("SELECT dataset, cond, n FROM norm_stats", c)
    fit_n = pd.read_sql("SELECT dataset, cond, count(*) n FROM readings_cond WHERE fold='fit' GROUP BY 1,2", c)
    assert chk.merge(fit_n, on=["dataset", "cond"]).pipe(lambda d: (d.n_x == d.n_y).all())
    print("all feature/leakage tests passed")

if __name__ == "__main__":
    main()
