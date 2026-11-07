import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import nasa_score, metrics, get_engine

def test_nasa_score_asymmetry():
    assert nasa_score([10]) > nasa_score([-10])             # late penalised more than early
    assert nasa_score([0]) == 0
    assert abs(nasa_score([10]) - (np.exp(1) - 1)) < 1e-9 and abs(nasa_score([-13]) - (np.exp(1) - 1)) < 1e-9

def test_metrics_overestimation():
    m = metrics([10, 20, 30], [12, 18, 35]); assert abs(m["overest_rate"] - 2 / 3) < 1e-9

def test_labels_and_cap():
    e = get_engine()
    bad = pd.read_sql("SELECT count(*) FROM rul_labels WHERE rul_capped <> LEAST(rul_uncapped,125)", e).iloc[0, 0]; assert bad == 0
    last0 = pd.read_sql("""SELECT count(*) FROM (SELECT dataset,unit_id,min(rul_uncapped) m FROM rul_labels WHERE split='train' GROUP BY 1,2) t WHERE m<>0""", e).iloc[0, 0]; assert last0 == 0

def test_features_are_causal():
    """rolling mean at cycle t must equal the mean of normalised values over cycles max(1,t-9)..t only (no future data)."""
    e = get_engine()
    d = pd.read_sql("SELECT cycle, s11_n, s11_rm10 FROM features WHERE dataset='FD002' AND split='train' AND unit_id=3 ORDER BY cycle", e)
    expect = d.s11_n.rolling(10, min_periods=1).mean(); assert np.allclose(d.s11_rm10, expect, atol=1e-4)

def test_group_split_has_no_engine_overlap():
    from sklearn.model_selection import GroupKFold
    g = np.repeat(np.arange(50), 20)
    for tr, te in GroupKFold(5, shuffle=True, random_state=0).split(g, groups=g): assert not set(g[tr]) & set(g[te])

def test_no_test_engine_in_training_frames():
    e = get_engine()
    n = pd.read_sql("SELECT count(*) FROM predictions WHERE eval_set='allcycles' AND split<>'train'", e).iloc[0, 0]; assert n == 0
