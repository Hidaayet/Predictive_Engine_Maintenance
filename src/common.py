import os, numpy as np
from pathlib import Path
from sqlalchemy import create_engine
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres@localhost:5433/cmapss")
RUL_CAP = 125
SEED = 42
def get_engine():
    return create_engine(DB_URL)
SENSORS = [f"s{i}" for i in range(1, 22)]
OPS = ["op_setting_1", "op_setting_2", "op_setting_3"]
def nasa_score(err):
    err = np.asarray(err, float)  # err = pred - true
    # exponent clipped at 500 so absurd extrapolations give a huge finite score instead of inf
    return float(np.sum(np.where(err < 0, np.exp(np.minimum(-err / 13, 500)) - 1, np.exp(np.minimum(err / 10, 500)) - 1)))
def metrics(y_true, y_pred):
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return dict(n=len(e), rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))),
                nasa_score=nasa_score(e), nasa_score_mean=nasa_score(e) / len(e),
                overest_rate=float(np.mean(e > 0)))
