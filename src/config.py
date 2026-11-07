"""Shared configuration. Override the DB connection with CMAPSS_DSN."""
import os
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
from pathlib import Path
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIG = ROOT / "reports" / "figures"
SQL = ROOT / "sql"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

DSN = os.environ.get("CMAPSS_DSN", "dbname=cmapss user=postgres host=localhost")
SEED = 42
RUL_CAP = 125
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SENSORS = [f"s{i}" for i in range(1, 22)]
OPS = ["op_setting_1", "op_setting_2", "op_setting_3"]
COLS = ["unit_id", "cycle"] + OPS + SENSORS


def connect():
    return psycopg2.connect(DSN)


def run_sql_file(path, conn=None):
    own = conn is None
    conn = conn or connect()
    with conn.cursor() as cur:
        cur.execute(Path(path).read_text())
    conn.commit()
    if own:
        conn.close()
