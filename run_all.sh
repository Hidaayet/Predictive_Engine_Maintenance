#!/usr/bin/env bash
# End-to-end pipeline: data -> PostgreSQL -> EDA -> SQL features -> models -> policy simulation -> report figures -> dashboard.
# Prerequisites: python3 + requirements.txt, PostgreSQL 14+ binaries. Set DATABASE_URL to use your own server instead of the local one.
set -euo pipefail
cd "$(dirname "$0")"
bash scripts/download_data.sh
[ -n "${DATABASE_URL:-}" ] || bash scripts/start_postgres.sh
cd src
python3 ingest.py                 # 1. schema + load + validation  -> reports/data_quality.md
python3 - <<'PY'                  # 1b. per-condition normalisation statistics (SQL)
from common import *; from sqlalchemy import text
with get_engine().begin() as c: c.execute(text(open(ROOT/"sql/01_norm_stats.sql").read()))
PY
python3 eda.py                    # 2. EDA + sensor shortlist       -> reports/eda_report.md
python3 make_feature_sql.py       # 3a. generate SQL feature query
python3 - <<'PY'                  # 3b. run it
from common import *; from sqlalchemy import text
with get_engine().begin() as c: c.execute(text(open(ROOT/"sql/02_features.sql").read()))
PY
python3 models.py                 # 4+5. models, cross-dataset experiments (~3 min on 2 cores)
python3 policy.py                 # 6. policy simulation + breakeven
python3 analysis.py               #    error analysis, figures, tables
python3 dashboard.py              #    static dashboard
cd .. && python3 -m pytest -q tests
echo "done. see reports/summary.md"
