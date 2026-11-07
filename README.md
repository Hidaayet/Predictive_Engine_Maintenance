# Predictive Engine Maintenance: When Should We Pull an Engine?

Can sensor-based Remaining Useful Life (RUL) prediction beat fixed-interval maintenance on cost, and how robust is that when operating conditions change? Hypothetical airline, NASA C-MAPSS simulated turbofan data. Costs are illustrative assumptions; nothing here is a claim about real savings, and the model is decision support only.

**Start with [`reports/summary.md`](reports/summary.md)** for the findings and recommendation.

## Headline results
| | |
|---|---|
| Best in-domain model | XGBoost, RUL capped at 125, per-condition normalised + rolling features. Test RMSE 15.3 / 26.6 / 27.1 cycles on FD001 / FD002 / FD004 (naive: 40.2 / 41.6 / 64.0) |
| FD001 → FD002 / FD004 | RMSE 43.3 / 48.0 raw (+63% / +77%); 29.8 / 35.6 with per-condition normalisation (+12% / +31%) |
| Where it fails | late-life errors are small (3.5–6 cycles RMSE in the last 10) but optimistic (64–80% overestimates); the 125 cap hurts long-life engines; FD001→FD004 has a +20 cycle bias near failure |
| Breakeven | RUL policy beats a tuned fixed interval from r ≈ 1.1–1.15 (34–44% cheaper at r = 10) but only if the threshold is calibrated on the target fleet; a threshold tuned on FD001 and moved to FD002 / FD004 loses to fixed above r ≈ 4.6 / 2.1 |

## Run it
```bash
pip install -r requirements.txt
./run_all.sh            # downloads data, starts a local PostgreSQL on :5433 (or set DATABASE_URL), runs everything (~5 min on 2 cores)
```
Needs Python 3.10+ and PostgreSQL 14+ binaries. `scripts/start_postgres.sh` creates a throw-away cluster in `.pgdata/`; to use your own server instead, export `DATABASE_URL=postgresql+psycopg2://user:pw@host:5432/db` first. Open `dashboard/index.html` in a browser for the dashboard.

## Layout
| Path | Content |
|---|---|
| `sql/00_schema.sql` | PostgreSQL schema (engines, sensor_readings, rul_labels, op_conditions, norm_stats, features, predictions, cost_results) |
| `sql/01_norm_stats.sql` | per-(dataset, operating condition) sensor mean/std from train engines |
| `sql/02_features.sql` | **generated** by `src/make_feature_sql.py`: causal rolling mean/std (10, 30 cycles) and per-condition z-scores as SQL window functions |
| `src/ingest.py` | load + integrity checks (`reports/data_quality.md`) |
| `src/eda.py` | EDA, sensor shortlist (`reports/eda_report.md`) |
| `src/models.py` | naive, linear, XGBoost; in-domain and cross-dataset experiments; metrics in `results/metrics.csv` |
| `src/policy.py` | policy simulation and cost-ratio sensitivity (`results/cost_results.csv`, `results/breakeven.csv`) |
| `src/analysis.py` | error analysis, figures (`reports/figures/`), `reports/results_tables.md` |
| `src/dashboard.py` | static dashboard `dashboard/index.html` |
| `tests/` | metric, label, causality and split-leakage checks |

## Method in brief
- **Splits are by engine, never by row.** In-domain all-cycle predictions are 5-fold `GroupKFold` out-of-fold by engine; official test engines are never used for training or tuning.
- **Labels:** RUL = last recorded cycle − cycle, capped (125) and uncapped, tested both ways. Test engines are truncated, so they are scored on their final cycle against `RUL_FDxxx.txt`.
- **Features:** 13 sensors chosen in EDA (constant and weak-trend sensors dropped), z-scored within 6 k-means operating conditions (fit on train only), plus causal 10-/30-cycle rolling statistics, computed in SQL.
- **Metrics:** RMSE, MAE, NASA asymmetric score, overestimation rate, cross-dataset drop, bootstrap CIs.
- **Policy simulation:** one engine lifecycle per held-out run-to-failure trajectory. Policies: run-to-failure, fixed interval (service at cycle T), RUL threshold (service after k consecutive predictions below τ), plus a perfect-RUL oracle bound. Cost = r per failure, 1 per planned service, compared as cost per 1000 engine-cycles because a planned service ends a lifecycle early. T and (τ, k) are tuned on a random half of engines and scored on the other half, 40 repeated splits.

## Deviations from the case study
- Primary keys are (dataset, split, unit_id) because unit ids repeat across datasets and train/test.
- Extra tables `op_conditions`, `norm_stats`; `predictions` has an `eval_set` column and `cost_results` stores the sensitivity grid.
- No LSTM (allowed only if justified; not attempted). Hyperparameters are fixed, not tuned.
- FD003 is loaded and profiled but not part of the experiments.
- N-CMAPSS, real-time integration and certification are out of scope.

Data: NASA Prognostics Data Repository, Turbofan Engine Degradation Simulation (C-MAPSS), fetched from a GitHub mirror.

