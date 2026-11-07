-- Schema for the predictive-maintenance case study (PostgreSQL).
-- Deviation from the case-study DDL: unit ids repeat across FD001..FD004 and across train/test,
-- so engines are keyed by (dataset, split, unit_id) and every child table carries that key.
DROP TABLE IF EXISTS cost_results, predictions, features, rul_labels, sensor_readings, op_conditions, norm_stats, engines CASCADE;

CREATE TABLE engines (
  dataset   text    NOT NULL CHECK (dataset IN ('FD001','FD002','FD003','FD004')),
  split     text    NOT NULL CHECK (split IN ('train','test')),  -- train = run-to-failure; test = truncated
  unit_id   integer NOT NULL,
  n_cycles  integer NOT NULL,                 -- cycles observed
  true_rul_at_end integer,                    -- only for split='test' (from RUL_FDxxx.txt)
  PRIMARY KEY (dataset, split, unit_id)
);

CREATE TABLE sensor_readings (
  dataset text NOT NULL, split text NOT NULL, unit_id integer NOT NULL, cycle integer NOT NULL,
  op_setting_1 real, op_setting_2 real, op_setting_3 real,
  s1 real, s2 real, s3 real, s4 real, s5 real, s6 real, s7 real, s8 real, s9 real, s10 real, s11 real,
  s12 real, s13 real, s14 real, s15 real, s16 real, s17 real, s18 real, s19 real, s20 real, s21 real,
  PRIMARY KEY (dataset, split, unit_id, cycle),
  FOREIGN KEY (dataset, split, unit_id) REFERENCES engines
);

-- RUL labels exist for every cycle of train engines; for test engines only the final cycle is labelled
CREATE TABLE rul_labels (
  dataset text NOT NULL, split text NOT NULL, unit_id integer NOT NULL, cycle integer NOT NULL,
  rul_uncapped integer NOT NULL,
  rul_capped   integer NOT NULL,              -- min(rul_uncapped, 125)
  PRIMARY KEY (dataset, split, unit_id, cycle),
  FOREIGN KEY (dataset, split, unit_id) REFERENCES engines
);

-- operating-condition cluster per reading (FD001/FD003: single condition 0; FD002/FD004: 6 clusters)
CREATE TABLE op_conditions (
  dataset text NOT NULL, split text NOT NULL, unit_id integer NOT NULL, cycle integer NOT NULL,
  condition_id smallint NOT NULL,
  PRIMARY KEY (dataset, split, unit_id, cycle),
  FOREIGN KEY (dataset, split, unit_id) REFERENCES engines
);

-- per (dataset, condition, sensor) mean/std from TRAIN-split engines only
CREATE TABLE norm_stats (
  dataset text NOT NULL, condition_id smallint NOT NULL, sensor text NOT NULL,
  mu double precision NOT NULL, sd double precision NOT NULL,
  PRIMARY KEY (dataset, condition_id, sensor)
);

CREATE TABLE predictions (
  experiment text NOT NULL,      -- e.g. 'FD001>FD002|xgb|capped|norm'
  model      text NOT NULL,
  eval_set   text NOT NULL,      -- 'oof' (held-out engines, all cycles) or 'test_last'
  dataset text NOT NULL, split text NOT NULL, unit_id integer NOT NULL, cycle integer NOT NULL,
  pred_rul real NOT NULL,
  PRIMARY KEY (experiment, eval_set, dataset, split, unit_id, cycle)
);

CREATE TABLE cost_results (
  scenario text NOT NULL, policy text NOT NULL, cost_ratio real NOT NULL,
  param real, n_engines integer, total_cost real, cost_per_1000_cycles real,
  n_failures integer, n_early_services integer, mean_wasted_rul real,
  PRIMARY KEY (scenario, policy, cost_ratio)
);
