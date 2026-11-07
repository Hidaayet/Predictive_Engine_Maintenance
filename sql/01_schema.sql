-- Schema for the predictive engine maintenance project (PostgreSQL 16).
-- Deviation from the brief's starter DDL: unit_id restarts at 1 in every dataset
-- and in both train/test files, so it cannot be a primary key. We use a surrogate
-- engine_id and keep (dataset, split, unit_id) as a unique natural key.
DROP TABLE IF EXISTS cost_results, predictions, rul_labels, sensor_readings, engines CASCADE;

CREATE TABLE engines (
  engine_id integer PRIMARY KEY,
  dataset   text    NOT NULL CHECK (dataset IN ('FD001','FD002','FD003','FD004')),
  split     text    NOT NULL CHECK (split IN ('train','test')),   -- official C-MAPSS files
  unit_id   integer NOT NULL,
  n_cycles  integer NOT NULL,   -- cycles recorded (train: life length; test: truncated length)
  fold      text    NOT NULL CHECK (fold IN ('fit','val','holdout','test')),  -- engine-level split
  UNIQUE (dataset, split, unit_id)
);

CREATE TABLE sensor_readings (
  engine_id    integer NOT NULL REFERENCES engines(engine_id),
  cycle        integer NOT NULL,
  op_setting_1 real, op_setting_2 real, op_setting_3 real,
  s1 real, s2 real, s3 real, s4 real, s5 real, s6 real, s7 real, s8 real, s9 real,
  s10 real, s11 real, s12 real, s13 real, s14 real, s15 real, s16 real, s17 real,
  s18 real, s19 real, s20 real, s21 real,
  PRIMARY KEY (engine_id, cycle)
);

CREATE TABLE rul_labels (
  engine_id    integer NOT NULL REFERENCES engines(engine_id),
  cycle        integer NOT NULL,
  rul_uncapped integer NOT NULL,
  rul_capped   integer NOT NULL,   -- min(rul_uncapped, 125)
  PRIMARY KEY (engine_id, cycle)
);

-- run_id encodes the experiment: '<train>__<test>__<features>__<label>'
CREATE TABLE predictions (
  run_id   text    NOT NULL,
  model    text    NOT NULL,
  engine_id integer NOT NULL REFERENCES engines(engine_id),
  cycle    integer NOT NULL,
  pred_rul real    NOT NULL,
  PRIMARY KEY (run_id, model, engine_id, cycle)
);

CREATE TABLE cost_results (
  scenario  text NOT NULL,
  policy    text NOT NULL,
  cost_ratio real NOT NULL,
  param     real,
  n_engines integer, total_cost real, cycles_operated integer,
  cost_per_cycle real, n_failures integer, n_premature integer,
  PRIMARY KEY (scenario, policy, cost_ratio)
);
