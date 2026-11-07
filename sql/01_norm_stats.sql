-- Per-(dataset, operating condition) mean/std of every sensor, computed on TRAIN engines only.
-- Normalising within condition removes the operating-regime signal from FD002/FD004 sensors.
TRUNCATE norm_stats;
INSERT INTO norm_stats (dataset, condition_id, sensor, mu, sd)
SELECT r.dataset, o.condition_id, v.sensor, avg(v.val), COALESCE(NULLIF(stddev_samp(v.val), 0), 1.0)
FROM sensor_readings r
JOIN op_conditions o USING (dataset, split, unit_id, cycle)
CROSS JOIN LATERAL (VALUES
  ('s1', r.s1), ('s2', r.s2), ('s3', r.s3), ('s4', r.s4), ('s5', r.s5), ('s6', r.s6), ('s7', r.s7),
  ('s8', r.s8), ('s9', r.s9), ('s10', r.s10), ('s11', r.s11), ('s12', r.s12), ('s13', r.s13), ('s14', r.s14),
  ('s15', r.s15), ('s16', r.s16), ('s17', r.s17), ('s18', r.s18), ('s19', r.s19), ('s20', r.s20), ('s21', r.s21)
) AS v(sensor, val)
WHERE r.split = 'train'
GROUP BY r.dataset, o.condition_id, v.sensor;
