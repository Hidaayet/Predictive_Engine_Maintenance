### In-domain, official test engines (final cycle), scored against the true (uncapped) RUL

**FD001**

| config (model / label / features)   |   RMSE |   MAE |   NASA score |   overestimation rate |
|:------------------------------------|-------:|------:|-------------:|----------------------:|
| xgb / capped / norm                 |  15.27 | 11.35 |       368.18 |                  0.57 |
| linear / capped / norm              |  17.4  | 14.26 |       434.9  |                  0.5  |
| xgb / capped / raw                  |  19.28 | 14.19 |       875.99 |                  0.55 |
| linear / capped / raw               |  22.48 | 17.74 |      1533.77 |                  0.55 |
| xgb / uncapped / norm               |  25.95 | 18.5  |      5911.41 |                  0.71 |
| linear / uncapped / norm            |  27.83 | 23.74 |      3263.8  |                  0.73 |
| xgb / uncapped / raw                |  29.04 | 21.14 |     23741    |                  0.67 |
| linear / uncapped / raw             |  31.92 | 26.6  |      7358.47 |                  0.77 |
| naive                               |  40.2  | 32.05 |     25527.8  |                  0.57 |

**FD002**

| config (model / label / features)   |   RMSE |   MAE |   NASA score |   overestimation rate |
|:------------------------------------|-------:|------:|-------------:|----------------------:|
| xgb / capped / norm                 |  26.58 | 17.77 |     10997    |                  0.41 |
| xgb / capped / raw                  |  27.81 | 19.9  |      9008.02 |                  0.42 |
| xgb / uncapped / norm               |  27.85 | 19.49 |     32861.9  |                  0.57 |
| xgb / uncapped / raw                |  28.7  | 21.22 |     27562.4  |                  0.57 |
| linear / capped / norm              |  29.3  | 20.93 |     12962.7  |                  0.46 |
| linear / uncapped / norm            |  29.58 | 23.45 |     16807.9  |                  0.61 |
| linear / capped / raw               |  31.02 | 23.26 |     14682.5  |                  0.47 |
| linear / uncapped / raw             |  31.75 | 25.89 |     17919.1  |                  0.61 |
| naive                               |  41.61 | 34.26 |     84738.5  |                  0.5  |

**FD004**

| config (model / label / features)   |   RMSE |   MAE |       NASA score |   overestimation rate |
|:------------------------------------|-------:|------:|-----------------:|----------------------:|
| xgb / capped / norm                 |  27.15 | 19.88 |   4702.62        |                  0.47 |
| xgb / capped / raw                  |  29.65 | 22.38 |   6355.62        |                  0.49 |
| linear / capped / norm              |  30.87 | 23.88 |   8329.97        |                  0.5  |
| linear / capped / raw               |  35.26 | 28.65 |  13671.9         |                  0.51 |
| xgb / uncapped / norm               |  37.8  | 26.73 |      7.8865e+06  |                  0.68 |
| xgb / uncapped / raw                |  40.79 | 30.15 |      1.9428e+06  |                  0.72 |
| linear / uncapped / norm            |  43.03 | 34.76 | 703765           |                  0.69 |
| linear / uncapped / raw             |  45.4  | 37.09 | 385177           |                  0.71 |
| naive                               |  64.04 | 53.16 |      3.47753e+06 |                  0.52 |

### Cross-dataset: XGBoost (capped label) trained on FD001 vs trained in-domain, official test final cycle

| test_set          | features   |   rmse_FD001_trained |   rmse_in_domain |   rmse_increase_pct |   nasa_FD001_trained |   nasa_in_domain |   overest_FD001_trained |   overest_in_domain |
|:------------------|:-----------|---------------------:|-----------------:|--------------------:|---------------------:|-----------------:|------------------------:|--------------------:|
| FD002             | raw        |                43.32 |            26.58 |               62.95 |             46538.3  |         10997    |                    0.54 |                0.41 |
| FD002             | norm       |                29.81 |            26.58 |               12.13 |             15453.6  |         10997    |                    0.41 |                0.41 |
| FD004             | raw        |                48    |            27.15 |               76.82 |             63000.9  |          4702.62 |                    0.46 |                0.47 |
| FD004             | norm       |                35.63 |            27.15 |               31.24 |             15792.1  |          4702.62 |                    0.5  |                0.47 |
| FD001 (reference) | raw        |                19.28 |            19.28 |                0    |               875.99 |           875.99 |                    0.55 |                0.55 |
| FD001 (reference) | norm       |                15.27 |            15.27 |                0    |               368.18 |           368.18 |                    0.57 |                0.57 |

Linear regression trained on FD001 for context (RMSE on the same test sets):

| tgt   | feats   |    rmse |
|:------|:--------|--------:|
| FD001 | raw     |    22.5 |
| FD002 | raw     | 34573.1 |
| FD004 | raw     | 35188.4 |
| FD001 | norm    |    17.4 |
| FD002 | norm    |    35.4 |
| FD004 | norm    |    55.9 |

### Where the model fails: error by lifecycle stage (all cycles of held-out engines, error vs capped RUL; XGBoost, capped, normalised)

| stage   | scenario            |     n |   rmse |   bias |   overest_rate |
|:--------|:--------------------|------:|-------:|-------:|---------------:|
| 0-10    | FD001 in-domain     |  1100 |   3.49 |   1.9  |           0.74 |
| 11-30   | FD001 in-domain     |  2000 |   7.72 |   3.1  |           0.64 |
| 31-60   | FD001 in-domain     |  3000 |  16.72 |   7.68 |           0.64 |
| 61-125  | FD001 in-domain     |  6500 |  19.31 |   2.35 |           0.57 |
| >125    | FD001 in-domain     |  8031 |  12.6  |  -6.98 |           0.23 |
| 0-10    | FD002 in-domain     |  2860 |   3.83 |   2.13 |           0.77 |
| 11-30   | FD002 in-domain     |  5200 |   8.65 |   3.47 |           0.65 |
| 31-60   | FD002 in-domain     |  7800 |  19.04 |   9.94 |           0.7  |
| 61-125  | FD002 in-domain     | 16900 |  18.97 |   2.47 |           0.57 |
| >125    | FD002 in-domain     | 20999 |  13.03 |  -7.19 |           0.21 |
| 0-10    | FD004 in-domain     |  2739 |   5.81 |   3.43 |           0.8  |
| 11-30   | FD004 in-domain     |  4980 |   9.84 |   4.71 |           0.71 |
| 31-60   | FD004 in-domain     |  7470 |  20.41 |  10.75 |           0.73 |
| 61-125  | FD004 in-domain     | 16185 |  20.38 |   3.51 |           0.59 |
| >125    | FD004 in-domain     | 29875 |  12.45 |  -6.27 |           0.22 |
| 0-10    | FD001->FD002 (norm) |  2860 |   7.14 |   5.35 |           0.88 |
| 11-30   | FD001->FD002 (norm) |  5200 |  10.15 |   5.15 |           0.73 |
| 31-60   | FD001->FD002 (norm) |  7800 |  18.42 |   7.73 |           0.63 |
| 61-125  | FD001->FD002 (norm) | 16900 |  20.43 |  -4.2  |           0.42 |
| >125    | FD001->FD002 (norm) | 20999 |  18.57 | -13.01 |           0.08 |
| 0-10    | FD001->FD004 (norm) |  2739 |  25.29 |  19.99 |           0.98 |
| 11-30   | FD001->FD004 (norm) |  4980 |  24.1  |  16.48 |           0.83 |
| 31-60   | FD001->FD004 (norm) |  7470 |  27.29 |  14.17 |           0.67 |
| 61-125  | FD001->FD004 (norm) | 16185 |  25.06 |  -3.18 |           0.45 |
| >125    | FD001->FD004 (norm) | 29875 |  21.39 | -15.38 |           0.11 |

### Error by operating condition cluster

|   condition_id |   FD001->FD002 (norm) |   FD001->FD004 (norm) |   FD002 in-domain |   FD004 in-domain |
|---------------:|----------------------:|----------------------:|------------------:|------------------:|
|              0 |                  18.8 |                  23.6 |              15.5 |              15.5 |
|              1 |                  17.3 |                  23.5 |              15.4 |              15.7 |
|              2 |                  17.5 |                  23.6 |              15.5 |              15.8 |
|              3 |                  18.7 |                  23.9 |              15.8 |              15.8 |
|              4 |                  17.3 |                  23.4 |              15.4 |              15.6 |
|              5 |                  18.6 |                  23.4 |              15.4 |              15.8 |

### Five worst test engines, FD001>FD001|xgb|capped|norm

| dataset   |   unit_id |   n_cycles |   true_rul |   pred_rul |   err |
|:----------|----------:|-----------:|-----------:|-----------:|------:|
| FD001     |        45 |        152 |        114 |       54.8 | -59.2 |
| FD001     |        74 |        137 |        126 |       89.1 | -36.9 |
| FD001     |        89 |        177 |        136 |      101.1 | -34.9 |
| FD001     |        93 |        244 |         85 |       52.9 | -32.1 |
| FD001     |        55 |        113 |        137 |      105.3 | -31.7 |

### Five worst test engines, FD004>FD004|xgb|capped|norm

| dataset   |   unit_id |   n_cycles |   true_rul |   pred_rul |   err |
|:----------|----------:|-----------:|-----------:|-----------:|------:|
| FD004     |       127 |         98 |        195 |      120   | -75   |
| FD004     |        83 |        274 |        172 |      101.3 | -70.7 |
| FD004     |       122 |        117 |        192 |      121.4 | -70.6 |
| FD004     |        95 |        110 |        171 |      100.6 | -70.4 |
| FD004     |       153 |        130 |        173 |      102.9 | -70.1 |

FD001 test engines with true RUL > 125 (beyond the cap): 11 of 100; their mean error is -24.0 cycles versus 2.9 for the others. The cap makes the model unable to predict long remaining lives.

### Policy comparison at r = 10 (evaluated on held-out half of engines, mean of 40 repeated splits)

| scenario                                    | policy   |   cost_per_1000_cycles |   n_failures |   n_early_services |   n_engines |   mean_wasted_rul |   param |   param_k |   failure_rate_pct |
|:--------------------------------------------|:---------|-----------------------:|-------------:|-------------------:|------------:|------------------:|--------:|----------:|-------------------:|
| FD001 in-domain                             | rtf      |                  48.49 |        50    |               0    |        50   |              0    |   nan   |       nan |             100    |
| FD001 in-domain                             | fixed    |                   8.79 |         0.75 |              49.25 |        50   |             78.72 |   127.5 |       nan |               1.5  |
| FD001 in-domain                             | rul      |                   5.53 |         0.52 |              49.48 |        50   |              8.48 |    15   |         5 |               1.05 |
| FD001 in-domain                             | oracle   |                   4.87 |         0    |              50    |        50   |              1    |   nan   |       nan |               0    |
| FD002 in-domain                             | rtf      |                  48.37 |       130    |               0    |       130   |              0    |   nan   |       nan |             100    |
| FD002 in-domain                             | fixed    |                   8.4  |         1.15 |             128.85 |       130   |             79.31 |   125   |       nan |               0.88 |
| FD002 in-domain                             | rul      |                   5.33 |         0.6  |             129.4  |       130   |             11.47 |    17.5 |         5 |               0.46 |
| FD002 in-domain                             | oracle   |                   4.86 |         0    |             130    |       130   |              1    |   nan   |       nan |               0    |
| FD004 in-domain                             | rtf      |                  40.66 |       124.5  |               0    |       124.5 |              0    |   nan   |       nan |             100    |
| FD004 in-domain                             | fixed    |                   8.39 |         1.52 |             122.98 |       124.5 |            115.37 |   130   |       nan |               1.22 |
| FD004 in-domain                             | rul      |                   4.7  |         0.92 |             123.58 |       124.5 |             19.14 |    25   |         3 |               0.74 |
| FD004 in-domain                             | oracle   |                   4.08 |         0    |             124.5  |       124.5 |              1    |   nan   |       nan |               0    |
| FD001->FD002 (normalised, tuned on target)  | rtf      |                  48.37 |       130    |               0    |       130   |              0    |   nan   |       nan |             100    |
| FD001->FD002 (normalised, tuned on target)  | fixed    |                   8.37 |         1.08 |             128.93 |       130   |             79.39 |   125   |       nan |               0.83 |
| FD001->FD002 (normalised, tuned on target)  | rul      |                   5.5  |         0.82 |             129.18 |       130   |             14.75 |    20   |         3 |               0.63 |
| FD001->FD002 (normalised, tuned on target)  | oracle   |                   4.86 |         0    |             130    |       130   |              1    |   nan   |       nan |               0    |
| FD001->FD004 (normalised, tuned on target)  | rtf      |                  40.67 |       124.5  |               0    |       124.5 |              0    |   nan   |       nan |             100    |
| FD001->FD004 (normalised, tuned on target)  | fixed    |                   8.47 |         1.82 |             122.68 |       124.5 |            114.39 |   130   |       nan |               1.47 |
| FD001->FD004 (normalised, tuned on target)  | rul      |                   5.07 |         0.5  |             124    |       124.5 |             41.59 |    45   |         1 |               0.4  |
| FD001->FD004 (normalised, tuned on target)  | oracle   |                   4.08 |         0    |             124.5  |       124.5 |              1    |   nan   |       nan |               0    |
| FD001->FD002 (normalised, tuned on FD001)   | rtf      |                  48.37 |       130    |               0    |       130   |              0    |   nan   |       nan |             100    |
| FD001->FD002 (normalised, tuned on FD001)   | fixed    |                   8.48 |         1.42 |             128.57 |       130   |             78.47 |   125   |       nan |               1.1  |
| FD001->FD002 (normalised, tuned on FD001)   | rul      |                  12.56 |        22    |             108    |       130   |              7.21 |    15   |         5 |              16.92 |
| FD001->FD002 (normalised, tuned on FD001)   | oracle   |                   4.86 |         0    |             130    |       130   |              1    |   nan   |       nan |               0    |
| FD001->FD004 (normalised, tuned on FD001)   | rtf      |                  40.67 |       124.5  |               0    |       124.5 |              0    |   nan   |       nan |             100    |
| FD001->FD004 (normalised, tuned on FD001)   | fixed    |                   8.44 |         1.65 |             122.85 |       124.5 |            115.25 |   130   |       nan |               1.33 |
| FD001->FD004 (normalised, tuned on FD001)   | rul      |                  26.11 |        74    |              50.5  |       124.5 |              7.04 |    15   |         5 |              59.44 |
| FD001->FD004 (normalised, tuned on FD001)   | oracle   |                   4.08 |         0    |             124.5  |       124.5 |              1    |   nan   |       nan |               0    |
| FD001->FD002 (raw sensors, tuned on target) | rtf      |                  48.37 |       130    |               0    |       130   |              0    |   nan   |       nan |             100    |
| FD001->FD002 (raw sensors, tuned on target) | fixed    |                   8.31 |         0.9  |             129.1  |       130   |             79.8  |   125   |       nan |               0.69 |
| FD001->FD002 (raw sensors, tuned on target) | rul      |                   6.88 |         1    |             129    |       130   |             51.83 |    65   |         1 |               0.77 |
| FD001->FD002 (raw sensors, tuned on target) | oracle   |                   4.86 |         0    |             130    |       130   |              1    |   nan   |       nan |               0    |
| FD001->FD004 (raw sensors, tuned on target) | rtf      |                  40.66 |       124.5  |               0    |       124.5 |              0    |   nan   |       nan |             100    |
| FD001->FD004 (raw sensors, tuned on target) | fixed    |                   8.34 |         1.3  |             123.2  |       124.5 |            116.3  |   130   |       nan |               1.04 |
| FD001->FD004 (raw sensors, tuned on target) | rul      |                   5.97 |         0    |             124.5  |       124.5 |             78.36 |    70   |         1 |               0    |
| FD001->FD004 (raw sensors, tuned on target) | oracle   |                   4.08 |         0    |             124.5  |       124.5 |              1    |   nan   |       nan |               0    |

### Breakeven table

| scenario                                    |   r_fixed_beats_rtf |   r_rul_beats_rtf |   r_rul_beats_fixed_5pct |   r_rul_worse_than_fixed |   r_rul_beats_fixed_robust |   gain_pct_at_r10 |   rul_win_share_at_r10 |   gain_pct_at_r5 |   rul_vs_oracle_gap_pct_at_r10 |
|:--------------------------------------------|--------------------:|------------------:|-------------------------:|-------------------------:|---------------------------:|------------------:|-----------------------:|-----------------:|-------------------------------:|
| FD001 in-domain                             |                 1.3 |              1.05 |                     1.1  |                   nan    |                       1.05 |             37.05 |                      1 |            36.98 |                          13.52 |
| FD002 in-domain                             |                 1.4 |              1.05 |                     1.1  |                   nan    |                       1.05 |             36.54 |                      1 |            34.96 |                           9.68 |
| FD004 in-domain                             |                 1.4 |              1.05 |                     1.15 |                   nan    |                       1.05 |             44.02 |                      1 |            41.65 |                          15.07 |
| FD001->FD002 (normalised, tuned on target)  |                 1.4 |              1.05 |                     1.15 |                   nan    |                       1.05 |             34.28 |                      1 |            33.86 |                          13.18 |
| FD001->FD004 (normalised, tuned on target)  |                 1.5 |              1.1  |                     1.15 |                   nan    |                       1.05 |             40.21 |                      1 |            36.34 |                          24.04 |
| FD001->FD002 (normalised, tuned on FD001)   |                 1.4 |              1.05 |                     1.15 |                     4.59 |                       1.05 |            -48.09 |                      0 |            -3.03 |                         158.47 |
| FD001->FD004 (normalised, tuned on FD001)   |                 1.5 |              1.15 |                     1.5  |                     2.14 |                       1.05 |           -209.26 |                      0 |           -77.32 |                         539.41 |
| FD001->FD002 (raw sensors, tuned on target) |                 1.4 |              1.15 |                     1.25 |                    46.54 |                       1.15 |             17.2  |                      1 |            17.28 |                          41.61 |
| FD001->FD004 (raw sensors, tuned on target) |                 1.4 |              1.2  |                     1.3  |                   nan    |                       1.15 |             28.48 |                      1 |            23.07 |                          46.17 |

### Bootstrap 95% CI of test RMSE (XGBoost, capped label)

| experiment                  |   n |   rmse |   lo95 |   hi95 |
|:----------------------------|----:|-------:|-------:|-------:|
| FD001>FD001|xgb|capped|norm | 100 |   15.3 |   12.7 |   18.3 |
| FD001>FD001|xgb|capped|raw  | 100 |   19.3 |   15.9 |   22.6 |
| FD001>FD002|xgb|capped|norm | 259 |   29.8 |   26.2 |   33.3 |
| FD001>FD002|xgb|capped|raw  | 259 |   43.3 |   40.3 |   46.3 |
| FD001>FD004|xgb|capped|norm | 248 |   35.6 |   32.4 |   38.6 |
| FD001>FD004|xgb|capped|raw  | 248 |   48   |   44.6 |   51.3 |
| FD002>FD002|xgb|capped|norm | 259 |   26.6 |   23   |   30   |
| FD002>FD002|xgb|capped|raw  | 259 |   27.8 |   24.7 |   31   |
| FD004>FD004|xgb|capped|norm | 248 |   27.1 |   24.3 |   29.8 |
| FD004>FD004|xgb|capped|raw  | 248 |   29.6 |   27   |   32.4 |
