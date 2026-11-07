# Predictive Engine Maintenance: findings and recommendations

*Hypothetical airline scenario on NASA's simulated C-MAPSS data. Costs are illustrative assumptions, so no savings are claimed for any real fleet; the results below are simulated comparisons. Decision support only. All numbers come from `results/` and `reports/results_tables.md` of the author's own run.*

## Results (Section 10 of the case study, completed)

**Best in-domain model and score.** XGBoost trained on the 125-cycle-capped label, with per-operating-condition normalised sensors plus 10- and 30-cycle causal rolling features, is best on every dataset. On the official test engines (final cycle, scored against the true uncapped RUL):

| Dataset | RMSE (95% CI) | MAE | NASA score | Overestimation rate | Naive baseline RMSE | Linear regression RMSE |
|---|---|---|---|---|---|---|
| FD001 | 15.3 (12.7–18.3) | 11.4 | 368 | 57% | 40.2 | 17.4 |
| FD002 | 26.6 (23.0–30.0) | 17.8 | 10,997 | 41% | 41.6 | 29.3 |
| FD004 | 27.1 (24.3–29.8) | 19.9 | 4,703 | 47% | 64.0 | 30.9 |

Three modelling choices mattered more than the model family. The cap (tested both ways, as the brief asks) cuts FD001 RMSE from 26.0 to 15.3 for XGBoost; the uncapped label is worse on all three datasets. The "norm" feature set (per-condition z-scores, plus a 30-cycle rolling mean, without raw operating settings) beats the "raw" set everywhere: 19.3 → 15.3 on FD001 (a single condition, so that gain comes from the extra window and dropping the settings, not regime correction), 27.8 → 26.6 on FD002 and 29.6 → 27.1 on FD004. XGBoost beats linear regression by 2–4 cycles RMSE, a modest gain. Only on FD002 does raw XGBoost post a better NASA score (9,008) than the normalised model (10,997); RMSE and MAE still favour normalised. An LSTM was not attempted: the brief allows it only if justified, and the dominant errors found below (the cap, regime shift, fault mode) are not problems a sequence architecture would obviously fix. I did not test that, and hyperparameters were set once with no tuning.

**Cross-dataset performance drop (trained on FD001, tested on other subsets).** The expected finding holds.

| Test set | FD001-trained, raw sensors | FD001-trained, normalised | In-domain model |
|---|---|---|---|
| FD002 | 43.3 RMSE (+63%) | 29.8 (+12%) | 26.6 |
| FD004 | 48.0 RMSE (+77%) | 35.6 (+31%) | 27.1 |

Without condition normalisation the transfer is poor, and for linear regression it is unusable: linear regression on raw FD001 features reaches RMSE of about 35,000 on FD002 and FD004 because raw operating-regime values lie far outside anything it saw. Normalisation recovers most of the FD002 gap; the remaining +12% is not statistically distinguishable from in-domain (the 95% confidence intervals, 26.2–33.3 and 23.0–30.0, overlap). FD004 stays clearly worse (+31%, intervals 32.4–38.6 and 24.3–29.8 do not overlap). FD004 has two fault modes and FD001 has one, and the FD001 model overestimates RUL badly near the end of life on FD004 (bias +20 cycles and 98% overestimation in the last 10 cycles). That is consistent with the second fault mode being invisible to the model, but I did not test the mechanism directly.

**Where the model fails.**
- *Lifecycle stage:* in-domain RMSE is 3.5–6 cycles in the last 10 cycles before failure, 7.7–10 in cycles 11–30, and about 17–20 when 31–125 cycles remain.
- *Direction of error:* in the last 30 cycles the model overestimates (too optimistic) 64–80% of the time in-domain. That is the costly direction, which is why the policy needs a safety threshold rather than "service at predicted zero".
- *The cap:* the largest single errors on test engines are long-life engines (true RUL 170–195 on FD004) predicted at about 100–120 because of the 125 cap. In FD001, the 11 engines with true RUL above 125 have mean error −24.0 cycles versus +2.9 for the rest. These are early-direction errors and are cheap in the policy simulation.
- *Operating condition:* in-domain error is flat across the six condition clusters (RMSE 15.4–15.8 each, all cycles), so after normalisation no regime is a weak spot.
- *Cross-dataset:* errors are worst late in life on FD004 (RMSE 25.3 in the last 10 cycles versus 5.8 in-domain).

**Breakeven cost ratio** (r = failure cost / planned-maintenance cost; costs compared as cost per 1000 engine-cycles, because a planned service ends a lifecycle early). Each policy parameter is tuned on one half of the engines and scored on the other half, over 40 repeated splits. The fixed interval is also tuned (best T for the fleet, about 125–130 cycles), so it is a strong benchmark; real fixed schedules may be better or worse tuned than that.

| Scenario | Best fixed interval beats run-to-failure from r ≈ | RUL policy beats run-to-failure from r ≈ | RUL cheaper than fixed by ≥5% from r ≈ | RUL policy saving vs fixed at r = 10 | RUL stops beating fixed at r ≈ |
|---|---|---|---|---|---|
| FD001 in-domain | 1.3 | 1.05 | 1.10 | 37% | never (r ≤ 50) |
| FD002 in-domain | 1.4 | 1.05 | 1.10 | 37% | never |
| FD004 in-domain | 1.4 | 1.05 | 1.15 | 44% | never |
| FD001 model on FD002, threshold re-tuned on FD002 | 1.4 | 1.05 | 1.15 | 34% | never |
| FD001 model on FD004, threshold re-tuned on FD004 | 1.5 | 1.10 | 1.15 | 40% | never |
| FD001 model on FD002, **threshold tuned on FD001** | 1.4 | 1.05 | 1.15 | **−48%** (worse) | **4.6** |
| FD001 model on FD004, **threshold tuned on FD001** | 1.5 | 1.15 | 1.50 | **−209%** (worse) | **2.1** |

At r = 10 in-domain, cost per 1000 cycles is 4.7–5.5 for the RUL policy against 8.4–8.8 for the best fixed interval, 4.1–4.9 for a perfect-RUL oracle and 40–48 for run-to-failure. The RUL policy lands within 10–15% of the oracle, wastes about 8–19 cycles of life per service versus about 79–115 for the fixed interval, and has 0.5–1.1% unplanned failures versus 0.9–1.5%. When the FD001 threshold is moved unchanged to FD002 and FD004, unplanned failures jump to 17% and 59%, which is what destroys the advantage.

**Recommendation to the planning manager.**
1. A sensor-based RUL threshold policy is worth pursuing. In this simulation it is cheaper than a well-tuned fixed interval for any failure-to-planned cost ratio above about 1.1–1.15 when it is calibrated on the fleet it serves, and the advantage is large (34–44% at r = 10), mostly from servicing about 70–95 cycles closer to the end of life.
2. The advantage depends on calibration to the fleet. A model and service threshold validated on one operating profile must not be moved to another. When the threshold was carried over from FD001, the RUL policy lost to the fixed interval above r ≈ 2–5 because of unplanned failures. Before any new fleet or operating regime, the threshold must be re-tuned on that fleet's own run-to-failure data.
3. Per-condition normalisation is required for any multi-regime fleet; with the raw feature set the FD001 model's transfer error is 63–77% worse than in-domain.
4. Use a conservative threshold (the tuned values were roughly 15–25 predicted cycles remaining, with 3–5 consecutive readings below it), because the model errs optimistic late in life.
5. Treat this as decision support for engineers. It is not a basis for relaxing regulatory maintenance intervals.

## Honest caveats
- Simulated data and assumed costs; only the sensitivity analysis is meaningful, not any single saving number.
- The policy simulation uses each run-to-failure train-split engine as one lifecycle with the model's out-of-fold (or cross-dataset) predictions, so predictions never come from a model that saw that engine. It assumes a serviced engine is renewed to as-new and ignores spares, hangar slots and schedules.
- The official test engines stop before failure, so they cannot be used for the cost simulation; they are used for accuracy metrics only (one final-cycle row per engine, so confidence intervals are wide: about ±3 cycles).
- RUL here is cycles to simulated failure, not a probability of an in-service event.
- "Tuned on target" scenarios are best-case for the model and for the fixed interval alike, since both get their parameters from the evaluated fleet's lifetimes.
- Small numeric differences between machines and library versions are normal for XGBoost; conclusions were stable across two independent runs.
- Unit ids repeat across datasets and splits, so the case-study DDL's single-column primary key was changed to a (dataset, split, unit_id) composite. No other schema change.
- Raw files come from a GitHub mirror of the NASA release. Row counts match the published files.