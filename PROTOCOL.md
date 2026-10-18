# PROTOCOL — Project C: household smart-meter forecasting, segmentation and price response (London)

Frozen 2026-10-14 before any model was fit (data inspection only:
coverage, timestamps, tariff alignment diagnostic below). Changes only as dated amendments.
Personal study project on public data; not affiliated with UK Power Networks, EDF Energy or Vistra.

## Data
- **Low Carbon London smart-meter data** (UK Power Networks, London Datastore, CC BY 4.0):
  5,566 households, half-hourly kWh, 2011-11..2014-02; `tou` = dynamic time-of-use group
  (1,123 households, dToU tariff during 2013 only) vs standard flat tariff (4,443).
  **Real meter readings** (unlike the ERCOT profiles of Project A). Not US, and 2012-2014.
- **dToU signal schedule** (Tariffs.xlsx): High 67.20p / Low 3.99p / Normal 11.76p per kWh,
  per half-hour of 2013; announced to customers one day ahead.
- **Group assignment was opt-in** (UK Data Service user guide 7857, section 2.1: "group N was not a
  true control"). Price **event days were scattered randomly** through 2013 (same guide, 2.2).
  So only within-group, event-vs-non-event contrasts are interpreted causally.
- Weather: Open-Meteo ERA5 hourly temperature for central London (reanalysis = actual weather).
- **Time alignment (decided from a diagnostic before freezing):** meter timestamps are interval
  *end* times in GMT; the tariff schedule lists interval *start* times in London local clock time.
  Evidence: the ToU-minus-Std consumption residual responds most strongly to the schedule shifted
  by -0.5 h in BST months and +0.5 h in GMT months (one hour apart, as this convention implies).
  All modules use interval start in GMT: `start = ts - 30 min`.

## Module C1 — day-ahead forecasting and the aggregation effect
- **Panel:** standard-tariff households with >= 90% of half-hours present 2012-07-01..2014-02-27
  (3,455). ToU households excluded (their 2013 load reacts to prices).
- **Issue:** 12:00 London time on D-1; target: 48 half-hours of local day D (GMT starts, 46/50 on
  clock-change days). **Availability:** meter data known through the end of D-2 (daily collection).
- **Portfolios (fixed before modelling, seed 0):** sizes 1, 10, 100, 1000 (20 random portfolios
  each) and all 3,455. Target = summed kWh.
- **Weather tracks:** (i) *no target-day weather* (forecast-valid); (ii) *actual ERA5 temperature on
  day D* = **oracle**, an upper bound on what a weather forecast could add, never called a forecast.
- **Validation:** 3 quarterly folds, target days 2013-01-01..2013-09-30; **holdout**
  2013-10-01..2014-02-27 (one fold per month boundary set in code), opened once.
  Refit at each fold start on data <= S-2.
- **Metrics:** per portfolio WAPE over the fold days; reported as the median over the 20 portfolios
  of each size (plus all-households WAPE). Household level (size 1) also MAE. Primary for selection:
  mean of the size-100 and size-1000 medians and the all-households WAPE.
- **Models:** seasonal naive (D-7 same half-hour), mean of last 4 same weekdays, LightGBM on
  households summed bottom-up, LightGBM directly on portfolio sums (top-down), Chronos-2 zero-shot on
  portfolio sums; weather variants only on track (ii) and labelled.
- **Statistics:** paired moving-block bootstrap (7-day blocks) on the daily all-households absolute
  error (and per size where reported).
- **Budget:** 25 validation runs, all logged; selection rule as in earlier projects (lowest primary;
  simpler model if the 95% CI of the gain includes 0).

## Module C2 — household segmentation (descriptive)
- All households with >= 90% coverage in 2013. Representation: mean daily load shape (half-hour
  value / daily mean) for weekdays and weekends (96 values), plus log mean daily kWh reported
  separately. KMeans k = 2..10; rule fixed now: highest silhouette among k with min ARI across 10
  seeds >= 0.9 and smallest cluster >= 2% of households. Stability: assignment agreement between
  fits on 2013-01..06 and 2013-07..12.

## Module C3 — response to dynamic price events (2013)
- **Estimand:** average change in ToU households' consumption during High and during Low
  half-hours, relative to Normal half-hours, net of common shocks. Because event timing was random,
  the contrast is identified within the ToU group; the Std group removes weather/calendar shocks
  common to London households (difference-in-differences in logs).
- **Estimator (pre-specified):** d_t = log(mean ToU kWh_t) - log(mean Std kWh_t) per half-hour
  (households with >= 90% coverage in 2013); regress d_t on High_t, Low_t and fixed effects for
  half-hour-of-day x weekday/weekend x month. Effect in % = exp(beta) - 1. Also: the 2 h before and
  after High periods (load shifting) and by event type (CM: low-high-low days; SB: others).
- **Inference:** day-block bootstrap (resample whole days, 2,000 draws).
- **Falsification:** the same regression on 2012-04..2012-12 with the 2013 schedule placed on the
  same weekday-aligned dates one year earlier (before the tariff existed): effects must be ~0.
- **Heterogeneity:** per-household responsiveness (each ToU household vs the Std mean), described
  by C2 cluster and consumption level. Descriptive only (opt-in group, no claim about other
  populations).

## Honesty rules
- No public leaderboard exists for these tasks; internal comparisons only.
- Results describe 2012-2014 London households on one supplier's trial; no claims about Texas or
  any company's customers. C3 effects are for an opt-in group.

## Amendments
