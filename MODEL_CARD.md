# Model card — Project C: London household smart meters (forecasting, segmentation, price response)

Written 2026-10-14. Data: Low Carbon London smart-meter trial (UK Power Networks, London Datastore,
CC BY 4.0), 5,566 households, half-hourly, 2011-11..2014-02 — **real meter readings, but London
2012-2014, not Texas and not any company's current customers**. Not affiliated with UK Power
Networks, EDF Energy or Vistra. **No public leaderboard exists for these tasks.**
Protocol: [PROTOCOL.md](PROTOCOL.md) (time alignment decided by a diagnostic before freezing).

## C3 — response to dynamic price events (2013)
Design: event days were scattered at random through 2013 (UK Data Service user guide 7857), so
within the dToU group event vs non-event half-hours are comparable; the standard-tariff group
removes common shocks (difference-in-differences in logs, FE half-hour x weekend x month,
day-block bootstrap). 1,038 dToU vs 4,071 standard households.

| Effect on dToU consumption | Estimate | 95% CI |
|---|---|---|
| High price (67.2p vs 11.8p) | **-6.1%** | [-6.9, -5.2] |
| Low price (4.0p) | **+6.2%** | [+5.2, +7.3] |
| High, network-constraint days (low-high-low) | -5.3% | [-6.6, -3.9] |
| High, balancing events | -6.4% | [-7.4, -5.3] |
| Low, constraint days / balancing | +4.0% / +8.8% | [+2.8, +5.1] / [+7.1, +10.6] |
| 2 h before / after High | +0.7% / +0.9% | CIs include 0 |
| **Placebo 2012** (same schedule, before the tariff) | High +0.5%, Low +0.4% | both CIs include 0 |

Limits: the dToU group opted in (not a random sample of customers; the effect is for willing
participants); one supplier, one year, 2013 London; per-household responses are noisy.

## C2 — household segmentation (descriptive)
Weekday + weekend normalized daily shape, 5,097 households (2013). Pre-set rule chose **k = 2**
(every k >= 3 produced a cluster with ~1% of households, below the 2% minimum). Silhouette is low
(0.12): household shapes form a continuum more than distinct types.

| Cluster | Households | Weekday peak | Night share 00:30-07:00 | Median kWh/day | Median High-price response (dToU members) |
|---|---|---|---|---|---|
| 0 early evening | 2,742 | 18:30 | 14% | 8.2 | -2.7% (n=569) |
| 1 late evening | 2,355 | 21:00 | 20% | 8.5 | -5.5% (n=467) |

Response differs between clusters (Kruskal-Wallis p = 0.0003) — descriptive, not causal.
Stability: same-centroid assignment on H1 vs H2 2013 agrees for 89% of households (ARI 0.61).
The ~1% cluster at k = 3: 55 households using 54% of their energy 00:30-07:00 with a 00:30 peak and
twice the median consumption — consistent with Economy 7 / night storage heating (**inferred; the
data has no heating label**).

## C1 — day-ahead forecasts and the aggregation effect
3,455 standard-tariff households; fixed random portfolios (20 each of 1, 10, 100, 1000 households,
plus all). Issue 12:00 on D-1, data known through D-2. WAPE (median over portfolios of each size).
Validation 2013-01..09; holdout 2013-10..2014-02 (opened once).

| Run | Val primary | Val n1 | Val n100 | Val all | Hold primary | Hold n1 | Hold n100 | Hold n1000 | Hold all |
|---|---|---|---|---|---|---|---|---|---|
| N1 same half-hour last week | 8.95 | 64.7 | 11.88 | 7.22 | 7.38 | 58.3 | 10.80 | 5.96 | 5.37 |
| N2 mean of last 4 weeks | 8.86 | 56.5 | 10.95 | 7.63 | 7.42 | 49.5 | 9.83 | 6.42 | 6.02 |
| T1 LightGBM top-down | 7.21 | 47.6 | 9.34 | 5.97 | 5.69 | 42.9 | 8.27 | 4.62 | 4.19 |
| T5 T1 + trend | 7.04 | 47.5 | 9.24 | 5.76 | — | | | | |
| C2 Chronos-2 LoRA | 6.35 | 44.6 | 8.70 | 4.95 | — | | | | |
| **C1 Chronos-2 zero-shot (selected)** | **6.22** | 44.9 | 8.64 | 4.78 | **5.55** | 41.0 | 8.21 | 4.49 | **3.95** |
| TX2 oracle: actual temperature change, no doy | 7.19 | 47.6 | 9.36 | 5.90 | 6.09 | 42.9 | 8.68 | 5.01 | 4.59 |

Paired (daily, 7-day blocks), C1 minus other:

| vs | Validation all / n100 | Holdout all / n100 |
|---|---|---|
| T1 | -1.14 [-1.75, -0.61] / -0.68 [-1.04, -0.38] | -0.22 [-0.80, +0.33] / -0.02 [-0.36, +0.30] |
| N2 | — | -2.08 [-3.01, -1.08] / -1.45 [-1.90, -0.98] |

Findings
- **Aggregation is the dominant effect:** a single household is ~41-45% WAPE even for the best model;
  100 households ~8-9%; 3,455 households ~4-5%. Model choice matters far less than portfolio size.
- Zero-shot Chronos-2 was best on validation and still best on holdout, but its lead over LightGBM
  shrank to a difference whose CI includes 0 on holdout. Fine-tuning did not help here (unlike
  Project A); combining with LightGBM did not help.
- Weather: with absolute temperature, the oracle model got worse (first-fold temperatures were outside
  the training range, +8% bias); as a change vs D-2 it gained ~0.6 pp on validation over the same model without weather (T4). On holdout TX2
  is worse than T1, **but TX2 also drops day-of-year, so that comparison mixes two changes**
  (T4 = T1 without doy was not run on holdout). London homes mostly heat with gas, so weak
  temperature dependence is plausible.
- Pre-registered hypothesis falsified: dropping day-of-year (T4) hurt by 0.5 pp.

Limits: 2012-2013 data, short training history (first fold has 5 months); portfolio targets use
size x mean of available readings; meter data latency assumed (D-2).
