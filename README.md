# London household smart meters: forecasting, segmentation and price response (Project C)

Status: **complete (holdout opened once, see [MODEL_CARD.md](MODEL_CARD.md))**; public data.
Data: Low Carbon London (UK Power Networks via London Datastore, CC BY 4.0) — London 2012-2014.
Not affiliated with UK Power Networks, EDF Energy or Vistra. **No public leaderboard exists.**

| Module | Question |
|---|---|
| C1 | How accurate are day-ahead forecasts for portfolios of 1 to 3,455 households? |
| C2 | Which household load-shape segments exist, and do they respond differently to prices? |
| C3 | How much do dynamic-tariff households change consumption at High / Low prices? |

## Layout
```
src/lcl/ingest.py, panel.py   raw CSV -> parquet -> half-hour x household matrix (time alignment)
src/lcl/c1.py                 C1 series, features, models (naive, LightGBM, Chronos-2), backtest
src/lcl/compare.py, combine.py paired bootstrap, equal-weight combos
src/lcl/segment.py            C2 clustering;  src/lcl/dr.py  C3 difference-in-differences
```

## Reproduce
```bash
PY=~/ML_forecast/.venv/bin/python
$PY -I -c "import sys; sys.path.insert(0,'.'); from src.lcl.ingest import main; main()"
$PY -m src.lcl.panel && $PY -m src.lcl.dr && $PY -m src.lcl.segment
$PY -m src.lcl.c1 N2_mean_4weeks T1_lgbm_topdown C1_chronos2_zero
```
Attribution: contains data from UK Power Networks' Low Carbon London project (CC BY 4.0).
