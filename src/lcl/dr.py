"""Module C3 (PROTOCOL.md): response of dToU households to price events.

d_t = log(mean ToU kWh_t) - log(mean Std kWh_t), households with >= 90% coverage in the period.
OLS of d_t on event dummies with fixed effects for half-hour-of-day x weekend x month (within
transformation), effects as exp(beta) - 1; day-block bootstrap (2,000 draws).
Specs: main (High, Low, 2 h before / after High); by event type (CM day = has both Low and High and
the High is enclosed by Low on that trial day; SB = all other event half-hours).
Falsification: 2012-04..2012-12 with the 2013 schedule moved 364 days earlier (same weekdays).
Output: reports/C3_price_response.json and printed table.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .panel import load

TZ = "Europe/London"


def schedule(shift_days=0):
    t = pd.read_parquet("data/processed/tariffs.parquet")
    loc = pd.DatetimeIndex(t["TariffDateTime"]).tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(t["Tariff"].values, index=loc.tz_convert("UTC").tz_localize(None))
    s = s[s.index.notna()]
    s = s[~s.index.duplicated()]
    s.index = s.index - pd.Timedelta(days=shift_days)
    return s


def event_frame(s):
    f = pd.DataFrame({"High": (s == "High").astype(float), "Low": (s == "Low").astype(float)}, index=s.index)
    hi = f["High"].values
    # 2 h (4 half-hours) before / after a High period, while the tariff is Normal
    pre = np.zeros(len(f)); post = np.zeros(len(f))
    for k in range(1, 5):
        pre[:-k] = np.maximum(pre[:-k], hi[k:]); post[k:] = np.maximum(post[k:], hi[:-k])
    nrm = (s == "Normal").values
    f["PreHigh"] = pre * nrm; f["PostHigh"] = post * nrm
    # CM vs SB by trial day (trial days start 05:00 local)
    day = (pd.DatetimeIndex(s.index).tz_localize("UTC").tz_convert(TZ) - pd.Timedelta(hours=5)).date
    g = pd.DataFrame({"day": day, "H": f["High"].values, "L": f["Low"].values})
    cm_days = set()
    for dd, x in g.groupby("day"):
        if x["H"].any() and x["L"].any():
            h = np.flatnonzero(x["H"].values); l = np.flatnonzero(x["L"].values)
            if (l < h.min()).any() and (l > h.max()).any():
                cm_days.add(dd)
    cm = np.array([d in cm_days for d in day])
    f["High_CM"] = f["High"] * cm; f["High_SB"] = f["High"] * ~cm
    f["Low_CM"] = f["Low"] * cm; f["Low_SB"] = f["Low"] * ~cm
    f["day"] = pd.DatetimeIndex(s.index).normalize()
    return f, len(cm_days)


def group_gap(a, b, period_cov):
    Y, times, hh, cov = load()
    t = pd.DatetimeIndex(times)
    m = (t >= a) & (t < b)
    tou = (cov[period_cov] >= 0.9) & cov["tou"]
    std = (cov[period_cov] >= 0.9) & ~cov["tou"]
    Ym = np.asarray(Y[m])
    ci = cov.index
    mt = np.nanmean(Ym[:, ci.get_indexer(cov.index[tou])], axis=1)
    ms = np.nanmean(Ym[:, ci.get_indexer(cov.index[std])], axis=1)
    return pd.Series(np.log(mt) - np.log(ms), index=t[m]), int(tou.sum()), int(std.sum())


def fit(d, X, cells):
    """Within-cell demeaning (FWL) then OLS."""
    def dm(v):
        mu = np.bincount(cells, weights=v) / np.bincount(cells)
        return v - mu[cells]
    yd = dm(d); Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    return np.linalg.lstsq(Xd, yd, rcond=None)[0]


def estimate(gap, ev, cols, n_boot=2000, seed=0):
    df = ev.join(gap.rename("d"), how="inner").dropna(subset=["d"])
    loc = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(TZ)
    slot = loc.hour * 2 + loc.minute // 30
    cells = pd.factorize(pd.Series(slot.astype(str)) + "_" + pd.Series((loc.dayofweek >= 5).astype(str)) + "_" + pd.Series(loc.month.astype(str)))[0]
    X = df[cols].values; d = df["d"].values
    b = fit(d, X, cells)
    days = df["day"].values; ud, inv = np.unique(days, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(ud))]
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ud), len(ud))
        idx = np.concatenate([rows_of[k] for k in pick])
        _, c2 = np.unique(cells[idx], return_inverse=True)
        bs.append(fit(d[idx], X[idx], c2))
    bs = np.array(bs)
    out = {}
    for j, c in enumerate(cols):
        lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        out[c] = {"pct": 100 * (np.exp(b[j]) - 1), "ci": [100 * (np.exp(lo) - 1), 100 * (np.exp(hi) - 1)],
                  "n_halfhours": int(X[:, j].sum())}
    return out


def main():
    res = {}
    s13 = schedule()
    ev13, ncm = event_frame(s13)
    gap13, ntou, nstd = group_gap("2013-01-01", "2014-01-01", "cov_2013")
    res["n_households"] = {"tou": ntou, "std": nstd}; res["cm_days"] = ncm
    res["main"] = estimate(gap13, ev13, ["High", "Low", "PreHigh", "PostHigh"])
    res["by_type"] = estimate(gap13, ev13, ["High_CM", "High_SB", "Low_CM", "Low_SB", "PreHigh", "PostHigh"])
    ev12, _ = event_frame(schedule(shift_days=364))
    gap12, n1, n2 = group_gap("2012-04-01", "2013-01-01", "cov_2012")
    res["placebo_2012"] = estimate(gap12, ev12, ["High", "Low", "PreHigh", "PostHigh"])
    res["placebo_2012_households"] = {"tou": n1, "std": n2}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/C3_price_response.json").write_text(json.dumps(res, indent=1, default=float))
    for spec in ["main", "by_type", "placebo_2012"]:
        print(spec)
        for k, v in res[spec].items():
            print(f"  {k:9s} {v['pct']:+6.2f}%  [{v['ci'][0]:+6.2f}, {v['ci'][1]:+6.2f}]  half-hours={v['n_halfhours']}")
    print(res["n_households"], "CM days", ncm, "placebo households", res["placebo_2012_households"])


if __name__ == "__main__":
    main()
