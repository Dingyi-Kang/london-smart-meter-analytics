"""Module C2 (PROTOCOL.md): household segmentation by daily load shape (2013).

Representation: mean weekday and weekend daily shape (half-hour value / that day-type's mean),
96 values, local clock half-hours; households with >= 90% coverage in 2013 (both tariff groups;
the dToU group's shape includes its price response, reported by group).
KMeans k = 2..10 (20 inits). Rule (fixed in the protocol): highest silhouette among k with min ARI
across 10 seeds >= 0.9 and smallest cluster >= 2% of households. Stability: fit on 2013-01..06 vs
2013-07..12, agreement of assignments (ARI). Also links C3: per-household High-price response by
cluster (descriptive). Outputs reports/C2_*.{md,csv,json}, reports/figures/C2_centroids.png
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from .panel import load
from .dr import schedule

TZ = "Europe/London"


def shapes(a, b, ids):
    Y, times, hh, cov = load()
    t = pd.DatetimeIndex(times); m = (t >= a) & (t < b)
    X = np.asarray(Y[m][:, cov.index.get_indexer(ids)])
    loc = t[m].tz_localize("UTC").tz_convert(TZ)
    slot = np.asarray(loc.hour * 2 + loc.minute // 30); we = np.asarray(loc.dayofweek >= 5)
    rows = []
    for w in [False, True]:
        S = np.vstack([np.nanmean(X[(slot == s) & (we == w)], axis=0) for s in range(48)]).T   # hh x 48
        rows.append(S / S.mean(axis=1, keepdims=True))
    F = np.hstack(rows)
    mean_kwh = np.nanmean(X, axis=0) * 48
    return F, mean_kwh


def responsiveness(ids_tou):
    """Per ToU household: log(kWh/Std mean) during High minus during Normal, same local half-hours."""
    Y, times, hh, cov = load()
    t = pd.DatetimeIndex(times); m = (t >= "2013-01-01") & (t < "2014-01-01")
    std = cov.index[(cov["cov_2013"] >= 0.9) & ~cov["tou"]]
    X = np.asarray(Y[m]); tm = t[m]
    sm = np.nanmean(X[:, cov.index.get_indexer(std)], axis=1)
    s = schedule().reindex(tm)
    loc = tm.tz_localize("UTC").tz_convert(TZ); slot = np.asarray(loc.hour * 2 + loc.minute // 30)
    hi = (s == "High").values; nm = (s == "Normal").values
    hs = np.unique(slot[hi]); base = nm & np.isin(slot, hs)
    R = np.log((X[:, cov.index.get_indexer(ids_tou)] + 0.01) / (sm[:, None] + 0.01))
    return 100 * (np.exp(np.nanmean(R[hi], axis=0) - np.nanmean(R[base], axis=0)) - 1)


def main():
    Y, times, hh, cov = load()
    ids = cov.index[cov["cov_2013"] >= 0.9]
    F, kwh = shapes("2013-01-01", "2014-01-01", ids)
    ok = np.isfinite(F).all(1) & (kwh > 0.5)
    ids, F, kwh = ids[ok], F[ok], kwh[ok]
    res = {}
    for k in range(2, 11):
        km = KMeans(k, n_init=20, random_state=0).fit(F)
        seeds = [KMeans(k, n_init=20, random_state=s).fit(F).labels_ for s in range(1, 11)]
        res[k] = {"silhouette": float(silhouette_score(F, km.labels_, sample_size=3000, random_state=0)),
                  "ari_seeds_min": float(min(adjusted_rand_score(km.labels_, l) for l in seeds)),
                  "min_share": float(np.bincount(km.labels_).min() / len(F))}
    okk = [k for k, r in res.items() if r["ari_seeds_min"] >= 0.9 and r["min_share"] >= 0.02]
    k = max(okk, key=lambda k: res[k]["silhouette"]) if okk else 2
    km = KMeans(k, n_init=20, random_state=0).fit(F)
    lab = km.labels_
    F1, _ = shapes("2013-01-01", "2013-07-01", ids); F2, _ = shapes("2013-07-01", "2014-01-01", ids)
    good = np.isfinite(F1).all(1) & np.isfinite(F2).all(1)
    l1 = km.predict(F1[good]); l2 = km.predict(F2[good])
    per_ari = float(adjusted_rand_score(l1, l2)); per_same = float((l1 == l2).mean())
    refit_ari = float(adjusted_rand_score(KMeans(k, n_init=20, random_state=0).fit(F1[good]).labels_,
                                          KMeans(k, n_init=20, random_state=0).fit(F2[good]).labels_))
    tou = cov.loc[ids, "tou"].values
    out = pd.DataFrame({"cluster": lab, "tou": tou, "kwh_day": kwh}, index=ids)
    resp = pd.Series(responsiveness(ids[tou]), index=ids[tou])
    out["high_response_pct"] = resp
    loc_slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    peak = {c: loc_slots[int(np.argmax(km.cluster_centers_[c][:48]))] for c in range(k)}
    summ = out.groupby("cluster").agg(n=("tou", "size"), share_tou=("tou", "mean"), kwh_day_median=("kwh_day", "median"),
                                      high_resp_median=("high_response_pct", "median"), high_resp_n=("high_response_pct", "count"))
    summ["weekday_peak"] = pd.Series(peak)
    summ["peak_to_mean_weekday"] = [float(km.cluster_centers_[c][:48].max()) for c in range(k)]
    summ["night_share_0030_0700"] = [float(km.cluster_centers_[c][1:14].sum() / 48) for c in range(k)]
    Path("reports").mkdir(exist_ok=True); Path("reports/figures").mkdir(exist_ok=True)
    out.to_csv("reports/C2_clusters.csv")
    json.dump({"chosen_k": k, "scores": res, "period_ari_assign": per_ari, "period_same_share": per_same,
               "period_ari_refit": refit_ari, "summary": summ.reset_index().to_dict("records")},
              open("reports/C2_summary.json", "w"), indent=1, default=float)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(48) / 2
    for c in range(k):
        ax[0].plot(x, km.cluster_centers_[c][:48], label=f"{c} (n={summ.loc[c, 'n']})"); ax[1].plot(x, km.cluster_centers_[c][48:])
    ax[0].set_title("Weekday shape"); ax[1].set_title("Weekend shape"); ax[0].set_ylabel("kWh / day mean")
    for a in ax: a.set_xlabel("local hour")
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig("reports/figures/C2_centroids.png", dpi=120)
    print("k", k, {kk: {a: round(b, 3) for a, b in v.items()} for kk, v in res.items()})
    print("period: assign ARI", round(per_ari, 3), "same share", round(per_same, 3), "refit ARI", round(refit_ari, 3))
    print(summ.round(3).to_string())
    from scipy.stats import kruskal
    g = [out.loc[out.cluster == c, "high_response_pct"].dropna().values for c in range(k)]
    g = [x for x in g if len(x) >= 10]
    print("Kruskal-Wallis on High response across clusters:", kruskal(*g))


if __name__ == "__main__":
    main()
