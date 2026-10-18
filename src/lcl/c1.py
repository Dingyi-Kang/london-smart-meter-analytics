"""Module C1 (PROTOCOL.md): day-ahead forecasts of household portfolios.

Series: fixed random portfolios (seed 0) of the 3,455-household standard-tariff panel, sizes
1/10/100/1000 (20 each) and all. Target per half-hour = size x mean over households with a reading
(so missing readings do not look like zero load). Local day D (Europe/London), issue 12:00 on D-1,
meter data known through the end of D-2.
Usage: python -m src.lcl.c1 EXP [...] [--holdout]
"""
import tempfile
import argparse, datetime as dt, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from .panel import load

TZ = "Europe/London"
SIZES = [1, 10, 100, 1000]
N_PORT = 20
BANK = pd.to_datetime(["2011-12-26", "2011-12-27", "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07", "2012-06-04",
                       "2012-06-05", "2012-08-27", "2012-12-25", "2012-12-26", "2013-01-01", "2013-03-29", "2013-04-01",
                       "2013-05-06", "2013-05-27", "2013-08-26", "2013-12-25", "2013-12-26", "2014-01-01"])
VAL = (["2013-01-01", "2013-04-01", "2013-07-01"], "2013-09-30")
HOLD = (["2013-10-01", "2013-11-01", "2013-12-01", "2014-01-01", "2014-02-01"], "2014-02-27")
DATA_START = "2012-07-01"
REG = Path("experiments/registry_C1.jsonl")
CACHE = Path("data/processed/c1_series.parquet")


def portfolios(hh_ids):
    rng = np.random.default_rng(0)
    P = {}
    for s in SIZES:
        for k in range(N_PORT):
            P[f"n{s}_{k:02d}"] = rng.choice(len(hh_ids), s, replace=False)
    P["all"] = np.arange(len(hh_ids))
    return P


def series():
    """Long frame: sid, size, t (GMT start), D (local day), slot, y, plus ERA5 temperature."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    Y, times, hh, cov = load()
    ids = cov.index[(cov["cov_c1"] >= 0.9) & ~cov["tou"]]
    t = pd.DatetimeIndex(times)
    m = (t >= DATA_START) & (t < "2014-02-28")
    Ym = np.asarray(Y[m][:, cov.index.get_indexer(ids)]); tm = t[m]
    loc = tm.tz_localize("UTC").tz_convert(TZ)
    D = loc.tz_localize(None).normalize(); slot = loc.hour * 2 + loc.minute // 30
    parts = []
    for sid, idx in portfolios(ids).items():
        y = np.nanmean(Ym[:, idx], axis=1) * len(idx) if len(idx) > 1 else Ym[:, idx[0]]
        parts.append(pd.DataFrame({"sid": sid, "size": len(idx), "t": tm, "D": D, "slot": slot, "y": y.astype("float32")}))
    df = pd.concat(parts, ignore_index=True)
    w = json.loads(Path("data/raw/weather/london_era5.json").read_text())["hourly"]
    temp = pd.Series(w["temperature_2m"], index=pd.to_datetime(w["time"])).resample("30min").interpolate()
    df["temp"] = temp.reindex(df["t"]).values
    df.to_parquet(CACHE, index=False)
    return df


def features(df):
    """Issue-safe features: same-slot lags D-2, D-7, D-14, D-21, D-28; 7-day level of D-8..D-2;
    observed mean temperature of D-2; calendar. Oracle: actual temperature of D (track ii)."""
    df = df.sort_values(["sid", "D", "slot"]).copy()
    key = df.set_index(["sid", "D", "slot"])["y"]
    key = key[~key.index.duplicated()]            # autumn clock change repeats 01:00-01:59 local
    for L in [2, 7, 14, 21, 28]:
        df[f"lag{L}"] = key.reindex(pd.MultiIndex.from_arrays([df["sid"], df["D"] - pd.Timedelta(days=L), df["slot"]])).values
    daily = df.groupby(["sid", "D"])["y"].mean()
    lvl = daily.groupby(level=0).transform(lambda s: s.rolling(7, min_periods=5).mean().shift(2))
    df["level"] = lvl.reindex(pd.MultiIndex.from_arrays([df["sid"], df["D"]])).values
    lvl2 = daily.groupby(level=0).transform(lambda s: s.rolling(7, min_periods=5).mean().shift(9))
    df["trend"] = df["level"] / lvl2.reindex(pd.MultiIndex.from_arrays([df["sid"], df["D"]])).values
    dt_ = df.groupby("D")["temp"].mean()
    df["temp_d2"] = dt_.reindex(df["D"] - pd.Timedelta(days=2)).values
    df["temp_d9"] = dt_.reindex(df["D"] - pd.Timedelta(days=9)).values
    df["temp_dmean"] = dt_.reindex(df["D"]).values                # oracle (actual day-D weather)
    df["dow"] = df["D"].dt.dayofweek; df["hol"] = df["D"].isin(BANK).astype(int)
    df["doy"] = df["D"].dt.dayofyear; df["logsize"] = np.log10(df["size"])
    df["xmas"] = ((df["D"].dt.month == 12) & (df["D"].dt.day >= 20) | (df["D"].dt.month == 1) & (df["D"].dt.day <= 3)).astype(int)
    return df


# ------------------------------------------------------------------ models
class Naive7:
    def fit(self, tr): return self
    def predict(self, te): return te["lag7"].values


class MeanWeeks:
    def fit(self, tr): return self
    def predict(self, te): return np.nanmean(te[["lag7", "lag14", "lag21", "lag28"]].values, axis=1)


class LGBMTopDown:
    """Global LightGBM over all portfolio series; target y / level (relative), L1 loss."""
    BASE = ["slot", "dow", "hol", "xmas", "doy", "logsize", "temp_d2"]
    def __init__(self, oracle=False, seeds=(0,), params=None, drop=(), delta=False, trend=False):
        self.oracle, self.seeds, self.drop, self.delta, self.trend = oracle, seeds, list(drop), delta, trend
        self.p = dict(objective="l1", learning_rate=0.03, n_estimators=1200, num_leaves=63, min_child_samples=100,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=16, verbose=-1)
        self.p.update(params or {})
    def _X(self, d):
        X = d[[c for c in self.BASE if c not in self.drop]].copy()
        for L in [2, 7, 14, 21, 28]:
            X[f"r{L}"] = d[f"lag{L}"] / d["level"]
        if self.trend:
            X["trend"] = d["trend"]; X["dtemp_obs"] = d["temp_d2"] - d["temp_d9"]
        if self.oracle and self.delta:           # change vs the observed D-2 mean: extrapolates with a relative target
            X["dtemp"] = d["temp_dmean"] - d["temp_d2"]; X["dtemp_slot"] = d["temp"] - d["temp_d2"]
        elif self.oracle:
            X["temp"] = d["temp"]; X["temp_dmean"] = d["temp_dmean"]
        return X
    def fit(self, tr):
        import lightgbm as lgb
        tr = tr[tr["level"] > 0]
        X, y = self._X(tr), tr["y"] / tr["level"]
        self.ms = [lgb.LGBMRegressor(**{**self.p, "random_state": s}).fit(X, y) for s in self.seeds]
        return self
    def predict(self, te):
        X = self._X(te)
        return np.mean([m.predict(X) for m in self.ms], axis=0) * te["level"].values


FULL = {}   # set in main(): the full feature frame (history for Chronos context)


class ChronosZero:
    """Chronos-2 zero-shot per series: context = half-hours up to the end of local day D-2 (last
    `ctx`), known covariates slot / dow / holiday; forecast D-1 and D, score D. GPU if available."""
    def __init__(self, ctx=2048, finetune=False, steps=1000, lr=1e-5, seed=0):
        self.ctx, self.ft, self.steps, self.lr, self.seed = ctx, finetune, steps, lr, seed
        self.pipe = None
    def fit(self, tr):
        if not self.ft:
            return self
        import torch
        from chronos import Chronos2Pipeline
        from chronos.chronos2.preprocess import from_data_frame
        last = tr["D"].max()
        data = FULL["df"]; data = data[data["D"] <= last].sort_values(["sid", "t"])
        data = pd.DataFrame({"item_id": data["sid"], "timestamp": data["t"], "target": data["y"].astype(float),
                             "slot": data["slot"], "dow": data["dow"], "hol": data["hol"]})
        inputs = from_data_frame(data, target_columns=["target"], prediction_length=96, known_covariates_names=["slot", "dow", "hol"])
        torch.manual_seed(self.seed)
        base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = base.fit(inputs, prediction_length=96, finetune_mode="lora", learning_rate=self.lr, num_steps=self.steps,
                             batch_size=32, context_length=self.ctx, output_dir=f"{tempfile.gettempdir()}/chronos_ft_C1/{last.date()}_s{self.seed}",
                             remove_printer_callback=True, save_strategy="no", seed=self.seed)
        self.pipe.model.eval()                    # fit() leaves train mode on (dropout)
        return self
    def predict(self, te):
        import torch
        from chronos import Chronos2Pipeline
        pipe = self.pipe or Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")
        df = FULL["df"]
        g = {sid: x.sort_values("t").reset_index(drop=True) for sid, x in df.groupby("sid")}
        out = pd.Series(np.nan, index=te.index)
        items = []
        for (sid, D), x in te.groupby(["sid", "D"]):
            h = g[sid]
            cut = int(np.searchsorted(h["D"].values, np.datetime64(D - pd.Timedelta(days=1))))   # first row of D-1
            end = int(np.searchsorted(h["D"].values, np.datetime64(D + pd.Timedelta(days=1))))
            past = h.iloc[max(0, cut - self.ctx):cut]; fut = h.iloc[cut:end]
            key = f"{sid}|{D.date()}"
            items.append((key, past, fut, x))
        byH = {}
        for it in items:
            byH.setdefault(len(it[2]), []).append(it)
        chunks = [lst[i:i + 512] for lst in byH.values() for i in range(0, len(lst), 512)]
        for ch in chunks:
            P = pd.concat([pd.DataFrame({"item_id": k, "timestamp": p["t"].values, "target": p["y"].values.astype(float),
                                         "slot": p["slot"].values, "dow": p["dow"].values, "hol": p["hol"].values}) for k, p, f, x in ch])
            F = pd.concat([pd.DataFrame({"item_id": k, "timestamp": f["t"].values, "slot": f["slot"].values,
                                         "dow": f["dow"].values, "hol": f["hol"].values}) for k, p, f, x in ch])
            H = max(len(f) for _, _, f, _ in ch)
            r = pipe.predict_df(P, future_df=F, prediction_length=H, quantile_levels=[0.5], id_column="item_id",
                                timestamp_column="timestamp", target="target")
            col = "predictions" if "predictions" in r.columns else "0.5"
            r = r.set_index(["item_id", "timestamp"])[col]
            for k, p, f, x in ch:
                out.loc[x.index] = r.loc[k].reindex(pd.DatetimeIndex(x["t"])).values
        return out.values


EXPERIMENTS = {
    "N1_naive_d7": dict(model=Naive7, hypothesis="Same half-hour one week earlier is the floor; household-level WAPE expected > 60%, all-households < 10%."),
    "N2_mean_4weeks": dict(model=MeanWeeks, hypothesis="Averaging the last 4 same weekdays beats N1 at every size (noise reduction), most at small sizes."),
    "T1_lgbm_topdown": dict(model=lambda: LGBMTopDown(), hypothesis="Global LightGBM on relative portfolio load (lags/level, calendar, observed D-2 temperature) beats N2, mainly at sizes >= 100 where the signal is predictable."),
    "TX_lgbm_topdown_oracle": dict(model=lambda: LGBMTopDown(oracle=True), hypothesis="ORACLE (actual day-D temperature): upper bound on what a weather forecast could add over T1; not a forecast."),
    # registered after T1 / TX: TX lost in fold 0 (+8% bias; Q1 2013 temperatures outside the Aug-Dec 2012 training range)
    "T4_lgbm_topdown_nodoy": dict(model=lambda: LGBMTopDown(drop=["doy"]),
                                  hypothesis="Dropping day-of-year beats T1: with < 1 year of training data doy cannot be learned and extrapolates badly (T1 is +3-4% biased in folds 0-1)."),
    "TX2_oracle_delta_nodoy": dict(model=lambda: LGBMTopDown(oracle=True, delta=True, drop=["doy"]),
                                   hypothesis="ORACLE with temperature as change vs D-2 (relative, like the target) and no doy beats T4: shows how much a perfect weather forecast could add. Not a forecast."),
    # registered after T4 (falsified: doy helps) / TX2
    "T5_lgbm_topdown_trend": dict(model=lambda: LGBMTopDown(trend=True),
                                  hypothesis="Adding the week-over-week level trend and the observed temperature change (D-2 vs D-9) to T1 reduces its seasonal bias (+3-4% in folds 0-1) and beats T1."),
    "C1_chronos2_zero": dict(model=lambda: ChronosZero(),
                             hypothesis="Zero-shot Chronos-2 (context to D-2, calendar covariates) is competitive with T1 at sizes >= 100 (within 0.5 pp) and better than N2 everywhere."),
    # registered after C1 (zero-shot, primary 6.22) and E1 (no gain)
    "C2_chronos2_lora": dict(model=lambda: ChronosZero(finetune=True),
                             hypothesis="LoRA fine-tuning Chronos-2 per fold on the 81 portfolio series (1000 steps, lr 1e-5, eval mode, 1 seed; settings from Project A) beats zero-shot C1 by >= 0.2 pp primary, as fine-tuning did in Project A."),
}


# ------------------------------------------------------------------ evaluation
def summarize(p):
    p = p[p["y"].notna() & p["pred"].notna()]
    w = p.groupby("sid").apply(lambda g: (g["pred"] - g["y"]).abs().sum() / g["y"].abs().sum(), include_groups=False)
    size = p.groupby("sid")["size"].first()
    m = {f"wape_med_n{s}": float(100 * w[size == s].median()) for s in SIZES}
    m["wape_all"] = float(100 * w["all"])
    m["primary"] = float(np.mean([m["wape_med_n100"], m["wape_med_n1000"], m["wape_all"]]))
    h = p[p["size"] == 1]
    m["mae_household_kwh"] = float((h["pred"] - h["y"]).abs().mean())
    return m


def git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def folds(spec):
    starts, end = spec
    ends = [pd.Timestamp(s) - pd.Timedelta(days=1) for s in starts[1:]] + [pd.Timestamp(end)]
    return [(pd.Timestamp(s), e) for s, e in zip(starts, ends)]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("exps", nargs="+"); ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    if a.holdout:
        import yaml
        cfg = Path("configs/final_C1.yaml")
        if not cfg.exists():
            sys.exit("Holdout locked: create configs/final_C1.yaml first.")
        allowed = yaml.safe_load(cfg.read_text())["holdout_set"]
        done = {json.loads(l)["experiment"] for l in REG.open() if json.loads(l)["phase"] == "holdout"} if REG.exists() else set()
        bad = [e for e in a.exps if e not in allowed or e in done]
        if bad:
            sys.exit(f"Not allowed on holdout: {bad}")
    df = features(series())
    FULL["df"] = df
    fl = folds(HOLD if a.holdout else VAL)
    REG.parent.mkdir(exist_ok=True)
    for name in a.exps:
        spec = EXPERIMENTS[name]; t0 = time.time()
        rid = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}{'_HOLDOUT' if a.holdout else ''}"
        rec = {"run_id": rid, "experiment": name, "phase": "holdout" if a.holdout else "validation", "hypothesis": spec["hypothesis"], "git": git()}
        try:
            out = []
            for i, (S, E) in enumerate(fl):
                tr = df[(df["D"] <= S - pd.Timedelta(days=2)) & df["y"].notna() & df["lag28"].notna()]
                te = df[(df["D"] >= S) & (df["D"] <= E)].copy()
                m = spec["model"]().fit(tr)
                te["pred"] = m.predict(te); te["fold"] = i
                out.append(te[["sid", "size", "t", "D", "slot", "y", "pred", "fold"]])
            p = pd.concat(out, ignore_index=True)
            rdir = Path("experiments/runs") / rid; rdir.mkdir(parents=True, exist_ok=True)
            p.to_parquet(rdir / "predictions.parquet", index=False)
            mt = summarize(p); rec.update(status="ok", metrics=mt, seconds=round(time.time() - t0, 1))
            print(f"{name:28s} primary={mt['primary']:.2f}% | n1 {mt['wape_med_n1']:.1f} n10 {mt['wape_med_n10']:.1f} "
                  f"n100 {mt['wape_med_n100']:.2f} n1000 {mt['wape_med_n1000']:.2f} all {mt['wape_all']:.2f} | hh MAE {mt['mae_household_kwh']:.4f}", flush=True)
        except Exception as e:
            rec.update(status="failed", error=repr(e)); print(name, "FAILED", repr(e)); raise
        finally:
            with REG.open("a") as f:
                f.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    main()
