"""Paired C1 comparison: python -m src.lcl.compare A B [--holdout] [--sid all|n100|n1000]
Daily WAPE of the chosen series (sid 'all', or the mean over the 20 portfolios of a size), 7-day
moving-block bootstrap of A - B."""
import glob, sys
import numpy as np
import pandas as pd


def load(e, h=False):
    fs = [f for f in sorted(glob.glob(f"experiments/runs/*_{e}{'_HOLDOUT' if h else ''}/predictions.parquet")) if h or "HOLDOUT" not in f]
    return pd.read_parquet(fs[-1])


def daily(p, sid):
    p = p[p["y"].notna() & p["pred"].notna()]
    p = p[p["sid"] == "all"] if sid == "all" else p[p["sid"].str.startswith(sid + "_")]
    g = p.assign(ae=(p["pred"] - p["y"]).abs()).groupby(["sid", "D"])[["ae", "y"]].sum()
    return (100 * g["ae"] / g["y"]).groupby(level="D").mean()


def boot(d, block=7, n=2000, seed=0):
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(d) / block))
    s = [np.concatenate([d[i:i + block] for i in rng.integers(0, len(d) - block + 1, nb)])[:len(d)].mean() for _ in range(n)]
    return np.percentile(s, [2.5, 97.5])


def main():
    a, b = sys.argv[1], sys.argv[2]; h = "--holdout" in sys.argv
    sid = sys.argv[sys.argv.index("--sid") + 1] if "--sid" in sys.argv else "all"
    da, db = daily(load(a, h), sid), daily(load(b, h), sid)
    d = (da - db.reindex(da.index)).dropna().values
    lo, hi = boot(d)
    print(f"{a} - {b} [{sid}]: {d.mean():+.2f} pp, 95% CI [{lo:+.2f}, {hi:+.2f}], days {len(d)}")


if __name__ == "__main__":
    main()
