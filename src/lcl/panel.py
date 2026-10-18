"""Wide half-hourly matrix (PROTOCOL.md time alignment: start = ts - 30 min, GMT).
Outputs data/processed/Y.npy (float32, time x household, NaN = missing), index files, coverage.
Off-grid timestamps (not :00/:30, ~0.002% of rows) are dropped; duplicates averaged."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

P = Path("data/processed")
START, END = "2011-11-01", "2014-03-01"


def build():
    con = duckdb.connect()
    d = con.execute(f"""select LCLid, any_value(tou) tou, ts - interval 30 minute as t, avg(kwh) kwh
        from '{P}/hh.parquet' where minute(ts) in (0, 30) group by LCLid, ts""").df()
    hh = d.groupby("LCLid")["tou"].first().sort_index()
    times = pd.date_range(START, END, freq="30min", inclusive="left")
    ti = times.get_indexer(pd.DatetimeIndex(d["t"]))
    hi = hh.index.get_indexer(d["LCLid"])
    ok = ti >= 0
    Y = np.full((len(times), len(hh)), np.nan, dtype=np.float32)
    Y[ti[ok], hi[ok]] = d["kwh"].values[ok]
    np.save(P / "Y.npy", Y)
    pd.Series(times).to_frame("t").to_parquet(P / "Y_times.parquet")
    hh.rename("tou").to_frame().to_parquet(P / "Y_households.parquet")
    def cov(a, b):
        m = (times >= a) & (times < b)
        return (~np.isnan(Y[m])).mean(0)
    c = pd.DataFrame({"tou": hh.values, "cov_c1": cov("2012-07-01", "2014-02-28"), "cov_2013": cov("2013-01-01", "2014-01-01"),
                      "cov_2012": cov("2012-04-01", "2013-01-01")}, index=hh.index)
    c.to_parquet(P / "coverage.parquet")
    print(Y.shape, json.dumps({"c1_panel_std": int(((c.cov_c1 >= 0.9) & ~c.tou).sum()),
                               "c3_2013_tou": int(((c.cov_2013 >= 0.9) & c.tou).sum()),
                               "c3_2013_std": int(((c.cov_2013 >= 0.9) & ~c.tou).sum())}))


def load():
    Y = np.load(P / "Y.npy", mmap_mode="r")
    return Y, pd.read_parquet(P / "Y_times.parquet")["t"], pd.read_parquet(P / "Y_households.parquet"), pd.read_parquet(P / "coverage.parquet")


if __name__ == "__main__":
    build()
