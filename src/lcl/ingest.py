"""Low Carbon London smart-meter data (UK Power Networks via London Datastore,
CC BY 4.0) -> data/processed/hh.parquet (LCLid, tou, ts, kwh). Raw extract is untrusted data:
read as text, values parsed numerically, nothing executed. Also parses Tariffs.xlsx (dToU signals)."""
import glob, json, sys
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.csv as pc, pyarrow.parquet as pq

RAW = Path("data/raw/extract"); OUT = Path("data/processed")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(RAW / "**" / "*.csv"), recursive=True))
    writer, n, bad = None, 0, 0
    for f in files:
        t = pc.read_csv(f, convert_options=pc.ConvertOptions(column_types={"KWH/hh (per half hour) ": pa.string()}))
        d = t.to_pandas()
        d.columns = ["LCLid", "tou", "ts", "kwh"]
        d["kwh"] = pd.to_numeric(d["kwh"].str.strip(), errors="coerce")
        bad += int(d["kwh"].isna().sum())
        d["ts"] = pd.to_datetime(d["ts"]).astype("datetime64[us]")
        d["tou"] = (d["tou"] == "ToU")
        d["kwh"] = d["kwh"].astype("float32")
        tb = pa.Table.from_pandas(d, preserve_index=False)
        writer = writer or pq.ParquetWriter(OUT / "hh.parquet", tb.schema)
        writer.write_table(tb); n += len(d)
    writer.close()
    tar = pd.read_excel("data/raw/Tariffs.xlsx")
    tar.to_parquet(OUT / "tariffs.parquet")
    json.dump({"files": len(files), "rows": n, "kwh_unparsed": bad, "tariff_rows": len(tar), "tariff_cols": list(map(str, tar.columns))},
              open("reports/ingest_manifest.json", "w"), indent=1)
    print(open("reports/ingest_manifest.json").read())


if __name__ == "__main__":
    main()
