"""download_data.py: refresh every input series from FRED into data/raw/.

    pip install requests pandas
    python download_data.py

No API key is needed: the script uses FRED's public CSV download. The geopolitical risk index
(GPR.csv, Caldara & Iacoviello) and the global policy uncertainty index (EPU_global.csv,
Baker, Bloom & Davis) are not on FRED; download them from their websites and save them in
data/raw/ with the same names when you want to update them.
"""

from __future__ import annotations

import io
import pathlib

import pandas as pd
import requests

import fx_forecasting as F

RAW = pathlib.Path(__file__).resolve().parent / "data" / "raw"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch(series_id: str, start: str = "1995-01-01", **params: str) -> pd.DataFrame:
    r = requests.get(URL, params={"id": series_id, "cosd": start, **params}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    return df


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    monthly = [*F.RATE_SERIES.values(), *F.CPI_SERIES.values(), *F.UNEMP_SERIES.values()]
    jobs = {sid: {} for sid in monthly}
    jobs.update({sid: {"fq": "Monthly", "fam": "eop"} for sid, _ in F.FX_SERIES.values()})  # end of month
    jobs["VIXCLS"] = {"fq": "Monthly", "fam": "avg"}
    for sid, params in jobs.items():
        df = fetch(sid, **params)
        df.to_csv(RAW / f"{sid}.csv", index=False)
        print(f"{sid:<22} {df['value'].notna().sum():>5} obs, last {df.dropna()['date'].iloc[-1]}")
    daily = fetch("DEXJPUS", start="2000-01-01")
    daily.to_csv(RAW / "DEXJPUS_daily.csv", index=False)
    print(f"{'DEXJPUS_daily':<22} {daily['value'].notna().sum():>5} obs")


if __name__ == "__main__":
    main()
