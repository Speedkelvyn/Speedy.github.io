"""make_sample.py: build sample.csv for the regression tool from the recession-risk study's data.

python make_sample.py

One row per month from January 1973. Columns:
    date           month
    recession_12m  1 if an NBER recession starts within the next 12 months, 0 if not, blank when the economy is
                   already in recession or the 12 months are not over (the tool drops those rows)
    mkt_ret_next   US stock-market return next month, % (CRSP value-weighted, from Kenneth French's daily file)
    term_spread    10-year minus 3-month Treasury yield, % points (blank before 1982)
    ebp            Gilchrist-Zakrajsek excess bond premium, % points (previous month's value)
    baa_spread     Baa corporate yield minus 10-year Treasury yield, % points (blank before 1986)
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "projects" / "recession-risk"))
import rec_data as D  # noqa: E402

d = D.build_dataset()
mkt = D.market_monthly()
out = d["X"][["TERM_10Y3M", "EBP", "BAA_SPR"]].rename(
    columns={"TERM_10Y3M": "term_spread", "EBP": "ebp", "BAA_SPR": "baa_spread"}
)
out.insert(0, "mkt_ret_next", (100 * mkt.shift(-1)).reindex(out.index))
out.insert(0, "recession_12m", d["targets"]["RECESSION"])
out = out[out.index >= "1973-01-01"].round(4)
out["recession_12m"] = out["recession_12m"].astype("Int64")
out.index = out.index.strftime("%Y-%m")
out.index.name = "date"
out.to_csv(HERE / "sample.csv")
print(f"sample.csv: {len(out)} rows, {out.isna().sum().to_dict()}")
