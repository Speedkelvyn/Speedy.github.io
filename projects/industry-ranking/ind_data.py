"""ind_data.py: panel of 49 US industries for ranking next month's winners and losers.

Data: Kenneth French's 49 value-weighted industry portfolios (monthly, July 1926 to August 2026) and the
Fama-French factors. Everything below is known at the end of month t.

Industry characteristics (cross-sectionally ranked each month to [-0.5, 0.5], as in Gu, Kelly & Xiu 2020)
    mom_1       last month's return (short-term reversal)
    mom_3, mom_6, mom_12   returns over the last 3, 6 and 12 months
    mom_12_1    the last 12 months skipping the latest month (Jegadeesh & Titman 1993; Moskowitz & Grinblatt 1999)
    rev_36      months t-35 to t-12 (long-term reversal, De Bondt & Thaler 1985)
    vol_12      volatility of the last 12 months;  dvol_12  downside volatility
    beta_36     market beta over the last 36 months
    bm, bm_rel  book-to-market of the industry and relative to its own 5-year average (valuation)
    mkt_rel_12  the industry's 12-month return minus the market's
State variables (the same for every industry in a month; used to let characteristics matter more or less
in different markets): market volatility, market 12-month return, term spread, excess bond premium,
Baa spread.

Target: next month's industry return minus the average industry return that month (a pure ranking target).
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd

RAW = pathlib.Path(__file__).resolve().parent / "data" / "raw"
CHARS = [
    "mom_1", "mom_3", "mom_6", "mom_12", "mom_12_1", "rev_36", "vol_12", "dvol_12", "beta_36", "bm", "bm_rel",
    "mkt_rel_12",
]  # fmt: skip
STATES = ["mkt_vol", "mkt_ret_12", "term", "ebp", "baa"]


def _section(lines: list[str], title: str) -> pd.DataFrame:
    """Read one block of the industry file: the header row after `title`, then rows until a blank line."""
    i = next(k for k, ln in enumerate(lines) if ln.strip() == title)
    header = [c.strip() for c in lines[i + 1].split(",")][1:]
    rows = []
    for ln in lines[i + 2 :]:
        if not ln.strip():
            break
        parts = [p.strip() for p in ln.split(",")]
        rows.append([parts[0], *[float(x) for x in parts[1 : len(header) + 1]]])
    df = pd.DataFrame(rows, columns=["key", *header]).set_index("key")
    return df.replace([-99.99, -999.0], np.nan)


def load_industries(raw: pathlib.Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly value-weighted returns (decimal) and annual book-to-market of the 49 industries."""
    lines = (raw / "49_Industry_Portfolios.csv").read_text(encoding="latin-1").splitlines()
    ret = _section(lines, "Average Value Weighted Returns -- Monthly") / 100
    ret.index = pd.to_datetime(ret.index, format="%Y%m")
    bm = _section(lines, "Sum of BE / Sum of ME")
    bm.index = bm.index.astype(int)
    return ret, bm


def load_market(raw: pathlib.Path = RAW) -> pd.DataFrame:
    rows = []
    for line in (raw / "F-F_Research_Data_Factors_monthly.csv").read_text(encoding="latin-1").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5 and re.fullmatch(r"\d{6}", p[0]):
            rows.append([p[0], *[float(x) / 100 for x in p[1:5]]])
    df = pd.DataFrame(rows, columns=["date", "mktrf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    df = df.set_index("date")
    df["mkt"] = df["mktrf"] + df["rf"]
    return df


def _monthly(path: pathlib.Path) -> pd.Series:
    df = pd.read_csv(path)
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return s[~s.index.duplicated(keep="last")].sort_index()


def cross_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Rank each month's values across industries and rescale to [-0.5, 0.5]."""
    r = x.rank(axis=1, method="average")
    n = x.notna().sum(axis=1)
    return r.sub(1).div((n - 1).replace(0, np.nan), axis=0) - 0.5


def characteristics(ret: pd.DataFrame, bm_annual: pd.DataFrame, mkt: pd.Series) -> dict[str, pd.DataFrame]:
    """Wide (month x industry) frames of the raw characteristics."""
    lr = np.log1p(ret)
    cum = lambda k, lag=0: np.expm1(lr.shift(lag).rolling(k, min_periods=k).sum())
    c = {
        "mom_1": ret,
        "mom_3": cum(3),
        "mom_6": cum(6),
        "mom_12": cum(12),
        "mom_12_1": cum(11, 1),
        "rev_36": cum(24, 12),
        "vol_12": ret.rolling(12, min_periods=12).std(),
        "dvol_12": ret.clip(upper=0).rolling(12, min_periods=12).std(),
    }
    mk = mkt.reindex(ret.index)
    cov = ret.rolling(36, min_periods=36).cov(mk)
    c["beta_36"] = cov.div(mk.rolling(36, min_periods=36).var(), axis=0)
    # Conservative timing: the annual book-to-market labelled year Y is first used in July of year Y+1, so even
    # if it rests on balance sheets that closed in Y it was public (about six months of slack) when it is used.
    bm_m = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for y in bm_annual.index:
        sel = (ret.index >= pd.Timestamp(y + 1, 7, 1)) & (ret.index < pd.Timestamp(y + 2, 7, 1))
        bm_m.loc[sel, :] = bm_annual.loc[y, ret.columns].to_numpy(dtype=float)
    lbm = np.log(bm_m.where(bm_m > 0))
    c["bm"] = lbm
    c["bm_rel"] = lbm - lbm.rolling(60, min_periods=36).mean()
    c["mkt_rel_12"] = c["mom_12"].sub(np.expm1(np.log1p(mk).rolling(12, min_periods=12).sum()), axis=0)
    return c


def state_variables(mkt: pd.Series, daily_rv: pd.Series | None, raw: pathlib.Path = RAW) -> pd.DataFrame:
    ebp = pd.read_csv(raw / "ebp.csv", parse_dates=["date"]).set_index("date")
    ebp.index = ebp.index.to_period("M").to_timestamp()
    st = pd.DataFrame(
        {
            "mkt_vol": np.log(daily_rv) if daily_rv is not None else np.nan,
            "mkt_ret_12": np.expm1(np.log1p(mkt).rolling(12, min_periods=12).sum()),
            "term": _monthly(raw / "T10Y3M.csv"),
            "ebp": ebp["ebp"].shift(1),
            "baa": _monthly(raw / "BAA10Y.csv"),
        }
    )
    return st


def monthly_realised_variance(raw: pathlib.Path = RAW) -> pd.Series:
    rows = []
    for line in (raw / "F-F_Research_Data_Factors_daily.csv").read_text(encoding="latin-1").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5 and re.fullmatch(r"\d{8}", p[0]):
            rows.append((p[0], (float(p[1]) + float(p[4])) / 100))
    d = pd.DataFrame(rows, columns=["date", "r"])
    d["date"] = pd.to_datetime(d["date"], format="%Y%m%d")
    rv = (d.set_index("date")["r"] ** 2).groupby(d["date"].dt.to_period("M").to_numpy()).sum()
    rv.index = rv.index.to_timestamp()
    return rv


def build_panel(raw: pathlib.Path = RAW, start: str = "1940-01-01") -> dict:
    """Long panel (month, industry) with ranked characteristics, state variables and next-month targets."""
    ret, bm_annual = load_industries(raw)
    fac = load_market(raw)
    mkt = fac["mkt"]
    chars = characteristics(ret, bm_annual, mkt)
    ranked = {k: cross_rank(v) for k, v in chars.items()}
    nxt = ret.shift(-1)
    y_rel = nxt.sub(nxt.mean(axis=1), axis=0)
    st = state_variables(mkt, monthly_realised_variance(raw), raw).reindex(ret.index)
    long = pd.concat(
        {k: v.stack(future_stack=True) for k, v in {**ranked, "y": y_rel, "ret_next": nxt}.items()}, axis=1
    )
    long.index.names = ["date", "industry"]
    long = long.reset_index()
    long = long.merge(st.reset_index(names="date"), on="date", how="left")
    long = long[long["date"] >= pd.Timestamp(start)].copy()
    return {"panel": long, "returns": ret, "factors": fac, "raw_chars": chars, "states": st}
