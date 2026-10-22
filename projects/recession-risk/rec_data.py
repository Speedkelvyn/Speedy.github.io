"""rec_data.py: real-time dataset for forecasting US recessions and stock-market stress one year ahead.

Targets (month t, forecast made at the end of the month)
    RECESSION  1 if an NBER recession starts in any of the next 12 months; defined for months when the economy
               is not already in recession (so persistence cannot flatter the model).
    BEAR       1 if the US stock market falls at least 20% below its month-t level at some point in the next
               12 months (CRSP value-weighted market, total return).

Predictors, known at the end of month t
    * the yield-curve slope (10-year minus 3-month and minus 2-year Treasury yields);
    * credit conditions: Baa - 10-year Treasury spread, the Gilchrist-Zakrajsek excess bond premium (EBP);
    * the Chicago Fed financial-conditions indexes, the St. Louis Fed stress index and the VIX;
    * labour-market and activity data from FRED-MD, with their publication lag;
    * bank lending standards and delinquencies (quarterly, lagged one quarter).

Real-time labelling: NBER announces turning points 4 to 21 months after the fact, so a recession label is
only used for training once it could have been known: a month s joins the training set at s + 24 months.
Stock-market labels are known once the 12 months have passed (s + 12 months).
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd

RAW = pathlib.Path(__file__).resolve().parent / "data" / "raw"
HORIZON = 12
RECESSION_LAG = 24  # months before a recession label counts as known
BEAR_LAG = 12
BEAR_THRESHOLD = -0.20

FINANCIAL = {
    "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA", "COMPAPFFx", "TB3SMFFM",
    "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM", "TWEXAFEGSMTHx", "EXSZUSx", "EXJPUSx",
    "EXUSUKx", "EXCAUSx", "S&P 500", "S&P div yield", "S&P PE ratio", "VIXCLSx",
}  # fmt: skip


def _monthly(path: pathlib.Path, col: str = "value", date_col: str = "date") -> pd.Series:
    df = pd.read_csv(path)
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    return s[~s.index.duplicated(keep="last")].sort_index()


def transform(x: pd.Series, code: int) -> pd.Series:
    """FRED-MD transformation codes (McCracken & Ng 2016, Appendix)."""
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 4:
        return np.log(x)
    if code == 5:
        return np.log(x).diff()
    if code == 6:
        return np.log(x).diff().diff()
    if code == 7:
        return (x / x.shift(1) - 1).diff()
    raise ValueError(f"unknown transformation code {code}")


def load_fred_md(raw: pathlib.Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(raw / "fred_md_2026-09.csv")
    codes = df.iloc[0, 1:].astype(int)
    df = df.iloc[1:].copy()
    df.index = pd.to_datetime(df.pop("sasdate"), format="%m/%d/%Y")
    df = df.apply(pd.to_numeric, errors="coerce")
    out = {}
    for c in df.columns:
        x = transform(df[c], int(codes[c]))
        out[f"md_{c}"] = x if c in FINANCIAL else x.shift(1)  # macro data: one-month publication lag
    # a few transformations recession models traditionally use, built from levels
    pay, un, ip = df["PAYEMS"].shift(1), df["UNRATE"].shift(1), df["INDPRO"].shift(1)
    out["PAYEMS_3m"] = np.log(pay).diff(3) * 400  # annualised 3-month employment growth, %
    out["PAYEMS_12m"] = np.log(pay).diff(12) * 100
    out["UNRATE_chg12"] = un.diff(12)
    # Sahm-rule style: unemployment above its 12-month low (October 2025 has no survey, so tolerate gaps)
    out["UNRATE_vs_min"] = un - un.rolling(12, min_periods=8).min()
    out["INDPRO_12m"] = np.log(ip).diff(12) * 100
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def load_signals(raw: pathlib.Path = RAW) -> pd.DataFrame:
    ebp = pd.read_csv(raw / "ebp.csv", parse_dates=["date"]).set_index("date")
    ebp.index = ebp.index.to_period("M").to_timestamp()
    q = lambda s: s.resample("MS").ffill().shift(3)
    return pd.DataFrame(
        {
            "TERM_10Y3M": _monthly(raw / "T10Y3M.csv"),
            "TERM_10Y2Y": _monthly(raw / "T10Y2Y.csv"),
            "BAA_SPR": _monthly(raw / "BAA10Y.csv"),
            "EBP": ebp["ebp"].shift(1),
            "GZ": ebp["gz_spread"].shift(1),
            "NFCI": _monthly(raw / "NFCI.csv").shift(1),
            "ANFCI": _monthly(raw / "ANFCI.csv").shift(1),
            "STLFSI": _monthly(raw / "STLFSI4.csv").shift(1),
            "VIX": _monthly(raw / "VIXCLS.csv"),
            "LOAN_STANDARDS": q(_monthly(raw / "DRTSCILM.csv")),
            "DELINQ_CI": q(_monthly(raw / "DRBLACBS.csv")),
            "DELINQ_ALL": q(_monthly(raw / "DRCLACBS.csv")),
        }
    )


def market_monthly(raw: pathlib.Path = RAW) -> pd.Series:
    """Monthly total return of the CRSP value-weighted market (decimal), from Kenneth French's daily file."""
    rows = []
    for line in (raw / "F-F_Research_Data_Factors_daily.csv").read_text(encoding="latin-1").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5 and re.fullmatch(r"\d{8}", p[0]):
            rows.append((p[0], (float(p[1]) + float(p[4])) / 100))
    d = pd.DataFrame(rows, columns=["date", "r"])
    d["date"] = pd.to_datetime(d["date"], format="%Y%m%d")
    g = d.set_index("date")["r"]
    m = (1 + g).groupby(g.index.to_period("M")).prod() - 1
    m.index = m.index.to_timestamp()
    return m


def recession_flag(raw: pathlib.Path = RAW) -> pd.Series:
    return _monthly(raw / "USREC.csv").astype(int)


def make_targets(rec: pd.Series, mkt: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """RECESSION (a recession starts within 12 months) and BEAR (market down 20% within 12 months)."""
    start = ((rec == 1) & (rec.shift(1) == 0)).astype(int)
    fwd_start = sum(start.shift(-k) for k in range(1, HORIZON + 1)).reindex(index)
    # months at the end of the sample without a full 12-month window have no label yet
    full = start.shift(-HORIZON).reindex(index).notna()
    recession = ((fwd_start > 0).astype(float)).where(full & (rec.reindex(index) == 0))
    wealth = (1 + mkt).cumprod()
    low = pd.concat([wealth.shift(-k) for k in range(1, HORIZON + 1)], axis=1).min(axis=1, skipna=False)
    dd = (low / wealth - 1).reindex(index)
    bear = (dd <= BEAR_THRESHOLD).astype(float).where(dd.notna())
    return pd.DataFrame({"RECESSION": recession, "BEAR": bear, "in_recession": rec.reindex(index)})


def build_dataset(raw: pathlib.Path = RAW, start: str = "1960-01-01") -> dict:
    sig, md = load_signals(raw), load_fred_md(raw)
    mkt = market_monthly(raw)
    X = pd.concat([sig, md], axis=1)
    X["MKT_RET_12M"] = (1 + mkt).rolling(12).apply(np.prod, raw=True).reindex(
        X.index
    ) - 1  # stock market's last year
    X["MKT_DD"] = ((1 + mkt).cumprod() / (1 + mkt).cumprod().cummax() - 1).reindex(
        X.index
    )  # distance from the peak
    X = X[(X.index >= pd.Timestamp(start)) & (X.index <= sig["TERM_10Y3M"].dropna().index.max())]
    tg = make_targets(recession_flag(raw), mkt, X.index)
    return {"X": X, "targets": tg, "market": mkt, "recession": recession_flag(raw)}
