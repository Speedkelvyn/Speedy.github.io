# Data

Downloaded on 21 October 2026. All files are in `raw/`.

| File | Contents | Source |
|---|---|---|
| `F-F_Research_Data_Factors_daily.csv` | Daily Fama–French factors and risk-free rate, 1926–August 2026; the market return is Mkt-RF + RF | Kenneth R. French Data Library |
| `fred_md_2026-09.csv` | FRED-MD, 126 monthly US macro and financial series from 1959; the first row holds the transformation codes | McCracken & Ng, Federal Reserve Bank of St. Louis |
| `VIXCLS_eom.csv` | CBOE VIX closing level on the last trading day of each month, 1990–September 2026 | Cboe, via Yahoo Finance (^VIX) |
| `ebp.csv` | Gilchrist–Zakrajšek credit spread and excess bond premium, from 1973 | Federal Reserve Board (FEDS Notes) |
| `BAA10Y.csv`, `T10Y3M.csv`, `T10Y2Y.csv` | Baa minus 10-year Treasury spread and term spreads, monthly | FRED |
| `NFCI.csv`, `ANFCI.csv`, `STLFSI4.csv` | Chicago Fed financial-conditions indexes and the St. Louis Fed stress index, monthly | FRED |
| `DRTSCILM.csv` | Net share of banks tightening standards on C&I loans (quarterly) | FRED, Senior Loan Officer Survey |
| `DRBLACBS.csv`, `DRCLACBS.csv` | Delinquency rates on business and all loans (quarterly) | FRED |
| `USREC.csv` | NBER recession indicator | FRED |

The VIX is the month-end level, not the monthly average that FRED publishes: a monthly average is stale by about half a month when used to forecast the next month. The FRED-MD file also carries a monthly-average VIX (`VIXCLSx`), which the models may use as one of the 126 macro series.
