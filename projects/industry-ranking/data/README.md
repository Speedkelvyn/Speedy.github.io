# Data

Downloaded on 21 October 2026. All files are in `raw/`.

| File | Contents | Source |
|---|---|---|
| `49_Industry_Portfolios.csv` | Monthly value-weighted returns of 49 industries (July 1926 – August 2026) and annual book-to-market (sum of BE / sum of ME) | Kenneth R. French Data Library |
| `F-F_Research_Data_Factors_monthly.csv` | Market excess return, SMB, HML and the risk-free rate, monthly | Kenneth R. French Data Library |
| `F-F_Research_Data_Factors_daily.csv` | The same factors daily; used for monthly realised market volatility | Kenneth R. French Data Library |
| `T10Y3M.csv` | 10-year minus 3-month Treasury spread, monthly (from 1982) | FRED |
| `BAA10Y.csv` | Baa corporate yield minus the 10-year Treasury yield, monthly (from 1986) | FRED |
| `ebp.csv` | Gilchrist–Zakrajšek credit spread and excess bond premium (from 1973) | Federal Reserve Board (FEDS Notes) |

The market-state variables start at different dates (the term spread in 1982, the Baa spread in 1986); a variable enters the models only after ten years of history.
