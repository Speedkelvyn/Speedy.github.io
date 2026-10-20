# Data

All series are in `raw/` as `date,value` CSV files, downloaded on 20 October 2026. Refresh them with `python download_data.py`.

| Variable | Countries | FRED series | Notes |
|---|---|---|---|
| Exchange rates (end of month) | EUR, GBP, JPY, CHF, CAD, AUD vs USD | DEXUSEU, DEXUSUK, DEXJPUS, DEXSZUS, DEXCAUS, DEXUSAL | Converted to USD per foreign unit, so a positive return is foreign appreciation |
| USD/JPY daily | Japan | DEXJPUS (`DEXJPUS_daily.csv`) | Part I, 2000 onwards |
| 3-month interest rates | US, euro area, UK, Switzerland, Canada, Australia | IR3TIB01{US,EZ,GB,CH,CA,AU}M156N | OECD Main Economic Indicators |
| Short rate, Japan | Japan | IRSTCI01JPM156N | Call rate; the 3-month series only starts in 2002 |
| Consumer prices | US, euro area, UK, Japan, Switzerland, Canada, Australia | CPIAUCSL, CP0000EZ19M086NEST, GBRCPIALLMINMEI, JPNCPIALLMINMEI, CHECPIALLMINMEI, CANCPIALLMINMEI, AUSCPIALLQINMEI | Inflation = 12-month log change (4-quarter for Australia) |
| Unemployment rate | same seven | LRHUTTTT{US,EZ,GB,JP,CA,AU}M156S, LRHUTTTTCHQ156S | Swiss data are annual before 2010 and quarterly after |
| VIX | Global | VIXCLS | Monthly average |
| Geopolitical risk (`GPR.csv`) | Global | not on FRED | Caldara & Iacoviello, matteoiacoviello.com/gpr.htm |
| Policy uncertainty (`EPU_global.csv`) | Global | not on FRED | Baker, Bloom & Davis, policyuncertainty.com (GEPU, current prices) |

## Coverage limits

FRED stopped updating some OECD series: Japanese CPI ends in June 2021 and euro-area unemployment in January 2023. Those currencies leave the panel when their data end; nothing is filled in after the last observation.
