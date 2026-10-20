"""fetch_yahoo.py — pull the Yahoo Finance feeds a valuation needs, for any ticker.

Reads the same endpoints as the terminal's LOAD tab (quoteSummary, the
fundamentals time series, daily charts for the stock, the S&P 500 and the
10-year yield, related symbols), using yfinance's session for the cookie and
crumb. Saves the raw snapshot so the valuation can be reproduced later.

    pip install yfinance
    python fetch_yahoo.py AAPL                 # writes AAPL_yahoo_<date>.json and AAPL_inputs.json
    python fetch_yahoo.py AAPL --peers MSFT GOOGL META

From Python:
    from fetch_yahoo import fetch
    feeds = fetch("AAPL")                      # dict research.build() takes
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import research as RP

Q1 = "https://query1.finance.yahoo.com"
MODULES = "price,summaryDetail,defaultKeyStatistics,financialData,earningsTrend,recommendationTrend,upgradeDowngradeHistory,assetProfile"
PEER_MODULES = "price,summaryDetail,defaultKeyStatistics,financialData,earningsTrend"


def _session():
    try:
        from yfinance.data import YfData
    except ImportError as e:
        raise SystemExit("pip install yfinance   (used only for Yahoo's cookie and crumb)") from e
    return YfData()


def _summary(yd, t, modules):
    j = yd.get_raw_json(f"{Q1}/v10/finance/quoteSummary/{t}", params={"modules": modules})
    res = (j.get("quoteSummary") or {}).get("result")
    return res[0] if res else None


def _timeseries(yd, t, period):
    j = yd.get_raw_json(
        f"{Q1}/ws/fundamentals-timeseries/v1/finance/timeseries/{t}",
        params={"type": ",".join(period + x for x in RP.TYPES), "period1": 1420070400, "period2": 1900000000},
    )
    return RP.series(j["timeseries"]["result"])


def _chart(yd, t, rng="2y"):
    j = yd.get_raw_json(f"{Q1}/v8/finance/chart/{t}", params={"range": rng, "interval": "1d"})
    return j["chart"]["result"][0]


def fetch(ticker, peers=None, today=None):
    t = ticker.upper()
    yd = _session()
    summary = _summary(yd, t, MODULES)
    if summary is None:
        raise SystemExit(f"Yahoo returned nothing for {t}")
    if peers is None:
        try:
            j = yd.get_raw_json(f"https://query2.finance.yahoo.com/v6/finance/recommendationsbysymbol/{t}")
            peers = [r["symbol"] for r in j["finance"]["result"][0]["recommendedSymbols"]][:8]
        except Exception:
            peers = []
    peer_data = {}
    for p in peers:
        try:
            peer_data[p] = _summary(yd, p, PEER_MODULES)
        except Exception:
            peer_data[p] = None
    tnx = _chart(yd, "%5ETNX", "5d")["meta"]["regularMarketPrice"]
    return {
        "ticker": t,
        "today": today or dt.date.today().isoformat(),
        "tnx": tnx,
        "summary": summary,
        "annual": _timeseries(yd, t, "annual"),
        "quarterly": _timeseries(yd, t, "quarterly"),
        "stock": RP.closes(_chart(yd, t)),
        "index": RP.closes(_chart(yd, "%5EGSPC")),
        "peers": peer_data,
        "built": f"Yahoo Finance feeds via fetch_yahoo.py, {dt.date.today().isoformat()}",
    }


def save_snapshot(feeds, path):
    """compact-v1 snapshot: research.from_fixture() reads it back, so a valuation can be rerun on the same data."""
    out = {
        k: feeds[k]
        for k in ("ticker", "today", "tnx", "summary", "annual", "quarterly", "stock", "index", "peers")
    }
    out["format"] = "compact-v1"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("--peers", nargs="*")
    a = ap.parse_args()
    feeds = fetch(a.ticker, a.peers)
    snap = f"{feeds['ticker']}_yahoo_{feeds['today']}.json"
    save_snapshot(feeds, snap)
    inp = RP.build(feeds)
    with open(f"{feeds['ticker']}_inputs.json", "w", encoding="utf-8") as f:
        json.dump(inp, f, indent=1)
    print(
        f"snapshot {snap}; inputs {feeds['ticker']}_inputs.json; model {inp['model']}; "
        f"{len(inp['data_quality']['blocking'])} blocking, {len(inp['data_quality']['warnings'])} warnings"
    )
