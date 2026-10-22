"""value_ticker.py: value any listed company with one command.

    python value_ticker.py AAPL                                  # fetch, value, write valuations/aapl/
    python value_ticker.py AAPL --peers MSFT GOOGL META --scenario Bull
    python value_ticker.py --snapshot AAPL_yahoo_2026-10-21.json  # rerun on saved data, no network
    python value_ticker.py AAPL --out my_folder --no-excel --no-page

Steps
    1. Pull Yahoo Finance's feeds for the ticker (fetch_yahoo.fetch, needs `pip install yfinance`), or read a
       snapshot saved by an earlier run.
    2. Build the inputs file with a provenance tag on every input (research.build).
    3. Print the valuation: cost of capital, three DCF scenarios, Monte Carlo, comparables, earnings quality
       and the verdict (run_valuation.report).
    4. Write, into one folder: the inputs file, the Yahoo snapshot, the interactive page (index.html) and the
       Excel workbook with live formulas.

The default folder, valuations/<ticker>/, has the same layout as dell/ and sndk/, so the page works where it is.
Exit status: 0 valued, 2 the inputs are missing something the model needs (the files are still written, so the
gaps can be filled in by hand and the page or `run_valuation.py` rerun).

Valuations are illustrations built on public data, not investment advice. Check the data-quality warnings.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "web"))
import build_page  # noqa: E402
import research as RP  # noqa: E402
import run_valuation  # noqa: E402

TICKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")  # class shares use a dash (BRK-B)


def clean_ticker(t: str) -> str:
    if not TICKER.match(t):
        raise SystemExit(
            f"'{t}' is not a ticker symbol (letters, digits, '.' or '-', at most 10 characters)."
        )
    return t.upper()


def default_folder(ticker: str) -> pathlib.Path:
    return HERE.parent / re.sub(r"[^a-z0-9]+", "-", ticker.lower()).strip("-")


def value(
    ticker: str | None = None,
    snapshot: pathlib.Path | None = None,
    peers: list[str] | None = None,
    scenario: str = "Base",
    mc: int = 5000,
    out: pathlib.Path | None = None,
    excel: bool = True,
    page: bool = True,
    fetcher=None,
    builder=None,
) -> tuple[dict, dict, dict[str, pathlib.Path]]:
    """Run the whole pipeline. Returns (inputs, engine result, files written).

    `fetcher` and `builder` default to fetch_yahoo.fetch and research.build; tests replace them.
    """
    if (ticker is None) == (snapshot is None):
        raise SystemExit("Give either a ticker or --snapshot FILE.")
    builder = builder or RP.build
    if snapshot is not None:
        feeds = RP.from_fixture(json.loads(pathlib.Path(snapshot).read_text(encoding="utf-8")))
        symbol = clean_ticker(feeds["ticker"])
    else:
        symbol = clean_ticker(ticker)
        if fetcher is None:
            import fetch_yahoo

            fetcher = fetch_yahoo.fetch
        feeds = fetcher(symbol, [clean_ticker(p) for p in peers] if peers else None)
        feeds["ticker"] = symbol

    folder = pathlib.Path(out) if out else default_folder(symbol)
    folder.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}

    if snapshot is None:
        import fetch_yahoo

        written["snapshot"] = folder / f"{symbol}_yahoo_{feeds['today']}.json"
        fetch_yahoo.save_snapshot(feeds, written["snapshot"])

    inp = builder(feeds)
    written["inputs"] = folder / f"{symbol}_inputs.json"
    written["inputs"].write_text(json.dumps(inp, indent=1), encoding="utf-8")

    R = run_valuation.report(inp, scenario, mc)
    dq = inp.get("data_quality") or {}
    print(f"Data quality: {len(dq.get('blocking', []))} blocking, {len(dq.get('warnings', []))} warnings")
    for w in dq.get("warnings", [])[:8]:
        print(f"   - {w['msg']}" if isinstance(w, dict) else f"   - {w}")

    if page:
        written["page"] = folder / "index.html"
        build_page.build(inp, written["page"])
    if excel and R["checks"]["ok"]:
        import excel_model

        written["workbook"] = folder / f"{symbol}_valuation.xlsx"
        excel_model.build_workbook(inp, str(written["workbook"]), scenario)
    return inp, R, written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker", nargs="?", help="Yahoo Finance symbol, e.g. AAPL or BRK-B")
    ap.add_argument("--snapshot", type=pathlib.Path, help="a saved *_yahoo_*.json file instead of fetching")
    ap.add_argument("--peers", nargs="*", help="comparable tickers (default: Yahoo's related symbols)")
    ap.add_argument("--scenario", default="Base", choices=["Bear", "Base", "Bull"])
    ap.add_argument("--mc", type=int, default=5000, help="Monte Carlo draws (default 5000)")
    ap.add_argument("--out", type=pathlib.Path, help="output folder (default valuations/<ticker>)")
    ap.add_argument("--no-excel", action="store_true", help="skip the Excel workbook")
    ap.add_argument("--no-page", action="store_true", help="skip the interactive page")
    a = ap.parse_args(argv)
    _, R, written = value(
        a.ticker, a.snapshot, a.peers, a.scenario, a.mc, a.out, not a.no_excel, not a.no_page
    )
    print("Written:")
    for kind, path in written.items():
        print(f"   {kind:<9} {path}")
    if not R["checks"]["ok"]:
        print("The model could not be valued from these inputs; fix the items listed above and rerun.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
