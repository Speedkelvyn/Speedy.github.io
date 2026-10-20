"""run_valuation.py: the Python version of the valuation pages.

Runs the same engine as the interactive terminals (valuations/dell, valuations/sndk)
on a company's inputs file and prints the full result: cost of capital, the three
DCF scenarios, Monte Carlo, comparables, earnings quality and the verdict.

    python run_valuation.py dell                      # Dell Technologies
    python run_valuation.py sndk --scenario Bull      # Sandisk, Bull case
    python run_valuation.py path/to/AAPL_inputs.json  # any inputs file
    python run_valuation.py dell --excel out.xlsx     # also write the live-formula workbook

DCF, WACC, comparables and regression match the web pages exactly. The Monte Carlo
uses NumPy's random generator instead of the page's, so its median (and the blended
fair value) can differ by a fraction of a percent.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import valuation_v2 as v  # noqa: E402

COMPANIES = {
    "dell": HERE.parent / "dell" / "dell_inputs.json",
    "sndk": HERE.parent / "sndk" / "SNDK_inputs.json",
}


def load(name_or_path: str) -> dict:
    p = COMPANIES.get(name_or_path.lower(), pathlib.Path(name_or_path))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def money(x: float) -> str:
    return "—" if x is None or not math.isfinite(x) else f"${x:,.2f}"


def pct(x: float, d: int = 1) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:+.{d}%}" if x < 0 else f"{x:.{d}%}"


def report(inp: dict, scenario: str = "Base", mc_n: int = 5000) -> dict:
    """Run the engine and print a readable summary. Returns the engine's result dict."""
    R = v.run_all(inp, scenario, mc_n=mc_n)
    m = inp["meta"]
    line = "─" * 64
    print(line)
    print(f"{m['name']} ({m['ticker']})   valued {m['valuation_date']}   scenario: {scenario}")
    print(line)
    if not R["checks"]["ok"]:
        print("NOT VALUED: these inputs are missing or invalid:")
        for b in R["checks"]["blocking"]:
            print(f"  [{b['path']}] {b['msg']}")
        return R

    W, vd, price = R["W"], R["verdict"], inp["market"]["price"]
    b = W["beta"]
    print("1  Discount rate")
    print(f"   Beta used {b['chosen']:.3f} (daily regression {b.get('daily', float('nan')):.3f})")
    print(f"   Cost of equity {W['ke']:.2%}   WACC {W['wacc']:.2%}   MRP {W['mrp']:.2%}")
    print("2  DCF value per share")
    for k, d in R["scen3"].items():
        print(f"   {k:<5} {money(d['perShare']):>12}   exit-multiple check {money(d['perShareExit'])}")
    mc = R["mc"]
    print("3  Cross-checks")
    print(f"   Monte Carlo median {money(mc['median'])}   p10–p90 {money(mc['p10'])} – {money(mc['p90'])}")
    print(f"   P(value > price)   {mc['probAbove']:.1%}")
    print(f"   Comparables blend  {money(R['rel']['blended'])}")
    print(f"   Peer regression    {money(R['reg']['implied'])}")
    q = vd.get("quality") or {}
    if q:
        print(f"4  Earnings quality grade {q.get('grade')}")
        for fl in q.get("flags", []):
            print(f"   - {fl['msg']}")
    print(line)
    conf = " (low confidence)" if vd.get("confidence") == "low" else ""
    print(f"VERDICT  {vd['call']}{conf}")
    print(f"   Fair value {money(vd['fair'])} vs price {money(price)}  →  upside {pct(vd['upside'])}")
    print(f"   Expected alpha {pct(vd['alpha'])} over the {vd['requiredReturn']:.2%} required return")
    for w in vd.get("warnings", []):
        print(f"   note: {w['msg']}")
    print(line)
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("company", nargs="?", default="dell", help="dell, sndk, or a path to an inputs JSON file")
    ap.add_argument("--scenario", default="Base", choices=["Bear", "Base", "Bull"])
    ap.add_argument("--mc", type=int, default=5000, help="Monte Carlo draws (default 5000)")
    ap.add_argument("--excel", metavar="PATH", help="also write the live-formula Excel workbook")
    a = ap.parse_args(argv)
    inp = load(a.company)
    R = report(inp, a.scenario, a.mc)
    if a.excel and R["checks"]["ok"]:
        import excel_model

        excel_model.build_workbook(inp, a.excel, a.scenario)
        print(f"Workbook written to {a.excel}")
    return 0 if R["checks"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
