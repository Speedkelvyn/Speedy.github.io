"""framework.py — the valuation, one step at a time, with every formula shown with its numbers.

Used by valuation_framework.ipynb. Each step returns tables (pandas) and the
formula lines it used, so you can see exactly how the value was reached:

    1 history → 2 cash flows → 3 discount rate → 4 discount → 5 price → 6 verdict

All numbers come from the same engine as the Excel workbook and the terminal
(valuation_v2.py), so the three always agree.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

import research as RP
import valuation_v2 as v

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

MODEL_NAME = {
    "fcff": "FCFF DCF (operating company)",
    "excess_return": "Excess-return model (bank / insurer)",
    "ddm": "Dividend discount model (REIT)",
}


def p(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{d}f}%"


def m(x, d=0):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{d}f}"


def usd(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"${x:,.2f}"


# ------------------------------------------------------------------ loading
def load(ticker=None, snapshot=None, inputs=None, peers=None):
    """Pull from Yahoo (ticker), rebuild from a saved snapshot, or open a hand-built inputs JSON."""
    if inputs:
        return json.load(open(inputs))
    if snapshot:
        return RP.build(RP.from_fixture(json.load(open(snapshot))))
    import fetch_yahoo as F

    feeds = F.fetch(ticker, peers)
    F.save_snapshot(feeds, f"{feeds['ticker']}_yahoo_{feeds['today']}.json")
    return RP.build(feeds)


def _get(o, path):
    for k in path.split("."):
        o = o[int(k)] if isinstance(o, list) else o[k]
    return o


def _set(o, path, val):
    ks = path.split(".")
    for k in ks[:-1]:
        o = o[int(k)] if isinstance(o, list) else o[k]
    if isinstance(o, list):
        o[int(ks[-1])] = val
    else:
        o[ks[-1]] = val


def override(inp, changes):
    """Your values over the pulled ones, e.g. {"capital.rf": 0.045, "scenarios.Base.margins.total.1": 0.09}.
    Returns the new inputs and a table of pulled vs used, with the source of the pulled value."""
    out = copy.deepcopy(inp)
    rows = []
    for path, val in (changes or {}).items():
        try:
            old = _get(out, path)
        except (KeyError, IndexError, TypeError):
            old = None
        _set(out, path, val)
        src = (inp.get("provenance") or {}).get(path, ["", ""])
        out.setdefault("provenance", {})[path] = [
            "assumption",
            f"your value (pulled: {old!r}, {src[0]} {src[1]})".strip(),
        ]
        rows.append({"input": path, "pulled": old, "your value": val, "pulled from": f"{src[0]}: {src[1]}"})
    return out, pd.DataFrame(rows)


def checks(inp):
    Q = v.check_inputs(inp)
    lines = [
        f"Model: {MODEL_NAME[v.model_of(inp)]}  ·  {inp['meta']['name']} ({inp['meta']['ticker']})  ·  price {usd(inp['market']['price'])}"
    ]
    if not Q["ok"]:
        lines.append("NOT VALUED — fix these inputs first (use override):")
        lines += [f"  ✗ [{b['path']}] {b['msg']}" for b in Q["blocking"]]
    lines += [f"  ! [{w['path']}] {w['msg']}" for w in Q["warnings"]]
    return Q["ok"], lines


def sources(inp):
    """Every input with where it came from: sourced, derived (formula given) or assumption."""
    rows = []
    for path, (kind, src) in sorted((inp.get("provenance") or {}).items()):
        try:
            val = _get(inp, path)
        except (KeyError, IndexError, TypeError):
            val = None
        if isinstance(val, (dict, list)):
            val = "(table)"
        rows.append({"input": path, "value": val, "origin": kind, "source / basis": src})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- 1 history
def step1_history(inp):
    h = inp.get("history", {}).get("annual", {})
    names = inp.get("history", {}).get("annual_names", {})
    df = (
        pd.DataFrame(
            {names.get(k, k): vals for k, vals in h.items() if k != "labels"}, index=h.get("labels")
        ).T
        if h.get("labels")
        else pd.DataFrame()
    )
    if "Revenue" in df.index:
        rev = df.loc["Revenue"].astype(float)
        extra = {"Revenue growth": rev / rev.shift(1) - 1}
        for k in ("Operating income", "Net income", "Cash from operations"):
            if k in df.index:
                extra[f"{k} ÷ revenue"] = df.loc[k].astype(float) / rev
        df = pd.concat([df, pd.DataFrame(extra).T])
    return df


def step1_quality(inp):
    E = v.earnings_quality(inp)
    lines = [f"Earnings-quality grade: {E['grade']}"]
    if E["grade"] == "n/a" and not E["years"]:
        return E, pd.DataFrame(), [*lines, E.get("reason", "")]
    keys = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
    tab = pd.DataFrame(
        {y["label"]: {**{k: y["idx"][k] for k in keys}, "M-score": y["m"]} for y in E["years"]}
    )
    tab["coefficient"] = [v.BENEISH[k] for k in keys] + [v.BENEISH["c"]]
    tab["manipulators' mean"] = [v.BENEISH_MANIP[k] for k in keys] + [np.nan]
    last = E["latest"]
    terms = " ".join(
        f"{'+' if v.BENEISH[k] >= 0 else '−'} {abs(v.BENEISH[k])}×{(last['idx'][k] if last['idx'][k] is not None else 1):.3f}"
        for k in keys
    )
    lines += [
        f"M = −4.84 {terms} = {last['m']:.2f}   (red above −1.78, amber above −2.22)",
        f"Accruals = (net income − operating cash flow) ÷ average assets = {p(E['accruals'], 1)}   (red above 10%, amber above 5%)",
        f"Cash conversion = Σ operating cash flow ÷ Σ net income, {E['cashYears']} years = {p(E['cashConv'], 0)}   (red below 50%, amber below 80%)",
    ]
    lines += [f"  {f['level'].upper()}: {f['msg']}" for f in E["flags"]]
    return E, tab, lines


# ---------------------------------------------------------- 2 cash flows
def step2_cash_flows(inp, scen="Base"):
    S = inp["scenarios"][scen]
    d = v.dcf(inp, S)
    rows, mdl = d["rows"], v.model_of(inp)
    cols = [r["label"] for r in rows]
    if mdl == "fcff":
        segs = v.seg_keys(inp)
        data = {f"Revenue: {s}": [r["seg"][s] for r in rows] for s in segs} if len(segs) > 1 else {}
        data.update(
            {
                "Revenue": [r["rev"] for r in rows],
                "Growth": [r["growth"] for r in rows],
                "EBIT": [r["ebit"] for r in rows],
                "EBIT margin": [r["margin"] for r in rows],
                "Tax rate": [r["tax"] for r in rows],
                "NOPAT": [r["nopat"] for r in rows],
                "+ D&A": [r["da"] for r in rows],
                "− Capex": [r["capex"] for r in rows],
                "− ΔNWC": [r["dnwc"] for r in rows],
                "FCFF": [r["fcff"] for r in rows],
            }
        )
        r0, op = rows[0], inp["operating"]
        lines = [
            f"Year 1 ({r0['label']}), {scen} case:",
            f"  Revenue = {m(r0['rev'])} (consensus or segment build); EBIT = Σ segment revenue × margin − corporate = {m(r0['ebit'])} ({p(r0['margin'], 1)})",
            f"  NOPAT = EBIT × (1 − tax) = {m(r0['ebit'])} × (1 − {p(r0['tax'], 1)}) = {m(r0['nopat'])}",
            f"  FCFF = NOPAT + D&A − capex − ΔNWC = {m(r0['nopat'])} + {m(r0['da'])} − {m(r0['capex'])} − {m(r0['dnwc'])} = {m(r0['fcff'])}",
            f"  D&A {p(op['da_pct'])} and capex {p(op['capex_pct'])} of revenue; ΔNWC = {p(op['nwc_pct'])} × change in revenue"
            + (
                f" (year 1: {p(op['nwc_pct'])} × {m(r0['rev'])} − opening NWC {m(op['nwc0'])})"
                if op.get("nwc0") is not None
                else ""
            ),
            f"  Growth after year 5 fades in a straight line to terminal g {p(inp['terminal']['g'])}; margins move from year 1 to the long-run level by year 5",
        ]
    elif mdl == "excess_return":
        data = {
            "ROE": [r["roe"] for r in rows],
            "Opening book": [r["bvBegin"] for r in rows],
            "Net income": [r["ni"] for r in rows],
            "− Equity charge (Ke × book)": [-r["charge"] for r in rows],
            "Excess return": [r["er"] for r in rows],
            "− Dividends + buybacks": [-r["dist"] for r in rows],
            "Closing book": [r["bvEnd"] for r in rows],
        }
        r0 = rows[0]
        lines = [
            f"Year 1 ({r0['label']}), {scen} case, {p(r0['frac'], 0)} of the year after the balance sheet:",
            f"  Net income = ROE × opening book × share of year = {p(r0['roe'], 1)} × {m(r0['bvBegin'])} × {r0['frac']:.3f} = {m(r0['ni'])}",
            f"  Equity charge = Ke × opening book × share = {p(d['ke'])} × {m(r0['bvBegin'])} × {r0['frac']:.3f} = {m(r0['charge'])}",
            f"  Excess return = {m(r0['ni'])} − {m(r0['charge'])} = {m(r0['er'])}",
            f"  Closing book = opening + net income × (1 − payout {p(inp['fin']['payout'], 1)}) = {m(r0['bvEnd'])}",
        ]
    else:
        data = {
            "Dividend growth": [r["g"] for r in rows],
            "Dividend per share": [r["dps"] for r in rows],
            "Share still to be paid": [r["frac"] for r in rows],
            "Dividend counted": [r["paid"] for r in rows],
        }
        r0 = rows[0]
        lines = [
            f"Dividend per share today {usd(inp['fin']['dps0'])}; year 1 = {usd(inp['fin']['dps0'])} × (1 + {p(r0['g'], 1)}) = {usd(r0['dps'])}",
            f"  Only {p(r0['frac'], 0)} of year 1 is still to be paid, so {usd(r0['paid'])} is counted",
            f"  Growth fades to terminal g {p(inp['terminal']['g'])} by {rows[-1]['label']}",
        ]
    return pd.DataFrame(data, index=cols).T, lines


# ------------------------------------------------------- 3 discount rate
def step3_discount_rate(inp):
    W, c = v.wacc(inp), inp["capital"]
    lines = [
        "Cost of equity (CAPM):",
        f"  1  Risk-free rate rf = 10-year Treasury = {p(c['rf'])}",
        f"  2  Expected market return Rm = {p(W['rm'])}",
        f"  3  Market risk premium = Rm − rf = {p(W['rm'])} − {p(c['rf'])} = {p(W['mrp'])}",
    ]
    st = None
    bd = c.get("beta_data") or {}
    if bd.get("dates"):
        s, mk = np.array(bd["stock"], float), np.array(bd["index"], float)
        rs, rm = s[1:] / s[:-1] - 1, mk[1:] / mk[:-1] - 1
        st = v.capm_beta(rs, rm)
        lines.append(
            f"  4a Daily regression: stock return = α + β × S&P 500 return, {st['n']} days {bd['dates'][0]} to {bd['dates'][-1]}"
        )
        lines.append(
            f"     β = {st['beta']:.4f}  (s.e. {st['seBeta']:.4f}, t {st['tBeta']:.1f}, R² {st['r2']:.3f}, 95% CI {st['ci95'][0]:.2f}–{st['ci95'][1]:.2f})"
        )
    src = c["beta_source"]
    how = {
        "daily_adj": "Blume-adjusted daily β = 0.67 × β + 0.33",
        "daily": "raw daily β",
        "blume": "Blume-adjusted published 5-year β",
        "raw": "published 5-year β",
        "bottom_up": "peer bottom-up β, relevered",
        "custom": "your β",
        "crsp_monthly": "CRSP monthly β",
        "crsp_monthly_adj": "Blume-adjusted CRSP monthly β",
    }.get(src, src)
    lines += [
        f"  4b β used = {W['beta']['chosen']:.4f} ({how})",
        f"  5  Equity risk premium for this stock = β × MRP = {W['beta']['chosen']:.4f} × {p(W['mrp'])} = {p(W['equityPremium'])}",
        f"  6  Ke = rf + β × MRP + CRP = {p(c['rf'])} + {p(W['equityPremium'])} + {p(c.get('crp', 0))} = {p(W['ke'])}",
    ]
    if v.is_fin(inp):
        lines.append(
            f"Discount rate = Ke = {p(W['ke'])} (banks, insurers and REITs are valued on equity cash flows)"
        )
    else:
        lines += [
            "Cost of capital:",
            f"  Kd after tax = {p(c['kd_pretax'])} × (1 − {p(c['tax_marginal'], 0)}) = {p(W['kd_after'])}   {inp.get('notes', {}).get('kd', '')}",
            f"  E = price × diluted shares = {m(W['E'])}; D = debt + leases = {m(W['D'])}",
            f"  WACC = E/(D+E) × Ke + D/(D+E) × Kd(1−t) = {p(W['we'], 1)} × {p(W['ke'])} + {p(W['wd'], 1)} × {p(W['kd_after'])} = {p(W['wacc'])}",
        ]
    return W, st, lines


def beta_plot(inp, ax=None):
    import matplotlib.pyplot as plt

    bd = inp["capital"].get("beta_data") or {}
    if not bd.get("dates"):
        return None
    s, mk = np.array(bd["stock"], float), np.array(bd["index"], float)
    rs, rm = s[1:] / s[:-1] - 1, mk[1:] / mk[:-1] - 1
    st = v.capm_beta(rs, rm)
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.scatter(rm * 100, rs * 100, s=6, alpha=0.5)
    x = np.linspace(rm.min(), rm.max(), 2)
    ax.plot(
        x * 100,
        (st["alpha"] + st["beta"] * x) * 100,
        color="C1",
        label=f"β = {st['beta']:.3f}, R² = {st['r2']:.2f}",
    )
    ax.set_xlabel("S&P 500 daily return, %")
    ax.set_ylabel(f"{inp['meta']['ticker']} daily return, %")
    ax.legend()
    return ax


# ------------------------------------------------------------ 4 discount
def step4_discount(inp, scen="Base"):
    d = v.dcf(inp, inp["scenarios"][scen])
    rows, mdl, tm = d["rows"], v.model_of(inp), inp["terminal"]
    r = d["wacc"]
    if mdl == "fcff":
        counted = [x["fcff"] * x["frac"] for x in rows]
    elif mdl == "excess_return":
        counted = [x["er"] for x in rows]
    else:
        counted = [x["paid"] for x in rows]
    tab = pd.DataFrame(
        {
            "Counted": counted,
            "Share of year": [x["frac"] for x in rows],
            "t (years)": [x["t"] for x in rows],
            "Discount factor (1+r)^−t": [x["df"] for x in rows],
            "Present value": [x["pv"] for x in rows],
        },
        index=[x["label"] for x in rows],
    ).T
    last = rows[-1]
    lines = [
        f"Discount rate r = {p(r)} ({'WACC' if mdl == 'fcff' else 'cost of equity'}); "
        f"{'mid-year' if inp['timing']['mid_year'] else 'year-end'} discounting; first year counts {p(rows[0]['frac'], 0)}",
        f"  Σ PV {rows[0]['label']}–{last['label']} = {m(d['pvExplicit'], 2 if mdl == 'ddm' else 0)}",
    ]
    if mdl == "fcff":
        lines += [
            f"  Terminal value = NOPAT × (1 + g) × (1 − g/RONIC) ÷ (WACC − g) = {m(last['nopat'])} × {1 + tm['g']:.3f} × (1 − {p(tm['g'], 1)}/{p(tm['ronic'], 0)}) ÷ ({p(r)} − {p(tm['g'], 1)}) = {m(d['tvGordon'])}",
            f"  PV of terminal value = {m(d['pvTV'])} ({p(d['tvShare'], 0)} of EV)   · exit-multiple check {tm['exit_multiple']}x EBITDA → {usd(d['perShareExit'])} a share",
            f"  Enterprise value = Σ PV + PV terminal = {m(d['pvExplicit'])} + {m(d['pvTV'])} = {m(d['ev'])}",
        ]
    elif mdl == "excess_return":
        lines += [
            f"  Terminal ROE = Ke + persistence × (long-run ROE − Ke) = {p(r)} + {p(tm.get('er_persistence', 1), 0)} × ({p(inp['scenarios'][scen]['roe'][1], 1)} − {p(r)}) = {p(d['roeTerminal'])}",
            f"  Terminal value = (terminal ROE − Ke) × closing book ÷ (Ke − g) = {m(d['tv'])}; PV = {m(d['pvTV'])}",
        ]
    else:
        lines += [
            f"  Terminal value = last dividend × (1 + g) ÷ (Ke − g) = {usd(last['dps'])} × {1 + tm['g']:.3f} ÷ ({p(r)} − {p(tm['g'], 1)}) = {usd(d['tv'])}; PV = {usd(d['pvTV'])}"
        ]
    return d, tab, lines


# --------------------------------------------------------------- 5 price
def step5_price(inp, scen="Base"):
    d = v.dcf(inp, inp["scenarios"][scen])
    mdl, b, sh = v.model_of(inp), inp.get("bridge", {}), v.diluted_shares(inp)
    if mdl == "fcff":
        items = [
            ("Enterprise value", d["ev"]),
            ("− Debt (ex leases)", -b["core_debt"]),
            ("− Leases", -b["leases"]),
            ("+ Cash", b["cash"]),
            ("+ Captive-finance net assets", d["dfsNet"]),
            ("+ Investments", b["investments"]),
            ("− Non-controlling interests", -b["nci"]),
            ("= Equity value (market cap)", d["equity"]),
        ]
    elif mdl == "excess_return":
        items = [
            ("Book equity", d["bv0"]),
            ("+ PV of excess returns", d["pvExplicit"]),
            ("+ PV of terminal", d["pvTV"]),
            ("= Equity value (market cap)", d["equity"]),
        ]
    else:
        items = [
            ("PV of dividends, per share", d["pvExplicit"]),
            ("+ PV of terminal, per share", d["pvTV"]),
            ("= Value per share", d["perShare"]),
            ("Equity value = value × diluted shares", d["equity"]),
        ]
    tab = pd.DataFrame(items, columns=["step", "value"]).set_index("step")
    price = inp["market"]["price"]
    lines = [
        f"Fair price = equity value ÷ diluted shares = {m(d['equity'])} ÷ {m(sh, 2)} = {usd(d['perShare'])}",
        f"Price today {usd(price)} → {p(d['perShare'] / price - 1, 1)} ({scen} case, {MODEL_NAME[mdl].split(' (')[0]} alone)",
    ]
    return tab, lines


# ------------------------------------------------------------- 6 verdict
def step6_verdict(inp, scen="Base", mc_n=5000):
    R = v.run_all(inp, scen, mc_n=mc_n)
    V, w = R["verdict"], inp["verdict"]["weights"]
    parts = {
        "dcf": R["base"]["perShare"],
        "mc": R["mc"]["median"],
        "comps": R["rel"]["blended"],
        "regression": R["reg"]["implied"],
    }
    names = {
        "dcf": MODEL_NAME[v.model_of(inp)].split(" (")[0],
        "mc": "Monte Carlo median",
        "comps": "Comps blend",
        "regression": "Peer regression",
    }
    tab = pd.DataFrame(
        [
            {
                "method": names[k],
                "value / share": parts[k],
                "weight": w.get(k, 0),
                "vs price": parts[k] / inp["market"]["price"] - 1
                if parts[k] is not None and np.isfinite(parts[k])
                else np.nan,
            }
            for k in parts
        ]
    ).set_index("method")
    comps = R["rel"]["table"]
    comps = comps if isinstance(comps, pd.DataFrame) else pd.DataFrame(comps).T
    lines = [
        f"Blended fair value = Σ weight × value ÷ Σ weight = {usd(V['fair'])}  vs price {usd(inp['market']['price'])} → {p(V['upside'], 1)}",
        f"Call (±{p(inp['verdict']['band'], 0)} band): {V['call']}"
        + (f"  (before the quality gate: {V['rawCall']})" if V.get("gated") else "")
        + ("  · low confidence (earnings quality C)" if V.get("confidence") == "low" else ""),
        f"Expected 1-year return = (fair ÷ price)^(1/{inp['verdict']['horizon_years']:g}) − 1 + dividend yield {p(V['divYield'])} = {p(V['expectedReturn'], 1)}",
        f"Required return Ke = {p(V['requiredReturn'])} → expected alpha = {p(V['alpha'], 1)}",
        f"Monte Carlo: {R['mc']['n']:,} runs, P(value > price) = {p(R['mc']['probAbove'], 0)}, p10–p90 {usd(R['mc']['p10'])}–{usd(R['mc']['p90'])}",
    ]
    lines += [f"  ! {x['msg']}" for x in V.get("warnings", [])]
    return R, tab, comps, lines


def sensitivity(inp, scen="Base"):
    S, W = inp["scenarios"][scen], v.wacc(inp)
    r0 = W["ke"] if v.is_fin(inp) else W["wacc"]
    rs = [r0 + d for d in (-0.02, -0.01, 0, 0.01, 0.02)]
    gs = [0.02, 0.025, 0.03, 0.035, 0.04]
    grid = v.sensitivity(inp, S, "wacc", rs, "g", gs)
    grid.index = [p(x) for x in rs]
    grid.columns = [p(x, 1) for x in gs]
    grid.index.name, grid.columns.name = ("Ke" if v.is_fin(inp) else "WACC"), "terminal g"
    rev = v.reverse(inp, S, "wacc", inp["terminal"]["g"] + 0.006, 0.3)
    return grid, [
        f"Reverse: the price {usd(inp['market']['price'])} implies a discount rate of {p(rev)} (model {p(r0)})"
    ]


def export_excel(inp, path, scen="Base"):
    import excel_model as X

    X.build_workbook(inp, path, scen)
    return path


def say(lines):
    print("\n".join(lines))


def drivers(inp):
    """The scenario drivers (Bear / Base / Bull) with where the Base values came from."""
    rows = []
    prov = inp.get("provenance") or {}

    def walk(o, path):
        if isinstance(o, dict):
            for k, x in o.items():
                walk(x, f"{path}.{k}")
        elif isinstance(o, list):
            for i, x in enumerate(o):
                walk(x, f"{path}.{i}")
        else:
            rows.append((path, o))

    for sn in ("Bear", "Base", "Bull"):
        walk(inp["scenarios"][sn], sn)
    df = pd.DataFrame(rows, columns=["key", "value"])
    df["scenario"] = df.key.str.split(".").str[0]
    df["driver"] = df.key.str.split(".", n=1).str[1]
    t = df.pivot(index="driver", columns="scenario", values="value")[["Bear", "Base", "Bull"]]
    t["Base comes from"] = [" ".join(prov.get(f"scenarios.Base.{d}", ["", ""])) for d in t.index]
    return t


PCT_ROWS = ("growth", "margin", "tax", "roe", "share", "rate", "vs price", "weight", "yield")


def fmt(df, per_share=False):
    """Readable tables: ratios as %, money as whole numbers (2 decimals for per-share)."""

    def rowfmt(name):
        n = str(name).lower()
        if any(k in n for k in PCT_ROWS):
            return "{:.1%}"
        if "factor" in n or n.startswith("t ("):
            return "{:.4f}"
        return "{:,.2f}" if per_share or "per share" in n or "dividend" in n else "{:,.0f}"

    sty = df.style
    for name in df.index:
        sty = sty.format(rowfmt(name), subset=pd.IndexSlice[[name], :], na_rep="—")
    return sty


def scenario_compare(inp):
    """Bear / Base / Bull side by side: the drivers that differ and what each case is worth."""
    price, out = inp["market"]["price"], {}
    for sn in ("Bear", "Base", "Bull"):
        d = v.dcf(inp, inp["scenarios"][sn])
        r0, mdl, col = d["rows"][0], v.model_of(inp), {}
        if mdl == "fcff":
            col["Revenue growth, year 2"] = p(d["rows"][1]["growth"], 1)
            col["EBIT margin, year 1 → year 5"] = (
                f"{p(d['rows'][0]['margin'], 1)} → {p(d['rows'][4]['margin'], 1)}"
            )
            col["FCFF, year 1 ($m)"] = m(r0["fcff"])
            col["Enterprise value ($m)"] = m(d["ev"])
        elif mdl == "excess_return":
            col["ROE, year 1 → long run"] = (
                f"{p(inp['scenarios'][sn]['roe'][0], 1)} → {p(inp['scenarios'][sn]['roe'][1], 1)}"
            )
            col["Justified P/B"] = f"{d['justifiedPB']:.2f}x"
        else:
            col["Dividend growth, years 1–4"] = p(inp["scenarios"][sn]["dps_growth"][0], 1)
        col["Terminal value share"] = p(d["tvShare"], 0)
        col["Value per share"] = usd(d["perShare"])
        col["vs price"] = p(d["perShare"] / price - 1, 1)
        out[sn] = col
    return pd.DataFrame(out)


def scenario_switch(inp):
    """Toggle buttons for Bear / Base / Bull; steps 2–6 re-run for the case you pick."""
    try:
        import ipywidgets as W
        from IPython.display import display
    except ImportError:
        print("pip install ipywidgets for the toggle; meanwhile:")
        return scenario_compare(inp)
    sw = W.ToggleButtons(options=["Bear", "Base", "Bull"], value="Base", description="Scenario")
    out = W.Output()

    def show(_=None):
        out.clear_output(wait=True)
        with out:
            sn = sw.value
            display(drivers(inp)[[sn, "Base comes from"]])
            cf, l2 = step2_cash_flows(inp, sn)
            display(fmt(cf, per_share=inp.get("model") == "ddm"))
            _, _, l4 = step4_discount(inp, sn)
            _, l5 = step5_price(inp, sn)
            _, _, _, l6 = step6_verdict(inp, sn, mc_n=1000)
            say([*l2, "", *l4, "", *l5, "", *l6])

    sw.observe(show, names="value")
    show()
    display(scenario_compare(inp))
    display(W.VBox([sw, out]))
