"""excel_model.py — the valuation framework as a live-formula Excel workbook.

Works for any company the inputs file describes (FCFF DCF for operating
companies, excess-return model for banks and insurers, dividend discount
model for REITs). The sheets follow the valuation in order:

    Dashboard    the result, the scenario switch, charts and how to use the workbook
    Inputs       every input: pulled value | your value | value used | source
    Scenarios    Bear / Base / Bull drivers; the scenario switch is on Inputs
    1 History    reported history and the earnings-quality check (Beneish, accruals)
    2 Cash flows the forecast that produces the cash flows
    3 Discount rate  rf, Rm, MRP, beta (SLOPE of daily returns), Ke, Kd, WACC
    4 Discount   cash flows × discount factors, terminal value, enterprise value
    5 Price      enterprise value → equity (market cap) → ÷ shares → price
    6 Verdict    model, comps and regression blended → call and expected alpha
    Comps        peer multiples and the peer regression, all as formulas
    Beta data    the daily closes behind the beta, with return formulas

Type into the yellow "Your value" column (or a blue cell on Scenarios) and the
whole workbook re-values. Monte Carlo is the only result written as a value
(it needs thousands of model runs); it is labelled where it is used.

The Python engine (valuation_v2.py) is the reference: tests/test_excel.py
recalculates the workbook in LibreOffice and checks every step against it.

    python excel_model.py inputs.json [out.xlsx] [Bear|Base|Bull]
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L
from openpyxl.workbook.defined_name import DefinedName

import valuation_v2 as v

N, EXPLICIT = v.N, v.EXPLICIT
FONT = "Arial"
F_IN = Font(name=FONT, color="0000FF")  # typed input
F_LINK = Font(name=FONT, color="008000")  # pulls from another sheet
F_TXT = Font(name=FONT)
F_B = Font(name=FONT, bold=True)
F_H = Font(name=FONT, bold=True, size=13)
F_NOTE = Font(name=FONT, italic=True, color="666666", size=9)
# palette: navy bands, one blue accent, soft yellow for your overrides
NAVY, ACCENT, INK, MUTED, RULE, PANEL = "1B2A41", "2E6DB4", "1F2937", "6B7280", "D9DEE5", "F3F5F8"
GREEN, RED, AMBER = "1E7B4A", "B42318", "B7791F"
FILL_Y = PatternFill("solid", fgColor="FFF4B8")
FILL_H = PatternFill("solid", fgColor=NAVY)
FILL_KEY = PatternFill("solid", fgColor="E7F0FA")
FILL_PANEL = PatternFill("solid", fgColor=PANEL)
FILL_SUB = PatternFill("solid", fgColor="E9EDF2")
TOP = Border(top=Side(style="thin"))
PCT, PCT1, MONEY, MONEY2, MULT, NUM, NUM3 = (
    "0.00%",
    "0.0%",
    '#,##0;(#,##0);"-"',
    '"$"#,##0.00',
    '0.0"x"',
    "#,##0.00",
    "0.000",
)
YC = 4  # first year column (D); A label, B total/scalar, C formula text
COLS = [L(YC + i) for i in range(N)]
SH = {
    "read": "Dashboard",
    "in": "Inputs",
    "sc": "Scenarios",
    "hist": "1 History",
    "cf": "2 Cash flows",
    "dr": "3 Discount rate",
    "disc": "4 Discount",
    "px": "5 Price",
    "ver": "6 Verdict",
    "comps": "Comps",
    "beta": "Beta data",
}


def q(sheet):
    return f"'{sheet}'"


def num(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


class Book:
    def __init__(self):
        self.wb = Workbook()
        self.ws = {}

    def sheet(self, key):
        if key not in self.ws:
            if not self.ws:
                ws = self.wb.active
                ws.title = SH[key]
            else:
                ws = self.wb.create_sheet(SH[key])
            self.ws[key] = ws
        return self.ws[key]

    def name(self, nm, key, cell):
        col = "".join(ch for ch in cell if ch.isalpha())
        row = "".join(ch for ch in cell if ch.isdigit())
        self.wb.defined_names[nm] = DefinedName(nm, attr_text=f"{q(SH[key])}!${col}${row}")
        return nm

    def name_range(self, nm, key, a, b):
        ca, ra = "".join(c for c in a if c.isalpha()), "".join(c for c in a if c.isdigit())
        cb, rb = "".join(c for c in b if c.isalpha()), "".join(c for c in b if c.isdigit())
        self.wb.defined_names[nm] = DefinedName(nm, attr_text=f"{q(SH[key])}!${ca}${ra}:${cb}${rb}")


def put(ws, r, c, val, font=None, fmt=None, fill=None, bold=False):
    cell = ws.cell(r, c, val)
    cell.font = font or (F_B if bold else F_TXT)
    if fmt:
        cell.number_format = fmt
    if fill:
        cell.fill = fill
    return cell


def header(ws, r, labels, start=1):
    for j, lab in enumerate(labels):
        put(ws, r, start + j, lab, F_B, fill=FILL_H)


def title(ws, text, sub):
    put(ws, 1, 1, text, F_H)
    put(ws, 2, 1, sub, F_NOTE)
    if ws.title not in ("Inputs", "Beta data", "Dashboard"):
        put(ws, 1, 3, '="Scenario: "&scen_name&"  ·  switch on Dashboard"', F_TXT)


# ------------------------------------------------------------------ Inputs
def inputs_sheet(B, inp, scen_name):
    ws = B.sheet("in")
    title(
        ws,
        "Inputs",
        "Yellow = type your own value to override the pulled one. Column D is what every formula uses.",
    )
    header(ws, 4, ["Input", "Pulled value", "Your value", "Value used", "Unit", "Origin", "Source / basis"])
    prov = inp.get("provenance", {})
    m, c, b, tm, tp = inp["market"], inp["capital"], inp.get("bridge", {}), inp["terminal"], inp["timing"]
    op = inp.get("operating", {})
    fin = v.is_fin(inp)
    rows = [
        ("Market", None),
        ("px_now", "Share price", m["price"], "$", "market.price"),
        ("shares_basic", "Shares outstanding", m["shares_basic"], "m", "market.shares_basic"),
        ("dilution", "Dilution (options, RSUs)", m["dilution"], "m", "market.dilution"),
        (
            "dps",
            "Dividend per share (for expected return)",
            m.get("dividend_ps", 0.0),
            "$",
            "market.dividend_ps",
        ),
        ("Cost of capital", None),
        ("rf", "Risk-free rate: 10-year Treasury yield", c["rf"], "%", "capital.rf"),
        (
            "rm",
            "Expected market return (Rm)",
            c["rm"] if c.get("rm") is not None else c["rf"] + c["erp"],
            "%",
            "capital.rm",
        ),
        ("crp", "Country risk premium", c.get("crp", 0.0), "%", "capital.crp"),
        ("size_prem", "Size premium", c.get("size_premium", 0.0), "%", "capital.size_premium"),
        (
            "beta_src",
            "Beta used: daily_adj, daily, blume, raw, crsp_monthly(_adj), bottom_up, custom",
            c["beta_source"],
            "",
            "capital.beta_source",
        ),
        ("beta_raw", "Beta, published 5-year monthly", c["beta_raw"], "", "capital.beta_raw"),
        (
            "beta_daily_in",
            "Beta, daily regression (used only if Beta data is empty)",
            num(c.get("beta_daily")),
            "",
            "capital.beta_daily",
        ),
        (
            "beta_crsp",
            "Beta, CRSP monthly regression",
            num(c.get("beta_crsp_monthly")),
            "",
            "capital.beta_crsp_monthly",
        ),
        ("beta_custom", "Beta, custom", c.get("beta_custom"), "", "capital.beta_custom"),
        (
            "beta_bu",
            "Beta, bottom-up from peers (relevered)",
            None if fin else v.bottom_up_beta(inp)["relevered"],
            "",
            None,
        ),
        ("kd", "Pre-tax cost of debt", c["kd_pretax"], "%", "capital.kd_pretax"),
        ("t_marg", "Marginal tax rate", c["tax_marginal"], "%", "capital.tax_marginal"),
        ("Timing and terminal", None),
        ("g", "Terminal growth", tm["g"], "%", "terminal.g"),
        (
            "first_period",
            "Years from today to the first fiscal year end",
            tp["first_period"],
            "yrs",
            "timing.first_period",
        ),
        ("stub", "Share of the first year after the last balance sheet", tp["stub"], "", "timing.stub"),
        (
            "midyear",
            "Mid-year convention (1 = yes, 0 = year-end)",
            1 if tp["mid_year"] else 0,
            "",
            "timing.mid_year",
        ),
        ("Verdict", None),
        ("band", "Fairly-valued band (±)", inp["verdict"]["band"], "%", "verdict.band"),
        (
            "horizon",
            "Years for price to converge to value",
            inp["verdict"]["horizon_years"],
            "yrs",
            "verdict.horizon_years",
        ),
        (
            "reg_min",
            "Peer regression used only if R² is at least",
            v.reg_min_r2(inp),
            "",
            "relative.reg_min_r2",
        ),
        (
            "eq_gate",
            "Earnings-quality gate (1 = grade D withholds the call)",
            0 if inp["verdict"].get("quality_gate") is False else 1,
            "",
            None,
        ),
    ]
    if fin:
        f = inp["fin"]
        rows += [
            ("Equity", None),
            ("fin_bv0", "Common book equity", f.get("bv0"), "$m", "fin.bv0"),
            ("payout", "Payout: (dividends + buybacks) ÷ earnings", f.get("payout"), "%", "fin.payout"),
            (
                "persist",
                "Share of the long-run ROE spread over Ke kept forever",
                tm.get("er_persistence", 1.0),
                "%",
                "terminal.er_persistence",
            ),
            ("fin_dps0", "Dividend per share now (annualised)", f.get("dps0"), "$", "fin.dps0"),
            ("roe_ttm", "Trailing ROE (for the P/B regression)", f.get("roe_ttm"), "%", "fin.roe_ttm"),
        ]
    else:
        y1 = inp.get("y1_segments") or inp["fy27_segments"]
        rows += [
            ("Operating", None),
            ("rev_y0", "Last fiscal-year revenue", inp["rev_y0"], "$m", "rev_y0"),
            ("corp_pct", "Corporate costs % revenue", op["corp_pct"], "%", "operating.corp_pct"),
            ("tax_y1", "Tax rate, year 1", op["tax0"], "%", "operating.tax0"),
            ("tax_lr", "Tax rate, long run (from year 5)", op["tax_lr"], "%", "operating.tax_lr"),
            ("da_pct", "D&A % revenue", op["da_pct"], "%", "operating.da_pct"),
            ("capex_pct", "Capex % revenue", op["capex_pct"], "%", "operating.capex_pct"),
            ("nwc_pct", "ΔNWC % of Δrevenue", op["nwc_pct"], "%", "operating.nwc_pct"),
            *(
                [
                    (
                        "nwc_open",
                        "Working capital at the last balance sheet",
                        op["nwc0"],
                        "$m",
                        "operating.nwc0",
                    )
                ]
                if op.get("nwc0") is not None
                else []
            ),
            ("ronic", "Return on new invested capital (terminal)", tm["ronic"], "%", "terminal.ronic"),
            (
                "exit_mult",
                "Exit EV/EBITDA multiple (cross-check)",
                tm["exit_multiple"],
                "x",
                "terminal.exit_multiple",
            ),
            ("tv_method", "Terminal method: gordon or exit", tm["method"], "", "terminal.method"),
        ]
        for s in v.seg_keys(inp):
            rows.append(
                (f"segy1_{s}", f"Year-1 revenue: {seg_name(inp, s)}", y1[s], "$m", f"y1_segments.{s}")
            )
        rows += [
            ("Equity bridge", None),
            ("cash", "Cash and equivalents", b.get("cash"), "$m", "bridge.cash"),
            ("core_debt", "Debt excluding leases", b.get("core_debt"), "$m", "bridge.core_debt"),
            ("leases", "Lease liabilities", b.get("leases", 0.0), "$m", "bridge.leases"),
            (
                "dfs_rec",
                "Captive-finance receivables",
                b.get("dfs_receivables", 0.0),
                "$m",
                "bridge.dfs_receivables",
            ),
            (
                "dfs_le",
                "Captive-finance lease equipment",
                b.get("dfs_lease_equipment", 0.0),
                "$m",
                "bridge.dfs_lease_equipment",
            ),
            ("dfs_debt", "Captive-finance debt", b.get("dfs_debt", 0.0), "$m", "bridge.dfs_debt"),
            (
                "investments",
                "Non-operating investments",
                b.get("investments", 0.0),
                "$m",
                "bridge.investments",
            ),
            ("nci", "Non-controlling interests", b.get("nci", 0.0), "$m", "bridge.nci"),
        ]
    r = 5
    for row in rows:
        if row[1] is None:
            put(ws, r, 1, row[0], F_B)
            r += 1
            continue
        nm, label, val, unit, path = row
        p = prov.get(path) if path else None
        put(ws, r, 1, label)
        fmt = (
            PCT if unit == "%" else NUM3 if "beta" in nm else NUM if unit in ("$", "yrs", "x", "") else MONEY
        )
        put(ws, r, 2, val, F_IN, fmt)
        put(ws, r, 3, None, F_IN, fmt, FILL_Y)
        put(ws, r, 4, f"=IF(ISBLANK(C{r}),B{r},C{r})", F_TXT, fmt)
        put(ws, r, 5, unit)
        put(ws, r, 6, p[0] if p else ("formula" if nm == "beta_bu" else "setting"))
        put(
            ws,
            r,
            7,
            p[1]
            if p
            else ("median unlevered peer beta, relevered at this company's D/E" if nm == "beta_bu" else ""),
        )
        B.name(nm, "in", f"D{r}")
        r += 1
    for col, w in zip("ABCDEFG", (58, 14, 12, 14, 6, 11, 100), strict=True):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "B5"


def seg_name(inp, s):
    return next((d.get("name") or s for d in v.defs(inp) if d["key"] == s), s)


# --------------------------------------------------------------- Scenarios
def scenarios_sheet(B, inp):
    ws = B.sheet("sc")
    title(
        ws,
        "Scenarios",
        "Blue cells are the drivers for each case; column E is the case picked by the switch on the Dashboard.",
    )
    header(ws, 4, ["Driver", "Bear", "Base", "Bull", "Used", "Notes"])
    sc, order, prov = inp["scenarios"], ["Bear", "Base", "Bull"], inp.get("provenance", {})
    r = [5]

    def drv(nm, label, vals, path):
        rr = r[0]
        put(ws, rr, 1, label)
        for j, x in enumerate(vals):
            put(ws, rr, 2 + j, x, F_IN, PCT1)
        put(ws, rr, 5, f"=CHOOSE(scen_sel,B{rr},C{rr},D{rr})", F_TXT, PCT1, FILL_KEY)
        p = prov.get(path.format("Base"))
        put(ws, rr, 6, p[1] if p else "", F_NOTE)
        B.name(nm, "sc", f"E{rr}")
        r[0] += 1

    mdl = v.model_of(inp)
    if mdl == "excess_return":
        drv("s_roe_0", "ROE, year 1", [sc[k]["roe"][0] for k in order], "scenarios.{}.roe.0")
        drv("s_roe_1", "ROE, long run (from year 5)", [sc[k]["roe"][1] for k in order], "scenarios.{}.roe.1")
    elif mdl == "ddm":
        for j in range(EXPLICIT):
            drv(
                f"s_dg_{j}",
                f"Dividend growth, year {j + 1}",
                [sc[k]["dps_growth"][j] for k in order],
                f"scenarios.{{}}.dps_growth.{j}",
            )
    else:
        for s in v.seg_keys(inp):
            for j in range(EXPLICIT):
                drv(
                    f"s_g_{s}_{j}",
                    f"Revenue growth: {seg_name(inp, s)}, year {j + 2}",
                    [sc[k]["growth"][s][j] for k in order],
                    f"scenarios.{{}}.growth.{s}.{j}",
                )
        for gname in sc["Base"]["margins"]:
            lab = (inp.get("margin_groups") or {}).get(gname, gname)
            drv(
                f"s_m0_{gname}",
                f"EBIT margin: {lab}, year 1",
                [sc[k]["margins"][gname][0] for k in order],
                f"scenarios.{{}}.margins.{gname}.0",
            )
            drv(
                f"s_m1_{gname}",
                f"EBIT margin: {lab}, long run (from year 5)",
                [sc[k]["margins"][gname][1] for k in order],
                f"scenarios.{{}}.margins.{gname}.1",
            )
    ws.column_dimensions["A"].width = 52
    ws.column_dimensions["F"].width = 80
    ws.freeze_panes = "B5"


# ---------------------------------------------------------------- 1 History
def history_sheet(B, inp):
    ws = B.sheet("hist")
    title(
        ws,
        "Step 1 · Historic performance",
        "Reported figures ($m) as pulled; growth and margins are formulas. Earnings quality below.",
    )
    h = inp.get("history", {}).get("annual", {})
    names = inp.get("history", {}).get("annual_names", {})
    r = 4
    if h.get("labels"):
        labs = h["labels"]
        header(ws, r, ["Line", *labs])
        r += 1
        rowof = {}
        for k, vals in h.items():
            if k == "labels":
                continue
            put(ws, r, 1, names.get(k, k))
            for j, x in enumerate(vals):
                put(ws, r, 2 + j, num(x), F_IN, NUM if "eps" in k else MONEY)
            rowof[k] = r
            r += 1
        rev = rowof.get("revenue")
        if rev:
            put(ws, r, 1, "Revenue growth")
            for j in range(1, len(labs)):
                a, b_ = f"{L(2 + j - 1)}{rev}", f"{L(2 + j)}{rev}"
                if re.search(r"H\d|Q\d|TTM", str(labs[j])) or re.search(r"H\d|Q\d|TTM", str(labs[j - 1])):
                    continue  # a part-year column is not comparable with a full year
                put(ws, r, 2 + j, f'=IF(AND(ISNUMBER({a}),ISNUMBER({b_})),{b_}/{a}-1,"")', F_TXT, PCT1)
            r += 1
            for k in ("op_income_gaap", "op_income", "net_income", "cfo"):
                if k in rowof:
                    put(ws, r, 1, f"{names.get(k, k)} ÷ revenue")
                    for j in range(len(labs)):
                        a, b_ = f"{L(2 + j)}{rowof[k]}", f"{L(2 + j)}{rev}"
                        put(ws, r, 2 + j, f'=IF(AND(ISNUMBER({a}),ISNUMBER({b_})),{a}/{b_},"")', F_TXT, PCT1)
                    r += 1
        put(ws, r, 1, inp.get("history", {}).get("annual_note", ""), F_NOTE)
        r += 2
    quality_block(B, ws, inp, r)
    ws.column_dimensions["A"].width = 48
    for j in range(2, 16):
        ws.column_dimensions[L(j)].width = 13


def quality_block(B, ws, inp, r):
    """Beneish M-score, Sloan accruals and cash conversion, as formulas on the pulled statement lines."""
    qd = inp.get("quality_data")
    put(ws, r, 1, "Earnings quality: are the earnings backed by cash?", F_H)
    r += 1
    if v.model_of(inp) == "excess_return" or not qd or len(qd.get("labels", [])) < 2:
        put(
            ws,
            r,
            1,
            "Not graded: "
            + (
                "banks and insurers have no cost of sales, and their operating cash flow includes lending flows."
                if v.model_of(inp) == "excess_return"
                else "no statement history."
            ),
            F_NOTE,
        )
        put(ws, r + 1, 1, "Grade")
        put(ws, r + 1, 2, "n/a", F_B)
        B.name("eq_grade", "hist", f"B{r + 1}")
        return
    labs, n = qd["labels"], len(qd["labels"])
    header(ws, r, ["Statement line ($m)", *labs])
    r += 1
    lines = [
        ("sales", "Revenue"),
        ("cogs", "Cost of revenue"),
        ("sga", "SG&A"),
        ("rec", "Receivables"),
        ("ca", "Current assets"),
        ("ppe", "Net PP&E"),
        ("ta", "Total assets"),
        ("cl", "Current liabilities"),
        ("ltd", "Long-term debt"),
        ("dep", "Depreciation & amortisation"),
        ("ni", "Net income (continuing)"),
        ("cfo", "Operating cash flow"),
    ]
    R = {}
    for k, lab in lines:
        put(ws, r, 1, lab)
        for j in range(n):
            put(ws, r, 2 + j, num(qd.get(k, [None] * n)[j]), F_IN, MONEY)
        R[k] = r
        r += 1
    put(ws, r, 1, inp.get("provenance", {}).get("quality_data", ["", ""])[1], F_NOTE)
    r += 2
    header(
        ws,
        r,
        [
            "Beneish index (blank = input missing; set to 1 in the score)",
            *labs[1:],
            "Coefficient",
            "Manipulators' mean",
        ],
    )
    r += 1

    def cl(k, j):
        return f"{L(2 + j)}{R[k]}"

    def guarded(cells, expr):
        return f"=IF(COUNT({','.join(cells)})={len(cells)},{expr},NA())"

    idx_rows = {}
    spec = {
        "DSRI": (["rec", "sales"], "({rec_t}/{sales_t})/({rec_p}/{sales_p})"),
        "GMI": (["sales", "cogs"], "(({sales_p}-{cogs_p})/{sales_p})/(({sales_t}-{cogs_t})/{sales_t})"),
        "AQI": (["ca", "ppe", "ta"], "(1-({ca_t}+{ppe_t})/{ta_t})/(1-({ca_p}+{ppe_p})/{ta_p})"),
        "SGI": (["sales"], "{sales_t}/{sales_p}"),
        "DEPI": (["dep", "ppe"], "({dep_p}/({dep_p}+{ppe_p}))/({dep_t}/({dep_t}+{ppe_t}))"),
        "SGAI": (["sga", "sales"], "({sga_t}/{sales_t})/({sga_p}/{sales_p})"),
        "LVGI": (["cl", "ltd", "ta"], "(({cl_t}+{ltd_t})/{ta_t})/(({cl_p}+{ltd_p})/{ta_p})"),
        "TATA": (["ni", "cfo", "ta"], None),
    }
    names = {
        "DSRI": "Days sales in receivables",
        "GMI": "Gross margin, prior ÷ current",
        "AQI": "Asset quality",
        "SGI": "Sales growth",
        "DEPI": "Depreciation rate, prior ÷ current",
        "SGAI": "SG&A ÷ sales",
        "LVGI": "Leverage",
        "TATA": "Accruals ÷ total assets",
    }
    for k, (need, expr) in spec.items():
        put(ws, r, 1, f"{k}: {names[k]}")
        for j in range(1, n):
            sub = {}
            for nd in need:
                sub[f"{nd}_t"], sub[f"{nd}_p"] = cl(nd, j), cl(nd, j - 1)
            if k == "TATA":
                f = guarded(
                    [cl("ni", j), cl("cfo", j), cl("ta", j)], f"({cl('ni', j)}-{cl('cfo', j)})/{cl('ta', j)}"
                )
            else:
                cells = [sub[f"{nd}_t"] for nd in need] + [sub[f"{nd}_p"] for nd in need]
                f = guarded(cells, expr.format(**sub))
            put(ws, r, 1 + j, f, F_TXT, NUM3)
        put(ws, r, 1 + n, v.BENEISH[k], F_IN)
        put(ws, r, 2 + n, v.BENEISH_MANIP[k], F_NOTE)
        idx_rows[k] = r
        r += 1
    put(ws, r, 1, "M-score = −4.84 + Σ coefficient × index", F_B)
    for j in range(1, n):
        c = L(1 + j)
        terms = "+".join(
            f"{L(1 + n)}{idx_rows[k]}*IFERROR({c}{idx_rows[k]},1)" for k in idx_rows if k != "TATA"
        )
        nerr = "+".join(f"ISERROR({c}{idx_rows[k]})" for k in idx_rows if k != "TATA")
        f = f"=IF(OR(ISERROR({c}{idx_rows['TATA']}),({nerr})>2),NA(),{v.BENEISH['c']}+{terms}+{L(1 + n)}{idx_rows['TATA']}*{c}{idx_rows['TATA']})"
        put(ws, r, 1 + j, f, F_B, "0.00")
    m_cell = f"{L(n)}{r}"
    B.name("eq_m", "hist", m_cell)
    r += 2
    last, prev = L(1 + n), L(n)
    k = min(3, n)
    first = L(2 + n - k)
    tests = [
        (
            "eq_accr",
            "Accruals ÷ average total assets (Sloan): (NI − CFO) ÷ avg assets",
            f"=IF(COUNT({last}{R['ni']},{last}{R['cfo']},{last}{R['ta']},{prev}{R['ta']})=4,({last}{R['ni']}-{last}{R['cfo']})/(({last}{R['ta']}+{prev}{R['ta']})/2),NA())",
            PCT1,
        ),
        (
            "eq_cc",
            f"Operating cash flow ÷ net income, last {k} years",
            f"=IF(AND(COUNT({first}{R['ni']}:{last}{R['ni']})={k},COUNT({first}{R['cfo']}:{last}{R['cfo']})={k},SUM({first}{R['ni']}:{last}{R['ni']})>0),SUM({first}{R['cfo']}:{last}{R['cfo']})/SUM({first}{R['ni']}:{last}{R['ni']}),NA())",
            "0%",
        ),
        (
            "eq_red",
            "Red flags: M > −1.78 · accruals > 10% · cash conversion < 50%",
            "=IF(ISNUMBER(eq_m),--(eq_m>-1.78),0)+IF(ISNUMBER(eq_accr),--(eq_accr>0.1),0)+IF(ISNUMBER(eq_cc),--(eq_cc<0.5),0)",
            "0",
        ),
        (
            "eq_amber",
            "Amber flags: M > −2.22 · accruals > 5% · cash conversion < 80%",
            "=IF(ISNUMBER(eq_m),--AND(eq_m>-2.22,eq_m<=-1.78),0)+IF(ISNUMBER(eq_accr),--AND(eq_accr>0.05,eq_accr<=0.1),0)+IF(ISNUMBER(eq_cc),--AND(eq_cc>=0.5,eq_cc<0.8),0)",
            "0",
        ),
        (
            "eq_grade",
            "Grade (A none · B one amber · C one red or two ambers · D two reds)",
            '=IF(AND(NOT(ISNUMBER(eq_m)),NOT(ISNUMBER(eq_accr)),NOT(ISNUMBER(eq_cc))),"n/a",IF(eq_red>=2,"D",IF(OR(eq_red=1,eq_amber>=2),"C",IF(eq_amber=1,"B","A"))))',
            None,
        ),
    ]
    for nm, lab, f, fmt in tests:
        put(ws, r, 1, lab, F_B if nm == "eq_grade" else F_TXT)
        put(ws, r, 2, f, F_B if nm == "eq_grade" else F_TXT, fmt, FILL_KEY if nm == "eq_grade" else None)
        B.name(nm, "hist", f"B{r}")
        r += 1
    put(
        ws,
        r,
        1,
        "Beneish (1999): M above −1.78 marked firms later caught manipulating. A flag is a reason to read the notes, not proof.",
        F_NOTE,
    )


# ------------------------------------------------------------- 2 Cash flows
def year_header(ws, inp, r, label):
    put(ws, r, 1, label, F_B, fill=FILL_H)
    put(ws, r, 2, "Total", F_B, fill=FILL_H)
    put(ws, r, 3, "How it is calculated", F_B, fill=FILL_H)
    for i, lab in enumerate(inp["meta"]["year_labels"][:N]):
        put(ws, r, YC + i, lab, F_B, fill=FILL_H)
    put(ws, r + 1, 1, "Year index i", F_NOTE)
    for i in range(N):
        put(ws, r + 1, YC + i, i, F_NOTE)
    return r + 1  # row holding i


def cashflow_sheet(B, inp):
    ws = B.sheet("cf")
    mdl = v.model_of(inp)
    sub = {
        "fcff": "Revenue by segment → EBIT → NOPAT → + D&A − capex − ΔNWC = unlevered free cash flow ($m)",
        "excess_return": "Book equity rolls forward by retained earnings; excess return = net income − Ke × opening book ($m)",
        "ddm": "Dividend per share grows at the scenario rates, then fades to terminal growth ($ per share)",
    }[mdl]
    title(ws, "Step 2 · Future cash flows", sub)
    ir = year_header(ws, inp, 4, "Line")
    r = ir + 1
    rows = {}

    def line(key, label, how, f_of_i, fmt=MONEY, bold=False, total=False):
        nonlocal r
        put(ws, r, 1, label, F_B if bold else F_TXT)
        put(ws, r, 3, how, F_NOTE)
        for i in range(N):
            put(ws, r, YC + i, f_of_i(i, COLS[i]), F_B if bold else F_TXT, fmt)
        if total:
            put(ws, r, 2, f"=SUM({COLS[0]}{r}:{COLS[-1]}{r})", F_TXT, fmt)
        rows[key] = r
        r += 1

    P = lambda i: COLS[i - 1]
    if mdl == "fcff":
        segs, D = v.seg_keys(inp), v.defs(inp)
        fade = N - 1 - EXPLICIT
        for s in segs:

            def gf(i, c, s=s):
                if i == 0:
                    return None
                if i <= EXPLICIT:
                    return f"=s_g_{s}_{i - 1}"
                return f"=s_g_{s}_{EXPLICIT - 1}+(g-s_g_{s}_{EXPLICIT - 1})*({c}${ir}-{EXPLICIT})/{fade}"

            line(
                f"g_{s}",
                f"Growth: {seg_name(inp, s)}",
                f"years 2–5 from Scenarios; then linear fade to terminal g over {fade} years",
                gf,
                PCT1,
            )
            line(
                f"rev_{s}",
                f"Revenue: {seg_name(inp, s)}",
                "year 1 from Inputs; then prior × (1 + growth)",
                lambda i, c, s=s: (
                    f"=segy1_{s}" if i == 0 else f"={P(i)}{rows.get(f'rev_{s}', r)}*(1+{c}{rows[f'g_{s}']})"
                ),
            )
        for gname in inp["scenarios"]["Base"]["margins"]:
            lab = (inp.get("margin_groups") or {}).get(gname, gname)
            line(
                f"m_{gname}",
                f"EBIT margin: {lab}",
                f"year 1 → long run in a straight line over {EXPLICIT} years, flat after",
                lambda i, c, gname=gname: (
                    f"=IF({c}${ir}>={EXPLICIT},s_m1_{gname},s_m0_{gname}+(s_m1_{gname}-s_m0_{gname})*{c}${ir}/{EXPLICIT})"
                ),
                PCT1,
            )
        line(
            "rev",
            "Revenue",
            "sum of segments",
            lambda i, c: "=" + "+".join(f"{c}{rows['rev_' + s]}" for s in segs),
            bold=True,
            total=True,
        )
        line(
            "growth",
            "Revenue growth",
            "revenue ÷ prior − 1 (year 1 vs last reported year)",
            lambda i, c: (
                f"={c}{rows['rev']}/rev_y0-1" if i == 0 else f"={c}{rows['rev']}/{P(i)}{rows['rev']}-1"
            ),
            PCT1,
        )
        line(
            "oi",
            "Segment operating income",
            "Σ segment revenue × its margin group",
            lambda i, c: (
                "="
                + (
                    "+".join(
                        f"{c}{rows['rev_' + d['key']]}*{c}{rows['m_' + d['margin']]}"
                        for d in D
                        if d["margin"]
                    )
                    or "0"
                )
            ),
        )
        line(
            "corp",
            "− Corporate costs (SBC, amortisation, other)",
            "corporate % × revenue",
            lambda i, c: f"=corp_pct*{c}{rows['rev']}",
        )
        line(
            "ebit",
            "EBIT",
            "segment OI − corporate costs",
            lambda i, c: f"={c}{rows['oi']}-{c}{rows['corp']}",
            bold=True,
            total=True,
        )
        line(
            "margin",
            "EBIT margin",
            "EBIT ÷ revenue",
            lambda i, c: f"={c}{rows['ebit']}/{c}{rows['rev']}",
            PCT1,
        )
        line(
            "tax",
            "Tax rate",
            f"year 1 → long run over {EXPLICIT} years",
            lambda i, c: f"=IF({c}${ir}>={EXPLICIT},tax_lr,tax_y1+(tax_lr-tax_y1)*{c}${ir}/{EXPLICIT})",
            PCT1,
        )
        line("nopat", "NOPAT", "EBIT × (1 − tax)", lambda i, c: f"={c}{rows['ebit']}*(1-{c}{rows['tax']})")
        line("da", "+ D&A", "D&A % × revenue", lambda i, c: f"=da_pct*{c}{rows['rev']}")
        line("capex", "− Capex", "capex % × revenue", lambda i, c: f"=capex_pct*{c}{rows['rev']}")
        has_nwc0 = inp["operating"].get("nwc0") is not None
        line(
            "dnwc",
            "− Change in working capital",
            "yr 1: NWC % × revenue − opening NWC; then NWC % × Δrevenue"
            if has_nwc0
            else "ΔNWC % × (revenue − prior revenue)",
            lambda i, c: (
                (f"=nwc_pct*{c}{rows['rev']}-nwc_open" if has_nwc0 else f"=nwc_pct*({c}{rows['rev']}-rev_y0)")
                if i == 0
                else f"=nwc_pct*({c}{rows['rev']}-{P(i)}{rows['rev']})"
            ),
        )
        line(
            "fcff",
            "Unlevered free cash flow (FCFF)",
            "NOPAT + D&A − capex − ΔNWC",
            lambda i, c: f"={c}{rows['nopat']}+{c}{rows['da']}-{c}{rows['capex']}-{c}{rows['dnwc']}",
            bold=True,
            total=True,
        )
        line("ebitda", "EBITDA", "EBIT + D&A", lambda i, c: f"={c}{rows['ebit']}+{c}{rows['da']}")
        B.name("cf_last_nopat", "cf", f"{COLS[-1]}{rows['nopat']}")
        B.name("cf_last_ebitda", "cf", f"{COLS[-1]}{rows['ebitda']}")
        cf_row = rows["fcff"]
    elif mdl == "excess_return":
        line(
            "roe",
            "ROE",
            f"year 1 → long run in a straight line over {EXPLICIT} years (Scenarios)",
            lambda i, c: f"=IF({c}${ir}>={EXPLICIT},s_roe_1,s_roe_0+(s_roe_1-s_roe_0)*{c}${ir}/{EXPLICIT})",
            PCT1,
        )
        line(
            "frac",
            "Share of the year counted",
            "year 1: the part after the last balance sheet",
            lambda i, c: "=stub" if i == 0 else "=1",
            PCT1,
        )
        line(
            "bv",
            "Opening book equity",
            "year 1: book at the balance sheet; then prior closing book",
            lambda i, c: "=fin_bv0" if i == 0 else f"={P(i)}{r + 4}",
        )
        line(
            "ni",
            "Net income to common",
            "ROE × opening book × share of year",
            lambda i, c: f"={c}{rows['roe']}*{c}{rows['bv']}*{c}{rows['frac']}",
            total=True,
        )
        line(
            "charge",
            "− Equity charge",
            "Ke × opening book × share of year",
            lambda i, c: f"=ke*{c}{rows['bv']}*{c}{rows['frac']}",
        )
        line(
            "dist", "− Dividends and buybacks", "payout × net income", lambda i, c: f"=payout*{c}{rows['ni']}"
        )
        line(
            "bvend",
            "Closing book equity",
            "opening + net income − distributions",
            lambda i, c: f"={c}{rows['bv']}+{c}{rows['ni']}-{c}{rows['dist']}",
        )
        line(
            "er",
            "Excess return",
            "net income − equity charge",
            lambda i, c: f"={c}{rows['ni']}-{c}{rows['charge']}",
            bold=True,
            total=True,
        )
        assert rows["bvend"] == rows["bv"] + 4
        B.name("cf_last_bv", "cf", f"{COLS[-1]}{rows['bvend']}")
        cf_row = rows["er"]
    else:
        e3 = f"s_dg_{EXPLICIT - 1}"
        line(
            "g",
            "Dividend growth",
            f"years 1–{EXPLICIT} from Scenarios; then linear fade to terminal g by year {N}",
            lambda i, c: (
                f"=s_dg_{i}" if i < EXPLICIT else f"={e3}+(g-{e3})*({c}${ir}-{EXPLICIT - 1})/{N - EXPLICIT}"
            ),
            PCT1,
        )
        line(
            "dps",
            "Dividend per share",
            "prior × (1 + growth); year 1 from today's annualised dividend",
            lambda i, c: (
                f"=fin_dps0*(1+{c}{rows['g']})"
                if i == 0
                else f"={P(i)}{rows.get('dps', r)}*(1+{c}{rows['g']})"
            ),
            MONEY2,
        )
        line(
            "frac",
            "Share still to be paid",
            "year 1: the part of the fiscal year still ahead",
            lambda i, c: "=MIN(1,MAX(0,first_period))" if i == 0 else "=1",
            PCT1,
        )
        line(
            "paid",
            "Dividend counted",
            "dividend × share still to be paid",
            lambda i, c: f"={c}{rows['dps']}*{c}{rows['frac']}",
            MONEY2,
            bold=True,
            total=True,
        )
        B.name("cf_last_dps", "cf", f"{COLS[-1]}{rows['dps']}")
        cf_row = rows["paid"]
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 13
    ws.column_dimensions["C"].width = 52
    for c in COLS:
        ws.column_dimensions[c].width = 12
    ws.freeze_panes = f"{COLS[0]}6"
    return rows, cf_row, ir


# ---------------------------------------------------------- 3 Discount rate
def discount_rate_sheet(B, inp):
    ws = B.sheet("dr")
    fin = v.is_fin(inp)
    title(
        ws,
        "Step 3 · Discount rate",
        "Cost of equity from CAPM with a beta regressed on daily returns; then debt and WACC."
        + (" This model discounts at the cost of equity: WACC is shown for reference only." if fin else ""),
    )
    header(ws, 4, ["Item", "Value", "How it is calculated"])
    have_bd = bool((inp["capital"].get("beta_data") or {}).get("dates"))
    rows = [
        ("Cost of equity", None, None, None),
        ("rf_x", "1 · Risk-free rate (10-year Treasury)", "=rf", "Inputs"),
        ("rm_x", "2 · Expected market return (Rm)", "=rm", "Inputs"),
        ("mrp", "3 · Market risk premium = Rm − rf", "=rm-rf", "Rm − rf"),
        (
            "beta_reg",
            "4a · Daily regression beta = SLOPE(stock returns, index returns)",
            "=SLOPE(beta_rs,beta_rm)" if have_bd else "=beta_daily_in",
            "OLS slope on the Beta data sheet" if have_bd else "no price data: value from Inputs",
        ),
        (
            "beta_r2",
            "     R² of the regression",
            "=RSQ(beta_rs,beta_rm)" if have_bd else '=""',
            "share of the stock's daily moves explained by the market",
        ),
        (
            "beta_se",
            "     Standard error of beta",
            "=STEYX(beta_rs,beta_rm)/SQRT(DEVSQ(beta_rm))" if have_bd else '=""',
            "s ÷ √Σ(x − x̄)²",
        ),
        ("beta_n", "     Number of daily returns", "=COUNT(beta_rs)" if have_bd else '=""', ""),
        (
            "beta_daily",
            "     Daily beta carried forward",
            "=IF(ISNUMBER(beta_reg),beta_reg,beta_daily_in)",
            "",
        ),
        (
            "beta_used",
            "4b · Beta used",
            '=IF(beta_src="raw",beta_raw,IF(beta_src="bottom_up",beta_bu,IF(beta_src="custom",beta_custom,'
            'IF(AND(beta_src="daily",ISNUMBER(beta_daily)),beta_daily,IF(AND(beta_src="daily_adj",ISNUMBER(beta_daily)),0.67*beta_daily+0.33,'
            'IF(AND(beta_src="crsp_monthly",ISNUMBER(beta_crsp)),beta_crsp,IF(AND(beta_src="crsp_monthly_adj",ISNUMBER(beta_crsp)),0.67*beta_crsp+0.33,'
            "0.67*beta_raw+0.33)))))))",
            "source chosen on Inputs; _adj = Blume 0.67 × β + 0.33",
        ),
        ("eq_prem", "5 · Equity risk premium for this stock = β × MRP", "=beta_used*mrp", "beta × (Rm − rf)"),
        (
            "ke",
            "6 · Cost of equity Ke = rf + β × MRP + CRP + size",
            "=rf+eq_prem+crp+size_prem",
            "the discount rate for equity holders",
        ),
        ("Cost of debt and WACC", None, None, None),
        ("kd_after", "After-tax cost of debt = Kd × (1 − t)", "=kd*(1-t_marg)", ""),
        ("e_val", "Equity at market value, $m", "=px_now*(shares_basic+dilution)", "price × diluted shares"),
        (
            "d_val",
            "Debt including leases, $m",
            "=core_debt+leases" if not fin else "=0",
            "book value" if not fin else "not used for banks, insurers or REITs",
        ),
        ("w_e", "Weight of equity E ÷ (D + E)", "=e_val/(e_val+d_val)", ""),
        ("w_d", "Weight of debt D ÷ (D + E)", "=d_val/(e_val+d_val)", ""),
        (
            "wacc",
            "WACC = wE × Ke + wD × Kd(1 − t)",
            "=w_e*ke+w_d*kd_after",
            "the discount rate for the whole firm" if not fin else "reference only",
        ),
    ]
    r = 5
    for nm, lab, f, how in rows:
        if lab is None:
            put(ws, r, 1, nm, F_B)
            r += 1
            continue
        key = nm in ("ke", "wacc", "beta_used")
        fmt = (
            NUM3
            if nm.startswith("beta")
            else "0"
            if nm == "beta_n"
            else MONEY
            if nm in ("e_val", "d_val")
            else PCT
        )
        put(ws, r, 1, lab, F_B if key else F_TXT)
        put(ws, r, 2, f, F_B if key else F_TXT, fmt, FILL_KEY if key else None)
        put(ws, r, 3, how, F_NOTE)
        B.name(nm, "dr", f"B{r}")
        r += 1
    if fin:
        # banks / insurers / REITs: the cash flows are to equity, so they are discounted at Ke
        B.name("disc_rate", "dr", f"B{r}")
        put(ws, r, 1, "Discount rate used in step 4 (cost of equity)", F_B)
        put(ws, r, 2, "=ke", F_B, PCT, FILL_KEY)
    else:
        B.name("disc_rate", "dr", f"B{r}")
        put(ws, r, 1, "Discount rate used in step 4 (WACC)", F_B)
        put(ws, r, 2, "=wacc", F_B, PCT, FILL_KEY)
    ws.column_dimensions["A"].width = 62
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 60


def beta_sheet(B, inp):
    ws = B.sheet("beta")
    bd = inp["capital"].get("beta_data") or {}
    title(
        ws,
        "Beta data",
        f"Daily adjusted closes for {inp['meta']['ticker']} and the S&P 500; returns are formulas. Paste a longer or different window to change the beta.",
    )
    header(ws, 4, ["Date", inp["meta"]["ticker"], "S&P 500", "Stock return", "Index return"])
    d, s, m = bd.get("dates", []), bd.get("stock", []), bd.get("index", [])
    for i, (a, b_, c) in enumerate(zip(d, s, m, strict=False)):
        r = 5 + i
        put(ws, r, 1, a)
        put(ws, r, 2, b_, F_IN, "0.0000")
        put(ws, r, 3, c, F_IN, "0.00")
        if i:
            put(ws, r, 4, f"=B{r}/B{r - 1}-1", F_TXT, "0.00%")
            put(ws, r, 5, f"=C{r}/C{r - 1}-1", F_TXT, "0.00%")
    last = 5 + max(len(d) - 1, 1)
    B.name_range("beta_rs", "beta", "D6", f"D{last}")
    B.name_range("beta_rm", "beta", "E6", f"E{last}")
    put(ws, 2, 7, (inp["capital"].get("beta_daily_stats") or {}).get("window", ""), F_NOTE)
    for col, w in zip("ABCDE", (12, 12, 12, 13, 13), strict=True):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A5"


# ---------------------------------------------------------------- 4 Discount
def discount_sheet(B, inp, cf_row, ir_cf):
    ws = B.sheet("disc")
    mdl = v.model_of(inp)
    title(
        ws,
        "Step 4 · Discount the cash flows",
        "PV = cash flow × (1 + r)^−t, with r from step 3; then the terminal value.",
    )
    ir = year_header(ws, inp, 4, "Line")
    r = ir + 1
    rows = {}
    src = q(SH["cf"])
    what = {"fcff": "FCFF counted", "excess_return": "Excess return counted", "ddm": "Dividend counted"}[mdl]

    def line(key, label, how, f_of_i, fmt=MONEY, bold=False, total=False):
        nonlocal r
        put(ws, r, 1, label, F_B if bold else F_TXT)
        put(ws, r, 3, how, F_NOTE)
        for i in range(N):
            put(ws, r, YC + i, f_of_i(i, COLS[i]), F_LINK if key == "cf" else (F_B if bold else F_TXT), fmt)
        if total:
            put(ws, r, 2, f"=SUM({COLS[0]}{r}:{COLS[-1]}{r})", F_B if bold else F_TXT, fmt)
        rows[key] = r
        r += 1

    fmt = MONEY2 if mdl == "ddm" else MONEY
    if mdl == "fcff":
        line("cf", "FCFF (full year)", "from step 2", lambda i, c: f"={src}!{c}{cf_row}", fmt)
        line(
            "frac",
            "Share of the year counted",
            "year 1: only the part after the last balance sheet (the rest is in today's cash)",
            lambda i, c: "=stub" if i == 0 else "=1",
            PCT1,
        )
        line("cnt", what, "FCFF × share of year", lambda i, c: f"={c}{rows['cf']}*{c}{rows['frac']}", fmt)
    elif mdl == "excess_return":
        line(
            "frac",
            "Share of the year counted",
            "year 1: the part after the last balance sheet",
            lambda i, c: "=stub" if i == 0 else "=1",
            PCT1,
        )
        line("cnt", what, "from step 2 (already × share of year)", lambda i, c: f"={src}!{c}{cf_row}", fmt)
    else:
        line(
            "frac",
            "Share still to be paid",
            "year 1: the part of the fiscal year still ahead",
            lambda i, c: "=MIN(1,MAX(0,first_period))" if i == 0 else "=1",
            PCT1,
        )
        line("cnt", what, "from step 2", lambda i, c: f"={src}!{c}{cf_row}", fmt)
    line(
        "t",
        "Years from today (t)",
        "first fiscal year end + i, less half a year's share if mid-year",
        lambda i, c: (
            f"=MAX(IF(midyear=1,first_period+{c}${ir}-0.5*{c}{rows['frac']},first_period+{c}${ir}),0)"
        ),
        "0.000",
    )
    line("df", "Discount factor", "(1 + r)^−t", lambda i, c: f"=(1+disc_rate)^(-{c}{rows['t']})", "0.0000")
    line(
        "pv",
        "Present value",
        "counted × discount factor",
        lambda i, c: f"={c}{rows['cnt']}*{c}{rows['df']}",
        fmt,
        bold=True,
        total=True,
    )
    r += 1
    B.name("pv_sum", "disc", f"B{rows['pv']}")
    if mdl == "fcff":
        out = [
            ("t_n", "Years to the last forecast year end", f"=first_period+{N - 1}", "0.00", ""),
            (
                "tv_gordon",
                "Terminal value, value driver = NOPAT × (1 + g) × (1 − g ÷ RONIC) ÷ (WACC − g)",
                "=cf_last_nopat*(1+g)*(1-g/ronic)/(disc_rate-g)",
                MONEY,
                "growth needs reinvestment: g ÷ RONIC of NOPAT is reinvested",
            ),
            (
                "pv_gordon",
                "PV of value-driver terminal value",
                "=tv_gordon*(1+disc_rate)^(-IF(midyear=1,t_n-0.5,t_n))",
                MONEY,
                "",
            ),
            (
                "tv_exit",
                "Terminal value, exit multiple × last EBITDA",
                "=exit_mult*cf_last_ebitda",
                MONEY,
                "cross-check",
            ),
            ("pv_exit", "PV of exit terminal value", "=tv_exit*(1+disc_rate)^(-t_n)", MONEY, ""),
            (
                "pv_tv",
                "PV of the terminal value used",
                '=IF(tv_method="exit",pv_exit,pv_gordon)',
                MONEY,
                "method on Inputs",
            ),
            ("ev", "Enterprise value = Σ PV + PV of terminal value", "=pv_sum+pv_tv", MONEY, ""),
            ("tv_share", "Terminal value share of EV", "=pv_tv/ev", PCT1, ""),
        ]
    elif mdl == "excess_return":
        out = [
            ("t_n", "Years to the last forecast year end", f"=first_period+{N - 1}", "0.00", ""),
            (
                "roe_t",
                "Terminal ROE = Ke + persistence × (long-run ROE − Ke)",
                "=ke+persist*(s_roe_1-ke)",
                PCT,
                "competition erodes the spread",
            ),
            (
                "tv",
                "Terminal value = (terminal ROE − Ke) × closing book ÷ (Ke − g)",
                "=(roe_t-ke)*cf_last_bv/(ke-g)",
                MONEY,
                "",
            ),
            ("pv_tv", "PV of terminal value", "=tv*(1+disc_rate)^(-IF(midyear=1,t_n-0.5,t_n))", MONEY, ""),
        ]
    else:
        out = [
            ("t_n", "Years to the last forecast year end", f"=first_period+{N - 1}", "0.00", ""),
            (
                "tv",
                "Terminal value = last dividend × (1 + g) ÷ (Ke − g)",
                "=cf_last_dps*(1+g)/(ke-g)",
                MONEY2,
                "Gordon growth",
            ),
            ("pv_tv", "PV of terminal value", "=tv*(1+disc_rate)^(-IF(midyear=1,t_n-0.5,t_n))", MONEY2, ""),
        ]
    for nm, lab, f, fm, how in out:
        put(ws, r, 1, lab, F_B if nm in ("ev", "pv_tv") else F_TXT)
        put(ws, r, 2, f, F_B if nm in ("ev",) else F_TXT, fm, FILL_KEY if nm == "ev" else None)
        put(ws, r, 3, how, F_NOTE)
        B.name(nm, "disc", f"B{r}")
        r += 1
    ws.column_dimensions["A"].width = 70
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 44
    for c in COLS:
        ws.column_dimensions[c].width = 12


# ------------------------------------------------------------------- 5 Price
def price_sheet(B, inp):
    ws = B.sheet("px")
    mdl = v.model_of(inp)
    title(
        ws,
        "Step 5 · From value to the share price",
        {
            "fcff": "Enterprise value − debt + cash (and other claims) = equity value, the estimated market cap; ÷ diluted shares = fair price.",
            "excess_return": "Book equity + PV of excess returns = equity value; ÷ diluted shares = fair price. Debt is part of the business, so there is no debt bridge.",
            "ddm": "The dividend model is already per share; × diluted shares gives the equity value.",
        }[mdl],
    )
    header(ws, 4, ["Item", "Value", "How it is calculated"])
    if mdl == "fcff":
        rows = [
            ("ev_x", "Enterprise value (step 4)", "=ev", MONEY, ""),
            ("x_debt", "− Debt excluding leases", "=-core_debt", MONEY, "Inputs"),
            ("x_leases", "− Lease liabilities", "=-leases", MONEY, "Inputs"),
            ("x_cash", "+ Cash", "=cash", MONEY, "Inputs"),
            (
                "dfs_net",
                "+ Captive-finance net assets (receivables + equipment − its own debt)",
                "=dfs_rec+dfs_le-dfs_debt",
                MONEY,
                "financing arms are added at net book value",
            ),
            ("x_inv", "+ Non-operating investments", "=investments", MONEY, "Inputs"),
            ("x_nci", "− Non-controlling interests", "=-nci", MONEY, "Inputs"),
            (
                "equity",
                "Equity value (estimated market cap)",
                "=ev_x+x_debt+x_leases+x_cash+dfs_net+x_inv+x_nci",
                MONEY,
                "",
            ),
        ]
    elif mdl == "excess_return":
        rows = [
            ("x_bv", "Book equity today", "=fin_bv0", MONEY, "Inputs"),
            ("x_pv", "+ PV of excess returns (step 4)", "=pv_sum", MONEY, ""),
            ("x_pvtv", "+ PV of terminal excess return (step 4)", "=pv_tv", MONEY, ""),
            ("equity", "Equity value (estimated market cap)", "=x_bv+x_pv+x_pvtv", MONEY, ""),
        ]
    else:
        rows = [
            ("x_pv", "PV of dividends (step 4), per share", "=pv_sum", MONEY2, ""),
            ("x_pvtv", "+ PV of terminal value, per share", "=pv_tv", MONEY2, ""),
            ("model_ps_ddm", "Value per share", "=x_pv+x_pvtv", MONEY2, ""),
            (
                "equity",
                "Equity value = value per share × diluted shares",
                "=model_ps_ddm*(shares_basic+dilution)",
                MONEY,
                "",
            ),
        ]
    rows += [
        (
            "dil_sh",
            "Diluted shares (m) = shares outstanding + dilution",
            "=shares_basic+dilution",
            NUM,
            "Inputs",
        ),
        (
            "model_ps",
            "Fair share price (this scenario)",
            "=equity/dil_sh",
            MONEY2,
            "equity value ÷ diluted shares",
        ),
        ("x_price", "Share price today", "=px_now", MONEY2, ""),
        ("model_up", "Upside / (downside)", "=model_ps/x_price-1", PCT1, ""),
        ("mcap_now", "Market cap today", "=px_now*dil_sh", MONEY, ""),
        ("model_mcap_gap", "Estimated vs actual market cap", "=equity/mcap_now-1", PCT1, ""),
    ]
    if mdl == "excess_return":
        rows += [
            ("jpb", "Justified price-to-book = equity ÷ book", "=equity/fin_bv0", MULT, ""),
            ("mpb", "Market price-to-book", "=mcap_now/fin_bv0", MULT, ""),
        ]
    r = 5
    for nm, lab, f, fm, how in rows:
        key = nm in ("equity", "model_ps")
        put(ws, r, 1, lab, F_B if key else F_TXT)
        put(ws, r, 2, f, F_B if key else F_TXT, fm, FILL_KEY if key else None)
        put(ws, r, 3, how, F_NOTE)
        if nm == "equity":
            ws.cell(r, 1).border = ws.cell(r, 2).border = TOP
        B.name(nm, "px", f"B{r}")
        r += 1
    ws.column_dimensions["A"].width = 66
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 48


# -------------------------------------------------------------------- Comps
def comps_sheet(B, inp, R):
    ws = B.sheet("comps")
    title(
        ws,
        "Comps",
        "Peer multiples from the data feed (blue, editable). Non-positive multiples are left blank, so they drop out of the statistic.",
    )
    w = inp["relative"]["weights"]
    cols, ix = inp["peers"]["columns"], {c: i for i, c in enumerate(inp["peers"]["columns"])}
    keys = [k for k in w if w[k] > 0 and k in cols]
    names = {
        "pe_fwd": "Forward P/E",
        "ev_ebitda": "EV/EBITDA",
        "ev_sales": "EV/Sales",
        "pb": "Price/book",
        "div_yield": "Dividend yield",
    }
    header(ws, 4, ["Peer", "Name"] + [names[k] for k in keys])
    core = inp["peers"]["core_set"]
    r = 5
    for tk in core:
        row = inp["peers"]["rows"][tk]
        put(ws, r, 1, tk)
        put(ws, r, 2, row[0])
        for j, k in enumerate(keys):
            x = row[ix[k]]
            put(
                ws,
                r,
                3 + j,
                x if x is not None and np.isfinite(x) and x > 0 else None,
                F_IN,
                PCT if k == "div_yield" else MULT,
            )
        r += 1
    n0, n1 = 5, max(5, r - 1)
    fn = {"median": "MEDIAN", "harmonic": "HARMEAN", "mean": "AVERAGE"}[inp["relative"]["stat"]]
    r += 1
    m = inp["market"]
    td = m.get("total_debt_incl_dfs")
    metric_rows = [
        ("c_eps", "Forward EPS (consensus)", m.get("eps_ntm"), MONEY2),
        ("c_ebitda", "EBITDA, trailing 12 months ($m)", m.get("ebitda_ttm"), MONEY),
        ("c_rev", "Revenue, trailing 12 months ($m)", m.get("revenue_ttm"), MONEY),
        (
            "c_nd",
            "Net debt on the vendor's basis: all debt + leases − cash ($m)",
            None
            if td is None
            else td
            + (inp.get("bridge") or {}).get("leases", 0.0)
            - ((inp.get("bridge") or {}).get("cash") or 0.0),
            MONEY,
        ),
    ]
    put(ws, r, 1, "Company metrics", F_B)
    r += 1
    for nm, lab, val, fm in metric_rows:
        put(ws, r, 1, lab)
        put(ws, r, 3, num(val), F_IN, fm)
        B.name(nm, "comps", f"C{r}")
        r += 1
    put(ws, r, 1, "Book per share = book ÷ diluted shares" if v.is_fin(inp) else "", F_TXT)
    if v.is_fin(inp):
        put(ws, r, 3, "=fin_bv0/(shares_basic+dilution)", F_TXT, MONEY2)
        B.name("c_bvps", "comps", f"C{r}")
    r += 2
    header(
        ws,
        r,
        [
            "Multiple",
            "",
            "Peer " + inp["relative"]["stat"],
            "Company today",
            "Implied price",
            "Weight",
            "Weight × price",
        ],
    )
    r += 1
    first = r
    impl = {
        "pe_fwd": ("{k}*c_eps", "px_now/c_eps"),
        "pb": ("{k}*c_bvps", "px_now/c_bvps"),
        "div_yield": ("fin_dps0/{k}", "fin_dps0/px_now"),
        "ev_ebitda": (
            "({k}*c_ebitda-c_nd)/(shares_basic+dilution)",
            "(px_now*(shares_basic+dilution)+c_nd)/c_ebitda",
        ),
        "ev_sales": (
            "({k}*c_rev-c_nd)/(shares_basic+dilution)",
            "(px_now*(shares_basic+dilution)+c_nd)/c_rev",
        ),
    }
    for j, k in enumerate(keys):
        col = L(3 + j)
        put(ws, r, 1, names[k])
        put(ws, r, 3, f"=IFERROR({fn}({col}{n0}:{col}{n1}),NA())", F_TXT, PCT if k == "div_yield" else MULT)
        put(ws, r, 4, f"=IFERROR({impl[k][1]},NA())", F_TXT, PCT if k == "div_yield" else MULT)
        put(ws, r, 5, f"=IFERROR({impl[k][0].format(k=f'C{r}')},NA())", F_TXT, MONEY2)
        put(ws, r, 6, w[k], F_IN, "0%")
        put(ws, r, 7, f"=IF(ISNUMBER(E{r}),E{r}*F{r},0)", F_TXT, MONEY2)
        r += 1
    put(ws, r, 1, "Blended comps value = Σ weight × price ÷ Σ weight of the multiples that worked", F_B)
    put(
        ws,
        r,
        5,
        f"=IF(SUMPRODUCT(--ISNUMBER(E{first}:E{r - 1}),F{first}:F{r - 1})>0,SUM(G{first}:G{r - 1})/SUMPRODUCT(--ISNUMBER(E{first}:E{r - 1}),F{first}:F{r - 1}),NA())",
        F_B,
        MONEY2,
        FILL_KEY,
    )
    B.name("comps_ps", "comps", f"E{r}")
    r += 3
    reg_block(B, ws, inp, r, R)
    ws.column_dimensions["A"].width = 60
    ws.column_dimensions["B"].width = 30
    for c in "CDEFG":
        ws.column_dimensions[c].width = 15


def reg_block(B, ws, inp, r, R):
    """Peer regression as formulas: forward P/E on EPS growth, or for banks P/B on ROE."""
    mdl = v.model_of(inp)
    if mdl == "ddm":
        put(ws, r, 1, "No peer regression for the dividend model.", F_NOTE)
        put(ws, r + 1, 1, "Regression value")
        put(ws, r + 1, 5, '=""')
        B.name("reg_ps", "comps", f"E{r + 1}")
        return
    ix = {c: i for i, c in enumerate(inp["peers"]["columns"])}
    xk, yk = ("roe", "pb") if mdl == "excess_return" else ("eps_g3y", "pe_fwd")
    put(
        ws,
        r,
        1,
        "Peer regression: "
        + ("price-to-book on ROE" if mdl == "excess_return" else "forward P/E on 3-year EPS growth"),
        F_H,
    )
    r += 1
    header(
        ws,
        r,
        [
            "Peer",
            "",
            "x: " + ("ROE" if mdl == "excess_return" else "EPS growth"),
            "y: " + ("P/B" if mdl == "excess_return" else "Fwd P/E"),
        ],
    )
    r += 1
    first = r
    for tk, row in inp["peers"]["rows"].items():
        x, y = row[ix[xk]] if xk in ix else None, row[ix[yk]] if yk in ix else None
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)) or x <= 0 or y <= 0:
            continue
        put(ws, r, 1, tk)
        put(ws, r, 3, x, F_IN, PCT1)
        put(ws, r, 4, y, F_IN, '0.00"x"')
        r += 1
    last = max(first, r - 1)
    xs, ys = f"C{first}:C{last}", f"D{first}:D{last}"
    own_x = "roe_ttm" if mdl == "excess_return" else "c_growth"
    sg = (inp.get("relative") or {}).get("subject_growth")
    sg_note = "consensus EPS growth (Inputs file)"
    if mdl != "excess_return" and (sg is None or not np.isfinite(sg)):
        sg = R["reg"]["dellGrowth"]
        sg_note = "model EPS CAGR year 1 → 4 for this scenario, from the engine (value)"
    out = [
        ("reg_b", "Slope b = SLOPE(y, x)", f"=SLOPE({ys},{xs})", "0.00"),
        ("reg_a", "Intercept a = INTERCEPT(y, x)", f"=INTERCEPT({ys},{xs})", "0.00"),
        ("reg_r2", "R²", f"=RSQ({ys},{xs})", "0.00"),
        ("reg_n", "Peers in the fit", f"=COUNT({xs})", "0"),
    ]
    if mdl != "excess_return":
        out.append(("c_growth", f"Company EPS growth used: {sg_note}", sg, PCT1))
    out += [
        (
            "reg_mult",
            "Fitted multiple = a + b × company " + ("ROE" if mdl == "excess_return" else "growth"),
            f"=reg_a+reg_b*{own_x}",
            "0.00",
        ),
        (
            "reg_ps",
            "Regression value per share (blank if R² is below the minimum on Inputs)",
            "=IF(AND(reg_n>=3,reg_r2>=reg_min),reg_mult*"
            + ("c_bvps" if mdl == "excess_return" else "c_eps")
            + ',"")',
            MONEY2,
        ),
    ]
    r += 1
    for nm, lab, f, fm in out:
        put(ws, r, 1, lab, F_B if nm == "reg_ps" else F_TXT)
        put(
            ws,
            r,
            5,
            f,
            F_IN if nm == "c_growth" else (F_B if nm == "reg_ps" else F_TXT),
            fm,
            FILL_KEY if nm == "reg_ps" else None,
        )
        B.name(nm, "comps", f"E{r}")
        r += 1


# ----------------------------------------------------------------- 6 Verdict
def verdict_sheet(B, inp, R):
    ws = B.sheet("ver")
    vw = inp["verdict"]["weights"]
    mdl = v.model_of(inp)
    title(
        ws,
        "Step 6 · Verdict",
        "Methods blended by weight; methods that give no value drop out. The call needs the blend to differ from the price by more than the band.",
    )
    header(
        ws,
        4,
        [
            "Method",
            "Value / share",
            "Weight",
            "vs price",
            "Weight × value",
            "Weight counted",
            "How it is computed",
        ],
    )
    absname = {
        "fcff": "DCF (FCFF), this scenario",
        "excess_return": "Excess-return model, this scenario",
        "ddm": "Dividend discount model, this scenario",
    }[mdl]
    rows = [
        (absname, "=model_ps", vw.get("dcf", 0), "steps 2–5"),
        (
            "Monte Carlo median",
            num(R["mc"]["median"]),
            vw.get("mc", 0),
            f"value from the Python engine: {R['mc']['n']:,} model runs with random drivers, discount rate and g (not a formula)",
        ),
        ("Comps blend", "=comps_ps", vw.get("comps", 0), "Comps sheet"),
        ("Peer regression", "=reg_ps", vw.get("regression", 0), "Comps sheet"),
    ]
    r = 5
    for lab, val, w, how in rows:
        put(ws, r, 1, lab)
        put(ws, r, 2, val, F_IN if not isinstance(val, str) else F_LINK, MONEY2)
        put(ws, r, 3, w, F_IN, "0%")
        put(ws, r, 4, f'=IF(ISNUMBER(B{r}),B{r}/px_now-1,"")', F_TXT, PCT1)
        put(ws, r, 5, f"=IF(ISNUMBER(B{r}),B{r}*C{r},0)", F_TXT, MONEY2)
        put(ws, r, 6, f"=IF(ISNUMBER(B{r}),C{r},0)", F_TXT, "0%")
        put(ws, r, 7, how, F_NOTE)
        r += 1
    a, b_ = 5, r - 1
    out = [
        (
            "fair",
            "Blended fair value = Σ weight × value ÷ Σ weight counted",
            f"=SUM(E{a}:E{b_})/SUM(F{a}:F{b_})",
            MONEY2,
        ),
        ("upside", "Upside / (downside) vs price", "=fair/px_now-1", PCT1),
        ("eq_grade_x", "Earnings-quality grade (step 1)", "=eq_grade", None),
        (
            "raw_call",
            "Call before the quality gate",
            '=IF(upside>band,"UNDERVALUED",IF(upside<-band,"OVERVALUED","FAIRLY VALUED"))',
            None,
        ),
        ("v_call", "Call", '=IF(AND(eq_gate=1,eq_grade_x="D"),"NO CALL",raw_call)', None),
        ("v_conf", "Confidence", '=IF(AND(eq_gate=1,eq_grade_x="C"),"low","normal")', None),
        (
            "exp_ret",
            "Expected 1-year return = (fair ÷ price)^(1 ÷ horizon) − 1 + dividend yield",
            "=(fair/px_now)^(1/horizon)-1+dps/px_now",
            PCT1,
        ),
        ("req_ret", "Required return (Ke, step 3)", "=ke", PCT1),
        ("alpha", "Expected alpha = expected − required", "=exp_ret-req_ret", PCT1),
    ]
    r += 1
    for nm, lab, f, fm in out:
        key = nm in ("fair", "v_call", "alpha")
        put(ws, r, 1, lab, F_B if key else F_TXT)
        put(ws, r, 2, f, F_B if key else F_TXT, fm, FILL_KEY if key else None)
        B.name(nm, "ver", f"B{r}")
        r += 1
    r += 1
    put(
        ws,
        r,
        1,
        f"Python engine at build ({inp['meta']['valuation_date']}): fair {R['verdict']['fair']:.4f}, {R['verdict']['call']}. "
        "Tests recalculate this workbook and require the same numbers.",
        F_NOTE,
    )
    ws.column_dimensions["A"].width = 72
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["G"].width = 80


# ----------------------------------------------------------------- Dashboard
def dashboard_sheet(B, inp, scen_name="Base"):
    from openpyxl.worksheet.datavalidation import DataValidation

    ws = B.sheet("read")
    m, mdl = inp["meta"], v.model_of(inp)
    model_lbl = {
        "fcff": "FCFF DCF",
        "excess_return": "Excess-return model (bank / insurer)",
        "ddm": "Dividend discount model (REIT)",
    }[mdl]
    put(ws, 1, 2, f"{m['name']}  ·  {m['ticker']}", Font(name=FONT, bold=True, size=18, color="FFFFFF"))
    put(
        ws,
        2,
        2,
        f"Valuation framework  ·  {model_lbl}  ·  valuation date {m['valuation_date']}  ·  {m.get('currency', 'USD')} millions except per share",
        Font(name=FONT, size=10, color="C9D3E0"),
    )
    # scenario switch
    put(ws, 4, 2, "SCENARIO", Font(name=FONT, bold=True, size=9, color=MUTED))
    put(ws, 5, 2, scen_name, Font(name=FONT, bold=True, size=14, color=NAVY), None, FILL_Y)
    ws.cell(5, 2).alignment = Alignment(horizontal="center", vertical="center")
    dv = DataValidation(type="list", formula1='"Bear,Base,Bull"', allow_blank=False, showDropDown=False)
    dv.error, dv.errorTitle, dv.prompt, dv.promptTitle = (
        "Pick Bear, Base or Bull",
        "Scenario",
        "Pick Bear, Base or Bull",
        "Scenario",
    )
    ws.add_data_validation(dv)
    dv.add("B5")
    B.name("scen_name", "read", "B5")
    put(
        ws,
        5,
        3,
        "◂ pick Bear, Base or Bull; every sheet re-values",
        Font(name=FONT, italic=True, size=9, color=MUTED),
    )
    put(
        ws, 4, 16, '=IF(scen_name="Bear",1,IF(scen_name="Bull",3,2))', Font(name=FONT, size=8, color="FFFFFF")
    )
    B.name("scen_sel", "read", "P4")
    # KPI tiles: label row 7, value row 8, note row 9; each tile is two columns wide
    tiles = [
        ("FAIR VALUE / SHARE", "=fair", MONEY2, "blended, step 6"),
        ("SHARE PRICE", "=px_now", MONEY2, m.get("price_date", "")),
        ("UPSIDE", "=upside", "+0.0%;-0.0%;0.0%", "fair ÷ price − 1"),
        ("CALL", "=v_call", None, f"±{inp['verdict']['band']:.0%} band"),
        ("EXPECTED ALPHA", "=alpha", "+0.0%;-0.0%;0.0%", "1-year, vs Ke"),
        ("DISCOUNT RATE", "=disc_rate", PCT, "WACC" if mdl == "fcff" else "cost of equity"),
        ("EARNINGS QUALITY", "=eq_grade", None, "grade A–D, step 1"),
    ]
    from openpyxl.formatting.rule import FormulaRule

    for i, (lab, f, fm, note) in enumerate(tiles):
        c0 = 2 + 2 * i
        for r in (7, 8, 9):
            for c in (c0, c0 + 1):
                ws.cell(r, c).fill = FILL_PANEL
        ws.merge_cells(start_row=7, start_column=c0, end_row=7, end_column=c0 + 1)
        ws.merge_cells(start_row=8, start_column=c0, end_row=8, end_column=c0 + 1)
        ws.merge_cells(start_row=9, start_column=c0, end_row=9, end_column=c0 + 1)
        put(ws, 7, c0, lab, Font(name=FONT, bold=True, size=8, color=MUTED), fill=FILL_PANEL)
        put(ws, 8, c0, f, Font(name=FONT, bold=True, size=18, color=NAVY), fm, FILL_PANEL)
        put(ws, 9, c0, note, Font(name=FONT, size=8, color=MUTED), fill=FILL_PANEL)
        for r in (7, 8, 9):
            ws.cell(r, c0).alignment = Alignment(horizontal="center", vertical="center")
        ref = f"{L(c0)}8"
        if lab == "CALL":
            for txt, col in (
                ("UNDERVALUED", GREEN),
                ("OVERVALUED", RED),
                ("FAIRLY VALUED", AMBER),
                ("NO CALL", MUTED),
            ):
                ws.conditional_formatting.add(
                    ref,
                    FormulaRule(
                        formula=[f'{ref}="{txt}"'],
                        font=Font(name=FONT, bold=True, size=15, color="FFFFFF"),
                        fill=PatternFill("solid", fgColor=col),
                    ),
                )
        if lab in ("UPSIDE", "EXPECTED ALPHA"):
            ws.conditional_formatting.add(
                ref, FormulaRule(formula=[f"{ref}>0"], font=Font(name=FONT, bold=True, size=18, color=GREEN))
            )
            ws.conditional_formatting.add(
                ref, FormulaRule(formula=[f"{ref}<0"], font=Font(name=FONT, bold=True, size=18, color=RED))
            )
        if lab == "EARNINGS QUALITY":
            for g, col in (("A", GREEN), ("B", "5B9A5B"), ("C", AMBER), ("D", RED)):
                ws.conditional_formatting.add(
                    ref,
                    FormulaRule(
                        formula=[f'{ref}="{g}"'], font=Font(name=FONT, bold=True, size=18, color=col)
                    ),
                )
    for r, h in ((1, 30), (2, 18), (5, 24), (7, 16), (8, 34), (9, 15)):
        ws.row_dimensions[r].height = h
    # chart data (columns R:U, small grey text; the charts read these live cells)
    dc = 20
    put(ws, 3, dc, "chart data (live)", Font(name=FONT, size=8, color=MUTED))
    labels = {"fcff": "DCF", "excess_return": "Excess return", "ddm": "DDM"}[mdl]
    rows = [
        (labels, "=model_ps"),
        ("Monte Carlo", f"={q(SH['ver'])}!B6"),
        ("Comps", "=comps_ps"),
        ("Regression", "=IF(ISNUMBER(reg_ps),reg_ps,0)"),
        ("Blended", "=fair"),
    ]
    put(ws, 4, dc, "Method", Font(name=FONT, size=8, color=MUTED))
    put(ws, 4, dc + 1, "Value", Font(name=FONT, size=8, color=MUTED))
    put(ws, 4, dc + 2, "Price", Font(name=FONT, size=8, color=MUTED))
    for j, (lab, f) in enumerate(rows):
        put(ws, 5 + j, dc, lab, Font(name=FONT, size=8, color=MUTED))
        put(ws, 5 + j, dc + 1, f, Font(name=FONT, size=8, color=MUTED), MONEY2)
        put(ws, 5 + j, dc + 2, "=px_now", Font(name=FONT, size=8, color=MUTED), MONEY2)
    put(ws, 11, dc, "Present value of", Font(name=FONT, size=8, color=MUTED))
    put(ws, 12, dc, "Forecast years", Font(name=FONT, size=8, color=MUTED))
    put(ws, 12, dc + 1, "=pv_sum", Font(name=FONT, size=8, color=MUTED), MONEY)
    put(ws, 13, dc, "Terminal value", Font(name=FONT, size=8, color=MUTED))
    put(ws, 13, dc + 1, "=pv_tv", Font(name=FONT, size=8, color=MUTED), MONEY)
    for c in range(dc, dc + 3):
        ws.column_dimensions[L(c)].width = 12
    # how to use, with links to each step
    r = 31
    put(ws, r, 2, "HOW TO USE", Font(name=FONT, bold=True, size=9, color=MUTED))
    steps = [
        ("1 History", "historic performance and earnings quality"),
        ("2 Cash flows", "the forecast that produces the cash flows"),
        ("3 Discount rate", "rf, Rm, MRP, beta from daily returns, Ke, WACC"),
        ("4 Discount", "present values and the terminal value"),
        ("5 Price", "enterprise value → equity value → ÷ shares → price"),
        ("6 Verdict", "methods blended into the call and expected alpha"),
        ("Inputs", "every input: pulled value, your value (yellow), source"),
        ("Scenarios", "Bear / Base / Bull drivers"),
    ]
    for j, (sh, what) in enumerate(steps):
        c = ws.cell(r + 1 + j, 2, sh)
        c.hyperlink = f"#'{sh}'!A1"
        c.font = Font(name=FONT, bold=True, color=ACCENT, underline="single")
        put(ws, r + 1 + j, 3, what, Font(name=FONT, color=INK))
    r += len(steps) + 2
    put(ws, r, 2, "COLOUR CODE", Font(name=FONT, bold=True, size=9, color=MUTED))
    put(ws, r + 1, 2, "123", F_IN)
    put(ws, r + 1, 3, "blue: typed input or pulled data", F_TXT)
    put(ws, r + 2, 2, "123", F_TXT)
    put(ws, r + 2, 3, "black: formula", F_TXT)
    put(ws, r + 3, 2, "", F_TXT, fill=FILL_Y)
    put(ws, r + 3, 3, "yellow: your value (overrides the pulled one)", F_TXT)
    put(ws, r + 4, 2, "", F_TXT, fill=FILL_KEY)
    put(ws, r + 4, 3, "pale blue: key result of a step", F_TXT)
    r += 6
    put(ws, r, 2, "SOURCES", Font(name=FONT, bold=True, size=9, color=MUTED))
    for k, s_ in (inp.get("sources") or {}).items():
        r += 1
        put(ws, r, 2, k, Font(name=FONT, size=9, color=INK))
        put(ws, r, 9, s_, Font(name=FONT, size=9, color=MUTED))
    ws.column_dimensions["A"].width = 3
    for c in range(2, 16):
        ws.column_dimensions[L(c)].width = 13
    ws.column_dimensions["B"].width = 16
    ws.sheet_view.showGridLines = False
    return dc


def dashboard_charts(B, inp, cf_row, dc):
    """Charts read live cells, so they move with the scenario switch and your overrides."""
    from openpyxl.chart import BarChart, DoughnutChart, LineChart, Reference

    ws, mdl = B.ws["read"], v.model_of(inp)
    c1 = BarChart()
    c1.type, c1.style = "bar", 10
    c1.title = "Value per share vs price"
    c1.add_data(Reference(ws, min_col=dc + 1, min_row=4, max_row=9), titles_from_data=True)
    c1.set_categories(Reference(ws, min_col=dc, min_row=5, max_row=9))
    c1.series[0].graphicalProperties.solidFill = ACCENT
    c1.series[0].graphicalProperties.line.noFill = True
    l1 = LineChart()
    l1.add_data(Reference(ws, min_col=dc + 2, min_row=4, max_row=9), titles_from_data=True)
    l1.series[0].graphicalProperties.line.solidFill = RED
    l1.series[0].graphicalProperties.line.dashStyle = "dash"
    l1.series[0].smooth = False
    c1.type = "col"
    c1 += l1
    c1.y_axis.numFmt = '"$"#,##0'
    c1.y_axis.majorGridlines = None
    c1.legend.position = "b"
    c1.height, c1.width = 9, 11
    ws.add_chart(c1, "B11")
    cf = B.ws["cf"]
    c2 = BarChart()
    c2.type, c2.style = "col", 10
    c2.title = {
        "fcff": "Free cash flow, $m",
        "excess_return": "Excess return, $m",
        "ddm": "Dividend, $ per share",
    }[mdl]
    c2.y_axis.scaling.min = 0
    c2.add_data(
        Reference(cf, min_col=YC, max_col=YC + N - 1, min_row=cf_row), from_rows=True, titles_from_data=False
    )
    c2.set_categories(Reference(cf, min_col=YC, max_col=YC + N - 1, min_row=4))
    c2.series[0].graphicalProperties.solidFill = NAVY
    c2.legend = None
    c2.y_axis.majorGridlines = None
    c2.height, c2.width = 9, 12
    ws.add_chart(c2, "G11")
    c3 = DoughnutChart()
    c3.title = "Forecast vs terminal"
    c3.add_data(Reference(ws, min_col=dc + 1, min_row=12, max_row=13), titles_from_data=False)
    c3.set_categories(Reference(ws, min_col=dc, min_row=12, max_row=13))
    from openpyxl.chart.series import DataPoint

    for i, col in enumerate((ACCENT, "A8C5E6")):
        pt = DataPoint(idx=i)
        pt.graphicalProperties.solidFill = col
        c3.series[0].dPt.append(pt)
    c3.holeSize = 55
    c3.height, c3.width = 9, 8.5
    ws.add_chart(c3, "L11")
    ws.print_area = f"A1:P{ws.max_row}"


TAB = {
    "read": NAVY,
    "in": "B7791F",
    "sc": "B7791F",
    "hist": "2E6DB4",
    "cf": "2E6DB4",
    "dr": "2E6DB4",
    "disc": "2E6DB4",
    "px": "2E6DB4",
    "ver": "1E7B4A",
    "comps": "6B7280",
    "beta": "6B7280",
}


def polish(B):
    """One look across every sheet: navy title band, navy table headers, hairline row rules, no gridlines."""
    thin = Side(style="thin", color=RULE)
    for key, ws in B.ws.items():
        ws.sheet_properties.tabColor = TAB.get(key, NAVY)
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.sheet_view.showGridLines = False
        maxc = max(ws.max_column, 12)
        if key == "read":
            for c in range(1, max(maxc, 16) + 1):
                for r in (1, 2):
                    ws.cell(r, c).fill = FILL_H
            continue
        for c in range(1, maxc + 1):
            ws.cell(1, c).fill = FILL_H
            ws.cell(2, c).fill = FILL_SUB
        t = ws.cell(1, 1)
        t.font = Font(name=FONT, bold=True, size=14, color="FFFFFF")
        sc = ws.cell(1, 3)
        if isinstance(sc.value, str) and sc.value.startswith('="Scenario'):
            sc.font = Font(name=FONT, bold=True, size=10, color="FFD166")
        ws.cell(2, 1).font = Font(name=FONT, italic=True, size=9, color=MUTED)
        ws.row_dimensions[1].height = 26
        ws.row_dimensions[2].height = 16
        for row in ws.iter_rows(min_row=3):
            filled = [c for c in row if c.value is not None]
            for c in row:
                if (
                    c.fill is not None
                    and c.fill.fgColor is not None
                    and c.fill.fgColor.rgb in ("00" + NAVY, "FF" + NAVY, NAVY)
                ):
                    c.font = Font(name=FONT, bold=True, color="FFFFFF", size=9)
                    c.alignment = Alignment(
                        horizontal="center" if c.column > 1 else "left", vertical="center", wrap_text=True
                    )
                elif (
                    c.fill is not None
                    and c.fill.fgColor is not None
                    and c.fill.fgColor.rgb in ("00E7F0FA", "FFE7F0FA")
                ):
                    c.border = Border(
                        top=Side(style="thin", color=ACCENT), bottom=Side(style="thin", color=ACCENT)
                    )
                    c.font = Font(name=FONT, bold=True, color=NAVY)
                elif (
                    filled and c.column <= maxc and (c.border.bottom is None or c.border.bottom.style is None)
                ):
                    c.border = Border(bottom=thin)
        for c in range(1, maxc + 1):
            letter = L(c)
            if ws.column_dimensions[letter].width in (None, 13.0) and c > 3:
                ws.column_dimensions[letter].width = 12


def build_workbook(inp, path, scen_name="Base", mc_n=5000):
    R = v.run_all(inp, scen_name, mc_n=mc_n)
    if not R["checks"]["ok"]:
        raise ValueError("NOT VALUED: " + "; ".join(b["msg"] for b in R["checks"]["blocking"]))
    B = Book()
    dc = dashboard_sheet(B, inp, scen_name)
    inputs_sheet(B, inp, scen_name)
    scenarios_sheet(B, inp)
    history_sheet(B, inp)
    _, cf_row, ir = cashflow_sheet(B, inp)
    discount_rate_sheet(B, inp)
    discount_sheet(B, inp, cf_row, ir)
    price_sheet(B, inp)
    verdict_sheet(B, inp, R)
    comps_sheet(B, inp, R)
    beta_sheet(B, inp)
    for ws in B.wb.worksheets:
        if ws.title == SH["read"]:
            continue
        for row in ws.iter_rows():
            for c in row:
                c.alignment = Alignment(vertical="top")
    polish(B)
    dashboard_charts(B, inp, cf_row, dc)
    B.wb.save(path)
    return R


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "dell_inputs.json"
    with open(src, encoding="utf-8") as f:
        inp = json.load(f)
    out = sys.argv[2] if len(sys.argv) > 2 else f"{inp['meta']['ticker']}_valuation.xlsx"
    scen = sys.argv[3] if len(sys.argv) > 3 else "Base"
    R = build_workbook(inp, out, scen)
    print(f"wrote {out}: engine fair {R['verdict']['fair']:.2f}, {R['verdict']['call']}")
