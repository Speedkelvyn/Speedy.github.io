"""research.py — turn Yahoo Finance's raw feeds into a valuation inputs file (Python).

Line-for-line port of web/research.js build(); tests/test_research_py.py checks
both give the same inputs file for the same feeds. Every input gets a
provenance tag inp["provenance"][path] = [kind, source] with kind
"sourced" (read from a feed), "derived" (computed from sourced numbers,
formula given) or "assumption" (a modelling choice you can change).

    from research import build
    inp = build(feeds)          # feeds from fetch_yahoo.fetch("AAPL") or a saved fixture
"""

from __future__ import annotations

import datetime as dt
import math
import re

RATINGS = [
    (8.5, "AAA", 0.0059),
    (6.5, "AA", 0.0070),
    (5.5, "A+", 0.0092),
    (4.25, "A", 0.0107),
    (3.0, "A-", 0.0121),
    (2.5, "BBB", 0.0147),
    (2.25, "BB+", 0.0193),
    (2.0, "BB", 0.0236),
    (1.75, "B+", 0.0285),
    (1.5, "B", 0.0347),
    (1.25, "B-", 0.0437),
    (0.8, "CCC", 0.0811),
    (0.65, "CC", 0.1000),
    (0.2, "C", 0.1400),
    (-1e9, "D", 0.1900),
]
TYPES = [
    "TotalRevenue",
    "OperatingIncome",
    "NetIncome",
    "DilutedEPS",
    "ReconciledDepreciation",
    "CapitalExpenditure",
    "OperatingCashFlow",
    "FreeCashFlow",
    "PretaxIncome",
    "TaxProvision",
    "InterestExpense",
    "CashAndCashEquivalents",
    "TotalDebt",
    "CurrentAssets",
    "CurrentLiabilities",
    "StockholdersEquity",
    "DilutedAverageShares",
    "AccountsReceivable",
    "Inventory",
    "AccountsPayable",
    "CapitalLeaseObligations",
    "OrdinarySharesNumber",
    "EBITDA",
    "MinorityInterest",
    "LongTermDebt",
    "CommonStockEquity",
    "TangibleBookValue",
    "NetIncomeCommonStockholders",
    "CashDividendsPaid",
    "RepurchaseOfCapitalStock",
    "TotalAssets",
    "CostOfRevenue",
    "SellingGeneralAndAdministration",
    "NetPPE",
    "NetIncomeContinuousOperations",
]


def synthetic(cov):
    for lo, rating, spread in RATINGS:
        if cov >= lo:
            return lo, rating, spread
    return RATINGS[-1]


def raw(o):
    if isinstance(o, dict):
        return o.get("raw")
    return o if isinstance(o, (int, float)) and not isinstance(o, bool) else None


def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _z(x):
    """JavaScript arithmetic treats null as 0; used only where the JS builder relies on that (note strings, Math.max)."""
    return 0 if x is None else x


def fx(x, d):
    """JavaScript Number.toFixed."""
    return f"{_z(x):.{d}f}"


def series(ts_result):
    """Yahoo fundamentals-timeseries result list → {Type: [[date, value], ...]} sorted by date."""
    out = {}
    for x in ts_result or []:
        t = ((x.get("meta") or {}).get("type") or [None])[0]
        if not t:
            continue
        vals = [
            [e["asOfDate"], e["reportedValue"]["raw"] if e.get("reportedValue") else None]
            for e in (x.get(t) or [])
            if e
        ]
        out[re.sub(r"^(annual|quarterly)", "", t)] = sorted(
            [e for e in vals if e[1] is not None], key=lambda e: e[0]
        )
    return out


def closes(chart_result):
    r = chart_result
    ind = r["indicators"]
    adj = (
        ind["adjclose"][0]["adjclose"]
        if ind.get("adjclose") and ind["adjclose"][0]
        else ind["quote"][0]["close"]
    )
    return [[_iso(ts), a] for ts, a in zip(r["timestamp"], adj, strict=False) if a is not None]


def _iso(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def _date(s):
    return dt.date.fromisoformat(s)


def _plus_year(s):
    d = _date(s)
    try:
        return d.replace(year=d.year + 1).isoformat()
    except ValueError:  # 29 Feb → 1 Mar, as JavaScript's setUTCFullYear does
        return dt.date(d.year + 1, 3, 1).isoformat()


def last(s):
    return s[-1][1] if s else None


def last_date(s):
    return s[-1][0] if s else None


def at(s, k):
    return s[-k][1] if s and len(s) >= k else None


def sum_last(s, k):
    return sum(x[1] for x in s[-k:]) if s and len(s) >= k else None


def fy(d):
    return "FY" + f"{_date(d).year % 100:02d}"


def find(s, d):
    for e in s or []:
        if e[0] == d:
            return e
    return None


def returns_from_prices(stock, index):
    m = dict(index)
    rows = sorted([r for r in stock if r[0] in m], key=lambda r: r[0])
    rs, rm, dates = [], [], []
    for i in range(1, len(rows)):
        (d0, s0), (d1, s1) = rows[i - 1], rows[i]
        rs.append(s1 / s0 - 1)
        rm.append(m[d1] / m[d0] - 1)
        dates.append(d1)
    return rs, rm, dates


def capm_beta(rs, rm, ppy=252):
    """Same summation order as model.js capmBeta, so the two agree to the last bit."""
    n = len(rs)
    mx, my = _jsum(rm) / n, _jsum(rs) / n
    sxy = sxx = syy = 0.0
    for i in range(n):
        sxy += (rm[i] - mx) * (rs[i] - my)
        sxx += (rm[i] - mx) ** 2
        syy += (rs[i] - my) ** 2
    b = sxy / sxx
    a = my - b * mx
    sse = 0.0
    for i in range(n):
        sse += (rs[i] - a - b * rm[i]) ** 2
    s2 = sse / (n - 2)
    se_b, se_a = math.sqrt(s2 / sxx), math.sqrt(s2 * (1 / n + mx * mx / sxx))
    return {
        "beta": b,
        "alpha": a,
        "seBeta": se_b,
        "seAlpha": se_a,
        "tBeta": b / se_b,
        "tAlpha": a / se_a,
        "r2": 1 - sse / syy,
        "n": n,
        "alphaAnnual": (1 + a) ** ppy - 1,
        "blume": 0.67 * b + 0.33,
        "ci95": [b - 1.96 * se_b, b + 1.96 * se_b],
        "volStock": math.sqrt(syy / (n - 1) * ppy),
        "volMkt": math.sqrt(sxx / (n - 1) * ppy),
    }


def _jsum(a):
    s = 0
    for x in a:
        s += x
    return s


def _clean(o):
    """JSON.stringify semantics: NaN/Infinity → null, undefined keys dropped."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if v is not _UNDEF}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


_UNDEF = object()


def _div(a, b):
    """JavaScript division: null counts as 0, x/0 → ±Infinity or NaN."""
    a, b = _z(a), _z(b)
    if b == 0:
        return math.nan if a == 0 else math.copysign(math.inf, a)
    return a / b


def build(f):
    """f = {ticker, summary, annual, quarterly, stock, index, tnx, peers, today, erp?, model?} — the same shape research.js takes."""
    S = f["summary"]
    pr, sd, ks, fd = (
        S.get("price") or {},
        S.get("summaryDetail") or {},
        S.get("defaultKeyStatistics") or {},
        S.get("financialData") or {},
    )
    A, Q = f.get("annual") or {}, f.get("quarterly") or {}
    P = {}

    def src(path, kind, s):
        P[path] = [kind, s]

    YQ, YT, YTQ = (
        "Yahoo Finance quoteSummary",
        "Yahoo Finance fundamentals (annual)",
        "Yahoo Finance fundamentals (quarterly)",
    )
    today = f.get("today") or dt.date.today().isoformat()
    M_ = 1e6
    DQ = {"blocking": [], "warnings": []}
    prof = S.get("assetProfile") or {}
    ind = prof.get("industry") or ""
    lev = (
        last(Q.get("TotalAssets")) / last(Q.get("CommonStockEquity"))
        if last(Q.get("TotalAssets")) and last(Q.get("CommonStockEquity"))
        else None
    )
    MODEL = f.get("model") or (
        "ddm"
        if re.match(r"^REIT", ind)
        else "excess_return"
        if re.match(r"^(Banks|Insurance - |Capital Markets|Mortgage Finance|Financial Conglomerates)", ind)
        or (ind == "Credit Services" and lev is not None and lev > 6)
        else "fcff"
    )
    FIN = MODEL != "fcff"
    fcff_only = re.compile(r"^(y1_segments|scenarios\.|operating\.|capital\.kd_pretax|bridge\.)")

    def warn(path, msg):
        if not (FIN and fcff_only.match(path)):
            DQ["warnings"].append({"path": path, "msg": msg})

    price = raw(pr.get("regularMarketPrice"))
    mcap = raw(pr.get("marketCap"))
    shares = raw(ks.get("impliedSharesOutstanding")) or raw(ks.get("sharesOutstanding"))
    if mcap and price and (not shares or abs(shares * price / mcap - 1) > 0.05):
        shares = mcap / price
    shares = shares / M_ if shares else None
    q_dil, q_ord = last(Q.get("DilutedAverageShares")), last(Q.get("OrdinarySharesNumber"))
    dilution = (q_dil - q_ord) / M_ if q_dil and q_ord and q_dil > q_ord else 0

    rev0 = last(A.get("TotalRevenue")) / M_ if num(last(A.get("TotalRevenue"))) is not None else None
    last_fye = last_date(A.get("TotalRevenue"))
    nfy = raw(ks.get("nextFiscalYearEnd"))
    next_fye = _iso(nfy) if nfy else None
    if not next_fye or next_fye <= today:
        next_fye = _plus_year(last_fye)
    while next_fye <= today:
        next_fye = _plus_year(next_fye)
    y0 = _date(next_fye).year
    labels = [f"FY{(y0 + i) % 100:02d}" for i in range(10)]
    bs_date = last_date(Q.get("TotalDebt")) or last_date(Q.get("CashAndCashEquivalents")) or last_fye
    days = lambda a, b: (_date(b) - _date(a)).days
    first_period = days(today, next_fye) / 365.25
    stub = min(1, max(0.02, days(bs_date, next_fye) / 365.25))

    tr = (S.get("earningsTrend") or {}).get("trend") or []
    T0 = next((t for t in tr if t.get("period") == "0y"), {})
    T1 = next((t for t in tr if t.get("period") == "+1y"), {})
    rev_y1 = raw((T0.get("revenueEstimate") or {}).get("avg"))
    rev_y2 = raw((T1.get("revenueEstimate") or {}).get("avg"))
    eps_y1 = raw((T0.get("earningsEstimate") or {}).get("avg"))
    eps_y2avg = raw((T1.get("earningsEstimate") or {}).get("avg"))
    g1eps = raw(T1.get("growth"))
    eps_y2g = eps_y1 * (1 + g1eps) if eps_y1 is not None and g1eps is not None else None
    eps_conflict = eps_y2avg is not None and eps_y2g is not None and abs(eps_y2avg / eps_y2g - 1) > 0.03
    TR = A.get("TotalRevenue")
    hist_g = at(TR, 1) / at(TR, 2) - 1 if TR and len(TR) >= 2 else None
    if not rev_y1:
        warn(
            "y1_segments.total",
            "No consensus revenue; next year uses last year's growth"
            if hist_g is not None
            else "No consensus revenue and under two years of history",
        )
    y1 = rev_y1 / M_ if rev_y1 else (rev0 * (1 + hist_g) if rev0 is not None and hist_g is not None else None)
    y2 = (
        rev_y2 / M_
        if rev_y2
        else (y1 * (1 + min(hist_g, 0.1)) if y1 is not None and hist_g is not None else None)
    )
    if rev_y1 and not rev_y2:
        warn(
            "scenarios.Base.growth.total.0",
            "No consensus for the second year; growth uses history capped at 10%",
        )
    g_next = _div(y2, y1) - 1

    ttm_rev, ttm_oi = sum_last(Q.get("TotalRevenue"), 4), sum_last(Q.get("OperatingIncome"), 4)
    if ttm_rev and ttm_oi is not None:
        m_ttm = ttm_oi / ttm_rev
    else:
        m_ttm = (
            last(A.get("OperatingIncome")) / last(A.get("TotalRevenue"))
            if last(A.get("OperatingIncome")) is not None and last(A.get("TotalRevenue"))
            else None
        )
    if not (ttm_rev and ttm_oi is not None):
        warn(
            "scenarios.Base.margins.total.0",
            "Fewer than 4 quarters of operating income; the last annual margin is used",
        )
    if m_ttm is not None and m_ttm < 0:
        warn(
            "scenarios.Base.margins.total.0",
            "Operating margin is negative; an FCFF DCF on today's margin is unreliable",
        )
    hist = [oi[1] / r for d, r in (TR or []) for oi in [find(A.get("OperatingIncome"), d)] if oi]
    m_avg = _jsum(hist) / len(hist) if hist else m_ttm
    m_min = min(hist) if hist else (m_ttm * 0.8 if m_ttm is not None else None)
    base_lr = (m_ttm + m_avg) / 2 if m_ttm is not None and m_avg is not None else None
    # a perpetual loss values a going concern below zero; when the worst year is a loss, Bear fades to the through-cycle average instead
    bear_lr = m_avg if m_min is not None and m_min <= 0 and m_avg is not None and m_avg > 0 else m_min
    if bear_lr is not m_min:
        warn(
            "scenarios.Bear.margins.total.1",
            f"Worst annual margin {fx(m_min * 100, 1)}% is a loss; Bear fades to the {len(hist)}-year average {fx(m_avg * 100, 1)}% instead",
        )

    pre, taxp = last(A.get("PretaxIncome")), last(A.get("TaxProvision"))
    eff_raw = taxp / pre if pre is not None and pre > 0 and taxp is not None else None
    eff = min(0.30, max(0.10, eff_raw)) if eff_raw is not None else 0.21
    if eff_raw is None:
        warn("operating.tax0", "No positive pre-tax income; tax rate set to the 21% statutory rate")
    elif eff_raw != eff:
        warn(
            "operating.tax0", f"Effective tax rate {fx(eff_raw * 100, 1)}% was clamped to {fx(eff * 100, 0)}%"
        )

    def avg3(s, absv=False):
        vals = [abs(x[1]) if absv else x[1] for x in (s or [])[-3:]]
        return _jsum(vals) / len(vals) if vals else None

    rev3 = avg3(TR)
    da_pct = (
        avg3(A.get("ReconciledDepreciation")) / rev3
        if avg3(A.get("ReconciledDepreciation")) is not None and rev3
        else None
    )
    capex_pct = (
        avg3(A.get("CapitalExpenditure"), True) / rev3
        if avg3(A.get("CapitalExpenditure"), True) is not None and rev3
        else None
    )
    # working capital: latest quarter's balances against that quarter's annualised revenue (a balance reflects the sales running at that date)
    qd = last_date(Q.get("TotalRevenue"))
    q_parts = (
        [
            (find(Q.get(k), qd) or [None, None])[1]
            for k in ("AccountsReceivable", "Inventory", "AccountsPayable")
        ]
        if qd
        else [None, None, None]
    )
    nwc_q = all(x is not None for x in q_parts)
    nwc_parts = (
        q_parts
        if nwc_q
        else [last(A.get("AccountsReceivable")), last(A.get("Inventory")), last(A.get("AccountsPayable"))]
    )
    if any(x is None for x in nwc_parts):
        warn(
            "operating.nwc_pct",
            "Receivables, inventory or payables missing in the feed; the missing item counts as zero",
        )
    nwc_level = (nwc_parts[0] or 0) + (nwc_parts[1] or 0) - (nwc_parts[2] or 0)
    nwc_rev = 4 * last(Q.get("TotalRevenue")) if nwc_q else rev0 * M_ if rev0 is not None else None
    nwc_pct = nwc_level / nwc_rev if nwc_rev else None
    nwc0 = nwc_level / M_
    int_exp = abs(last(A.get("InterestExpense"))) / M_ if last(A.get("InterestExpense")) is not None else 0
    if last(A.get("InterestExpense")) is None:
        warn(
            "capital.kd_pretax",
            "No interest expense in the feed; coverage looks unlimited and the cost of debt gets the AAA spread",
        )
    # year-1 margin from consensus EPS: EBIT = EPS × diluted shares ÷ (1 − tax) + interest, so the model's year-1 EPS equals consensus
    sh_dil = shares + dilution if shares is not None else None
    m_cons = (
        (eps_y1 * sh_dil / (1 - eff) + int_exp) / y1
        if eps_y1 is not None and eps_y1 > 0 and rev_y1 and y1 and sh_dil
        else None
    )
    if m_cons is not None and m_ttm is not None and abs(m_cons - m_ttm) > 0.10:
        warn(
            "scenarios.Base.margins.total.0",
            f"Consensus EPS implies a {fx(m_cons * 100, 1)}% operating margin next year against {fx(m_ttm * 100, 1)}% trailing; Base year 1 uses consensus",
        )
    both = m_cons is not None and m_ttm is not None
    m1 = {
        "Base": _first(m_cons, m_ttm),
        "Bear": min(m_cons, m_ttm) if both else m_ttm,
        "Bull": max(m_cons, m_ttm) if both else m_ttm,
    }
    # trailing EV/EBITDA and EV/Sales mislead when sales are ramping hard: lean on forward P/E instead
    ramp = (
        4 * last(Q.get("TotalRevenue")) / ttm_rev
        if ttm_rev and last(Q.get("TotalRevenue")) is not None
        else None
    )
    rampy = not FIN and ramp is not None and ramp > 1.3
    if rampy:
        warn(
            "relative.weights",
            f"Latest quarter annualised is {fx(ramp, 2)}x trailing revenue; trailing EV/EBITDA and EV/Sales understate the business, so comps use forward P/E only",
        )
    sh = A.get("OrdinarySharesNumber") or []
    buyback = max(0, 1 - (sh[-1][1] / sh[-3][1]) ** 0.5) if len(sh) >= 3 else 0

    rf = f["tnx"] / 100 if num(f.get("tnx")) is not None else None
    erp = f.get("erp") if f.get("erp") is not None else 0.0423
    bd, beta_data = None, None
    if f.get("stock") and f.get("index"):
        rs, rm, dates = returns_from_prices(f["stock"], f["index"])
        if len(rs) > 504:
            rs, rm, dates = rs[-504:], rm[-504:], dates[-504:]
        if len(rs) >= 60:
            bd = capm_beta(rs, rm)
            bd["window"] = f"{dates[0]} to {dates[-1]}"
            bd["frequency"] = "daily (Yahoo adjusted closes)"
            step = max(1, len(rs) // 600)
            bd["points"] = [[rm[i], y] for i, y in enumerate(rs) if i % step == 0]
            bd["xmin"], bd["xmax"] = min(rm), max(rm)
            im = dict(f["index"])
            al = sorted([x for x in f["stock"] if x[0] in im], key=lambda x: x[0])[-(len(rs) + 1) :]
            beta_data = {
                "dates": [x[0] for x in al],
                "stock": [x[1] for x in al],
                "index": [im[x[0]] for x in al],
            }
    ebit_l = last(A.get("OperatingIncome")) / M_ if num(last(A.get("OperatingIncome"))) is not None else None
    cov = ebit_l / int_exp if int_exp > 0 and ebit_l is not None else 99
    _, rating, spread = synthetic(cov)

    cash_raw = _first(
        last(Q.get("CashAndCashEquivalents")), last(A.get("CashAndCashEquivalents")), raw(fd.get("totalCash"))
    )
    debt_raw = _first(last(Q.get("TotalDebt")), last(A.get("TotalDebt")), raw(fd.get("totalDebt")))
    cash = cash_raw / M_ if num(cash_raw) is not None else None
    total_debt = debt_raw / M_ if num(debt_raw) is not None else None
    leases = _first(last(Q.get("CapitalLeaseObligations")), 0) / M_
    nci = _first(last(Q.get("MinorityInterest")), last(A.get("MinorityInterest")), 0) / M_
    if last(Q.get("TotalDebt")) is None and debt_raw is not None:
        warn(
            "bridge.core_debt",
            f"Debt is from the {'last annual report' if last(A.get('TotalDebt')) is not None else 'financialData summary'}, not the latest quarter",
        )
    if last(Q.get("CashAndCashEquivalents")) is None and cash_raw is not None:
        warn("bridge.cash", "Cash is not from the latest quarter")
    fin_ccy, px_ccy = fd.get("financialCurrency") or None, pr.get("currency") or None

    fc = {}
    if FIN:
        cse_q = Q.get("CommonStockEquity") or Q.get("StockholdersEquity")
        cse_a = A.get("CommonStockEquity") or A.get("StockholdersEquity")
        bv_raw = _first(last(cse_q), last(cse_a))
        fc["bv0"] = bv_raw / M_ if num(bv_raw) is not None else None
        fc["bvDate"] = last_date(cse_q) or last_date(cse_a)
        fc["tbv0"] = (
            last(Q.get("TangibleBookValue")) / M_ if last(Q.get("TangibleBookValue")) is not None else None
        )
        ni_q = Q.get("NetIncomeCommonStockholders") or Q.get("NetIncome")
        ni_ttm = _first(
            sum_last(ni_q, 4), last(A.get("NetIncomeCommonStockholders")), last(A.get("NetIncome"))
        )
        eq_win = [x[1] for x in (cse_q or [])[-5:]]
        avg_eq = _jsum(eq_win) / len(eq_win) if eq_win else bv_raw
        fc["roeTTM"] = ni_ttm / avg_eq if ni_ttm is not None and avg_eq is not None and avg_eq > 0 else None
        ni_a = A.get("NetIncomeCommonStockholders") or A.get("NetIncome") or []
        rh = []
        for d, val in ni_a:
            i = next((j for j, e in enumerate(cse_a or []) if e[0] == d), -1)
            if i > 0:
                rh.append(val / ((cse_a[i][1] + cse_a[i - 1][1]) / 2))
        fc["roeHist"] = rh
        roe_avg = _jsum(rh) / len(rh) if rh else fc["roeTTM"]
        fc["roe1"] = (
            eps_y1 * shares / fc["bv0"] if eps_y1 is not None and shares and fc["bv0"] else fc["roeTTM"]
        )
        fc["roeLR"] = (
            (fc["roeTTM"] + roe_avg) / 2 if fc["roeTTM"] is not None and roe_avg is not None else None
        )
        fc["roeMin"] = (
            min([*rh, fc["roeTTM"] if fc["roeTTM"] is not None else math.inf]) if rh else fc["roeTTM"]
        )
        pay = []
        for d, val in ni_a[-3:]:
            dv, bb = find(A.get("CashDividendsPaid"), d), find(A.get("RepurchaseOfCapitalStock"), d)
            if val > 0 and dv:
                pay.append((abs(dv[1]) + abs(bb[1] if bb else 0)) / val)
        fc["payout"] = min(1.2, max(0, _jsum(pay) / len(pay))) if pay else None
        dps = []
        for d, val in A.get("CashDividendsPaid") or []:
            s_ = find(A.get("DilutedAverageShares"), d)
            if s_:
                dps.append([d, abs(val) / s_[1]])
        fc["dpsHist"] = dps
        dg = (
            (dps[-1][1] / dps[-4][1]) ** (1 / 3) - 1
            if len(dps) >= 4
            else (dps[-1][1] / dps[-2][1] - 1 if len(dps) >= 2 else None)
        )
        fc["dpsG"] = min(0.10, max(-0.05, dg)) if dg is not None else None
        drate = _first(raw(sd.get("dividendRate")), raw(sd.get("trailingAnnualDividendRate")))
        fc["dps0"] = drate if drate is not None and drate > 0 else None
        ffo = (
            last(A.get("NetIncome")) + last(A.get("ReconciledDepreciation"))
            if last(A.get("NetIncome")) is not None and last(A.get("ReconciledDepreciation")) is not None
            else None
        )
        fc["coverage"] = (
            ffo / abs(last(A.get("CashDividendsPaid")))
            if ffo is not None and last(A.get("CashDividendsPaid"))
            else None
        )
        if fc["bv0"] is None:
            DQ["blocking"].append(
                {
                    "path": "fin.bv0",
                    "msg": "Common book equity is missing from the feed; enter it from the latest 10-Q",
                }
            )
        if MODEL == "excess_return":
            if fc["payout"] is None:
                warn("fin.payout", "No dividend or buyback history; payout set to 50%")
            if eps_y1 is None:
                warn("scenarios.Base.roe.0", "No consensus EPS; year-1 ROE uses the trailing ROE")
            if len(rh) < 2:
                warn(
                    "scenarios.Base.roe.1",
                    "Under two years of ROE history; the long-run ROE leans on the trailing year",
                )
            if fc["tbv0"] is not None and fc["bv0"] and fc["tbv0"] / fc["bv0"] < 0.6:
                warn(
                    "fin.tbv0",
                    f"Tangible book is only {fx(fc['tbv0'] / fc['bv0'] * 100, 0)}% of book; goodwill-heavy balance sheet",
                )
        else:
            if fc["dps0"] is None:
                DQ["blocking"].append(
                    {"path": "fin.dps0", "msg": "No dividend rate in the feed; a dividend model needs one"}
                )
            if fc["dpsG"] is None:
                warn(
                    "scenarios.Base.dps_growth.0",
                    "Under two years of dividend history; dividend growth set to terminal growth",
                )
            if fc["coverage"] is not None and fc["coverage"] < 1:
                warn(
                    "fin.coverage",
                    f"Dividends exceed net income + D&A (cover {fx(fc['coverage'], 2)}x); the dividend may not be sustainable",
                )
            warn(
                "fin.dps0",
                "REIT value uses dividends; FFO/AFFO are not in Yahoo's feed. Check the payout against AFFO in the supplement",
            )

    fin_scen = {}
    fin_rel_w = {"excess_return": {"pe_fwd": 0.5, "pb": 0.5}, "ddm": {"div_yield": 0.5, "ev_ebitda": 0.5}}
    fin_ver_w = {
        "excess_return": {"dcf": 0.4, "mc": 0.2, "comps": 0.3, "regression": 0.1},
        "ddm": {"dcf": 0.5, "mc": 0.2, "comps": 0.3, "regression": 0},
    }
    if FIN:
        fin_scen["excess_return"] = {
            "Bear": {"roe": [fc["roe1"], fc["roeMin"]]},
            "Base": {"roe": [fc["roe1"], fc["roeLR"]]},
            "Bull": {
                "roe": [
                    fc["roe1"],
                    max(fc["roeTTM"], fc["roeLR"])
                    if fc["roeLR"] is not None and fc["roeTTM"] is not None
                    else None,
                ]
            },
        }
        g0 = (
            fc["dpsG"]
            if fc["dpsG"] is not None
            else min(0.03, max(0.01, (rf if rf is not None else 0.04) - 0.01))
        )
        fin_scen["ddm"] = {
            "Bear": {"dps_growth": [g0 - 0.02] * 4},
            "Base": {"dps_growth": [g0] * 4},
            "Bull": {"dps_growth": [g0 + 0.01] * 4},
        }

    cols = [
        "name",
        "price",
        "mcap",
        "ev",
        "beta",
        "pe_fwd",
        "ev_ebitda",
        "ev_sales",
        "eps_g3y",
        "debt",
        "rev_ttm",
        "pb",
        "div_yield",
        "roe",
    ]
    rows = {}
    for t, ps in (f.get("peers") or {}).items():
        if not ps:
            continue
        ppr, psd, pks, pfd = (
            ps.get("price") or {},
            ps.get("summaryDetail") or {},
            ps.get("defaultKeyStatistics") or {},
            ps.get("financialData") or {},
        )
        ptr = next(
            (x for x in ((ps.get("earningsTrend") or {}).get("trend") or []) if x.get("period") == "+1y"),
            None,
        )
        rows[t] = [
            ppr.get("longName") or ppr.get("shortName") or t,
            raw(ppr.get("regularMarketPrice")),
            (raw(ppr.get("marketCap")) or 0) / M_,
            (raw(pks.get("enterpriseValue")) or 0) / M_,
            raw(psd.get("beta")),
            _first(raw(psd.get("forwardPE")), raw(pks.get("forwardPE"))),
            raw(pks.get("enterpriseToEbitda")),
            raw(pks.get("enterpriseToRevenue")),
            raw(ptr.get("growth")) if ptr else None,
            (raw(pfd.get("totalDebt")) or 0) / M_,
            (raw(pfd.get("totalRevenue")) or 0) / M_,
            raw(pks.get("priceToBook")),
            raw(psd.get("dividendYield")),
            raw(pfd.get("returnOnEquity")),
        ]
    peer_list = list(rows)
    if len(peer_list) < 3:
        warn(
            "peers.core_set",
            f"Only {len(peer_list)} peer{'' if len(peer_list) == 1 else 's'} with data; comps are weak",
        )
    missing = [t for t, x in (f.get("peers") or {}).items() if not x]
    if missing:
        warn("peers.core_set", f"No data returned for {', '.join(missing)}")
    if not bd:
        warn(
            "capital.beta_daily",
            "Not enough daily prices for a regression beta; the Blume-adjusted 5-year beta is used",
        )
    weak_beta = bool(bd and bd["r2"] < 0.05 and raw(sd.get("beta")) is not None)
    if weak_beta:
        warn(
            "capital.beta_source",
            f"Daily regression beta {fx(bd['beta'], 2)} explains only {fx(bd['r2'] * 100, 1)}% of the stock's moves (R²); "
            f"using Yahoo's 5-year monthly beta {fx(raw(sd.get('beta')), 2)}, Blume-adjusted. Switch on the BETA tab",
        )

    def scen(dgv, m_y1, m_lr):
        g = g_next + dgv
        return {
            "growth": {"total": [g, g * 0.75 + 0.01, g * 0.5 + 0.015, g * 0.35 + 0.02]},
            "margins": {"total": [m_y1, m_lr]},
        }

    ud = []
    for h in ((S.get("upgradeDowngradeHistory") or {}).get("history") or [])[:8]:
        suffix = (
            " (downgrade)" if h.get("action") == "down" else " (upgrade)" if h.get("action") == "up" else ""
        )
        ud.append(
            [
                h.get("firm"),
                h.get("toGrade") + suffix,
                h.get("currentPriceTarget") or None,
                _iso(h["epochGradeDate"]),
            ]
        )
    rt = ((S.get("recommendationTrend") or {}).get("trend") or [{}])[0] or {}
    div_rate = _first(raw(sd.get("dividendRate")), raw(sd.get("trailingAnnualDividendRate")), 0)

    ann_labels = [fy(d) for d, _ in (TR or [])]

    def pick(s):
        return [(x[1] / M_ if x else None) for d, _ in (TR or []) for x in [find(s, d)]]

    QR = (Q.get("TotalRevenue") or [])[-8:]
    qlab = [d[:7] for d, _ in QR]

    def qpick(s):
        return [(x[1] / M_ if x else None) for d, _ in QR for x in [find(s, d)]]

    def qcol(k):
        return [(x[1] / M_ if x else None) for d in [e[0] for e in (TR or [])] for x in [find(A.get(k), d)]]

    consensus_note = (
        "Built from Yahoo Finance consensus. Base-case growth starts at consensus revenue for the next two fiscal years and fades after that."
        + (
            f" Yahoo's {labels[1]} EPS average ({fx(eps_y2avg, 2)}) disagrees with its own growth figure (implies {fx(eps_y2g, 2)}); the growth-implied figure is used."
            if eps_conflict
            else ""
        )
    )
    inp = {
        "meta": {
            "ticker": f["ticker"],
            "name": pr.get("longName") or pr.get("shortName") or f["ticker"],
            "exchange": pr.get("exchangeName") or "",
            "currency": px_ccy or "USD",
            "financial_currency": fin_ccy or px_ccy or "USD",
            "units": "millions except per-share",
            "valuation_date": today,
            "price_date": today,
            "fiscal_year_end_next": next_fye,
            "balance_sheet_date": bs_date,
            "year_labels": labels,
            "sector": prof.get("sector") or "",
            "industry": prof.get("industry") or "",
            "built": f.get("built") or "live research (Yahoo Finance feeds via the Claude app browser)",
        },
        "market": {
            "price": price,
            "prev_close": _first(raw(pr.get("regularMarketPreviousClose")), price),
            "shares_basic": shares,
            "dilution": dilution,
            "dividend_ps": div_rate,
            "high_52w": _first(raw(sd.get("fiftyTwoWeekHigh")), price),
            "low_52w": _first(raw(sd.get("fiftyTwoWeekLow")), price),
            "eps_ntm": _first(raw(ks.get("forwardEps")), eps_y2g, eps_y1),
            "ebitda_ttm": raw(fd.get("ebitda")) / M_ if num(raw(fd.get("ebitda"))) is not None else None,
            "revenue_ttm": raw(fd.get("totalRevenue")) / M_
            if num(raw(fd.get("totalRevenue"))) is not None
            else None,
            "total_debt_incl_dfs": total_debt,
        },
        "model": MODEL,
        "rev_y0": rev0,
        "segment_defs": [{"key": "total", "name": "Total revenue", "margin": "total", "report": "Company"}],
        "margin_groups": {"total": "Company EBIT margin"},
        "ui": {
            "key_segment": "total",
            "key_margin": "total",
            "key_segment_label": "Revenue",
            "key_margin_label": "EBIT",
        },
        "y1_segments": {"total": y1},
        "scenarios": fin_scen[MODEL]
        if FIN
        else {
            "Bear": scen(-0.05, m1["Bear"], bear_lr),
            "Base": scen(0, m1["Base"], base_lr),
            "Bull": scen(0.04, m1["Bull"], max(_z(m_ttm), _z(base_lr))),
        },
        "fin": {
            "bv0": fc["bv0"],
            "tbv0": fc["tbv0"],
            "bv_date": fc["bvDate"],
            "roe_ttm": fc["roeTTM"],
            "roe_hist": fc["roeHist"],
            "payout": fc["payout"] if fc["payout"] is not None else 0.5,
            "dps0": fc["dps0"],
            "dps_hist": fc["dpsHist"],
            "dps_cagr": fc["dpsG"],
            "dividend_cover": fc["coverage"],
        }
        if FIN
        else _UNDEF,
        "operating": {
            "corp_pct": 0,
            "tax0": eff,
            "tax_lr": max(eff, 0.18),
            "da_pct": da_pct,
            "capex_pct": capex_pct,
            "nwc_pct": nwc_pct,
            "nwc0": nwc0,
            "interest_net": int_exp,
            "buyback_pct": buyback,
        },
        "terminal": {
            "method": "gordon",
            "g": min(0.03, max(0.01, rf - 0.01)) if rf is not None else 0.025,
            "ronic": 0.15,
            "exit_multiple": 12.0,
            **({"er_persistence": 0.5} if FIN else {}),
        },
        "timing": {"first_period": first_period, "stub": stub, "mid_year": True},
        "capital": {
            "rf": rf,
            "erp": erp,
            "rm": rf + erp if rf is not None else None,
            "crp": 0,
            "beta_raw": _first(raw(sd.get("beta")), bd["beta"] if bd else 1),
            "beta_source": "daily_adj" if bd and not weak_beta else "blume",
            "beta_custom": bd["beta"] if bd else _first(raw(sd.get("beta")), 1),
            "beta_daily": bd["beta"] if bd else None,
            "beta_daily_stats": bd,
            "beta_data": beta_data,
            "kd_pretax": rf + spread if rf is not None else None,
            "tax_marginal": 0.21,
            "size_premium": 0,
        },
        "bridge": {
            "cash": cash,
            "core_debt": total_debt - leases if total_debt is not None else None,
            "leases": leases,
            "dfs_receivables": 0,
            "dfs_lease_equipment": 0,
            "dfs_debt": 0,
            "investments": 0,
            "nci": nci,
        },
        "consensus": {
            "eps": {labels[0]: eps_y1, labels[1]: eps_y2g if eps_conflict else _first(eps_y2avg, eps_y2g)},
            "revenue": {
                labels[0]: rev_y1 / M_ if rev_y1 else None,
                labels[1]: rev_y2 / M_ if rev_y2 else None,
            },
            "targets": {
                "low": raw(fd.get("targetLowPrice")),
                "mean": raw(fd.get("targetMeanPrice")),
                "median": raw(fd.get("targetMedianPrice")),
                "high": raw(fd.get("targetHighPrice")),
                "n": raw(fd.get("numberOfAnalystOpinions")) or 0,
            },
            "ratings": {
                "strong_buy": rt.get("strongBuy") or 0,
                "buy": rt.get("buy") or 0,
                "hold": rt.get("hold") or 0,
                "sell": rt.get("sell") or 0,
                "strong_sell": rt.get("strongSell") or 0,
            },
            "recent": ud,
            "note": consensus_note,
        },
        "peers": {
            "columns": cols,
            "rows": rows,
            "core_set": peer_list,
            "note": "Peer list starts from Yahoo's related-symbols feed (stocks viewed together), not an industry screen. Edit it to true comparables.",
        },
        "relative": {
            "weights": fin_rel_w[MODEL]
            if FIN
            else {"pe_fwd": 1, "ev_ebitda": 0, "ev_sales": 0}
            if rampy
            else {"pe_fwd": 0.4, "ev_ebitda": 0.4, "ev_sales": 0.2},
            "stat": "median",
            "subject_growth": g1eps,
        },
        "verdict": {
            "weights": fin_ver_w[MODEL] if FIN else {"dcf": 0.4, "mc": 0.2, "comps": 0.3, "regression": 0.1},
            "band": 0.10,
            "horizon_years": 1.0,
        },
        "alt_models": {
            k: {"scenarios": fin_scen[k], "relative_weights": fin_rel_w[k], "verdict_weights": fin_ver_w[k]}
            for k in ["excess_return", "ddm"]
            if k != MODEL and (k != "ddm" or fc["dps0"] is not None)
        }
        if FIN
        else _UNDEF,
        "history": {
            "annual": {
                "labels": ann_labels,
                "revenue": pick(TR),
                "op_income": pick(A.get("OperatingIncome")),
                "net_income": pick(A.get("NetIncome")),
                "eps_diluted": [
                    (x[1] if x else None) for d, _ in (TR or []) for x in [find(A.get("DilutedEPS"), d)]
                ],
                "cfo": pick(A.get("OperatingCashFlow")),
                "capex": [None if x is None else abs(x) for x in pick(A.get("CapitalExpenditure"))],
                "fcf": pick(A.get("FreeCashFlow")),
            },
            "annual_names": {
                "revenue": "Revenue",
                "op_income": "Operating income",
                "net_income": "Net income",
                "eps_diluted": "Diluted EPS",
                "cfo": "Cash from operations",
                "capex": "Capex",
                "fcf": "Free cash flow",
            },
            "annual_note": "Yahoo Finance standardised statements. Operating income can differ from the company's GAAP line where Yahoo normalises unusual items.",
            "quarters": {
                "labels": qlab,
                "revenue": qpick(Q.get("TotalRevenue")),
                "op_income": qpick(Q.get("OperatingIncome")),
            },
            "quarter_names": {"revenue": "Revenue", "op_income": "Operating income"},
            "quarter_margins": [["Operating margin", "op_income", "revenue"]],
            "prices": (f.get("stock") or [])[-85:],
        },
        "quality_data": {
            "labels": [fy(d) for d, _ in (TR or [])],
            "dates": [d for d, _ in (TR or [])],
            "sales": qcol("TotalRevenue"),
            "cogs": qcol("CostOfRevenue"),
            "sga": qcol("SellingGeneralAndAdministration"),
            "rec": qcol("AccountsReceivable"),
            "ca": qcol("CurrentAssets"),
            "ppe": qcol("NetPPE"),
            "ta": qcol("TotalAssets"),
            "cl": qcol("CurrentLiabilities"),
            "ltd": qcol("LongTermDebt"),
            "dep": qcol("ReconciledDepreciation"),
            "ni": qcol("NetIncomeContinuousOperations")
            if A.get("NetIncomeContinuousOperations")
            else qcol("NetIncome"),
            "cfo": qcol("OperatingCashFlow"),
        },
        "bridge_names": {
            "cash": "Cash & equivalents",
            "core_debt": "Total debt ex leases",
            "leases": "Capital leases",
            "investments": "Investments",
            "nci": "Non-controlling interests",
            "dfs_receivables": "Captive-finance receivables",
            "dfs_lease_equipment": "Captive-finance lease equipment",
            "dfs_debt": "Captive-finance debt",
        },
        "bridge_note": "Captive-finance arms and non-operating investments are not in Yahoo's feed. Check the 10-K and enter them if the company has them.",
        "notes": {
            "kd": f"(rf + {rating} spread, coverage {fx(cov, 1)}x)",
            "erp": "Damodaran implied ERP, Jan-2026 (edit to update)",
            "rm": f"Rm = 10Y yield + ERP ({fx(rf * 100, 2)}% + {fx(erp * 100, 2)}%). Replace with your own expected S&P 500 return."
            if rf is not None
            else "10-year yield missing: enter it and set Rm.",
            "segments": "Single revenue line. Split into reported segments for a sell-side-grade model.",
            "operating": f"Margins: TTM {fx(_z(m_ttm) * 100, 1)}%, {len(hist)}-year average {fx(_z(m_avg) * 100, 1)}%, low {fx(_z(m_min) * 100, 1)}%. D&A and capex are 3-year averages of sales.",
            "comps": "All multiples come from the same vendor, so definitions match across peers.",
        },
        "sources": {
            "Price, market cap, shares, beta (5Y), forward P/E, EV multiples, debt, cash, EBITDA, revenue (TTM), targets": f"Yahoo Finance quoteSummary, {today}",
            "Annual and quarterly statements": f"Yahoo Finance fundamentals-timeseries, {today}",
            "Daily prices and beta regression": f"Yahoo Finance chart API (adjusted closes) for {f['ticker']} and ^GSPC, {bd['window'] if bd else 'n/a'}",
            "Risk-free rate": f"^TNX (10-year Treasury yield) {fx(f['tnx'], 3) + '%' if f.get('tnx') is not None else 'n/a'}, {today}",
            "Equity risk premium": "Damodaran implied ERP, Jan-2026 (assumption, editable)",
            "Cost of debt": f"Synthetic rating {rating} from interest coverage {fx(cov, 1)}x (Damodaran large-firm table)",
            "Peers": f"Yahoo related symbols: {', '.join(peer_list) or 'none'}",
        },
        "provenance": P,
        "data_quality": DQ,
    }
    if mcap and price and shares and abs(shares * M_ * price / mcap - 1) > 0.02:
        warn("market.shares_basic", "Shares × price differs from the quoted market cap by more than 2%")

    src("market.price", "sourced", f"{YQ} price.regularMarketPrice")
    src(
        "market.shares_basic",
        "sourced",
        f"{YQ} defaultKeyStatistics.impliedSharesOutstanding (checked against marketCap ÷ price)",
    )
    src(
        "market.dilution",
        "derived" if dilution else "assumption",
        f"{YTQ}: diluted average shares − ordinary shares, latest quarter"
        if dilution
        else "no dilution found in feed",
    )
    src("market.dividend_ps", "sourced", f"{YQ} summaryDetail.dividendRate")
    src("market.eps_ntm", "sourced", f"{YQ} defaultKeyStatistics.forwardEps")
    src("market.ebitda_ttm", "sourced", f"{YQ} financialData.ebitda (TTM)")
    src("market.revenue_ttm", "sourced", f"{YQ} financialData.totalRevenue (TTM)")
    src("market.total_debt_incl_dfs", "sourced", f"{YTQ} TotalDebt, {bs_date}")
    src("rev_y0", "sourced", f"{YT} TotalRevenue, {last_fye}")
    src(
        "y1_segments.total",
        "sourced" if rev_y1 else "derived",
        f"{YQ} earningsTrend 0y revenue estimate (consensus)"
        if rev_y1
        else "last revenue × last year's growth",
    )
    for s in ["Bear", "Base", "Bull"]:
        src(
            f"scenarios.{s}.growth.total.0",
            "sourced" if s == "Base" and rev_y2 else "assumption",
            f"consensus: {labels[1]} ÷ {labels[0]} revenue estimate − 1"
            if s == "Base"
            else f"Base {'−5pp' if s == 'Bear' else '+4pp'}",
        )
        for k in (1, 2, 3):
            src(f"scenarios.{s}.growth.total.{k}", "assumption", "fade: g×0.75+1%, g×0.5+1.5%, g×0.35+2%")
        src(
            f"scenarios.{s}.margins.total.0",
            "derived",
            f"TTM operating margin {YTQ} (4 quarters)"
            if m_cons is None
            else f"{YQ} consensus {labels[0]} EPS × diluted shares ÷ (1 − tax) + interest, ÷ consensus revenue (consensus EPS may be on an adjusted basis)"
            if s == "Base"
            else f"{'lower' if s == 'Bear' else 'higher'} of consensus-implied and TTM operating margin",
        )
        src(
            f"scenarios.{s}.margins.total.1",
            "assumption",
            "midpoint of TTM and multi-year average margin"
            if s == "Base"
            else (
                "multi-year average margin (the lowest year is a loss)"
                if bear_lr is not m_min
                else "lowest margin in the annual history"
            )
            if s == "Bear"
            else "TTM margin held",
        )
    src("operating.tax0", "derived", f"{YT} TaxProvision ÷ PretaxIncome (clamped 10–30%)")
    src("operating.tax_lr", "assumption", "max(effective rate, 18%)")
    src("operating.da_pct", "derived", f"{YT} D&A ÷ revenue, 3-year average")
    src("operating.capex_pct", "derived", f"{YT} capex ÷ revenue, 3-year average")
    src(
        "operating.nwc_pct",
        "derived",
        f"{YTQ} (receivables + inventory − payables) ÷ (4 × revenue), quarter to {qd}"
        if nwc_q
        else f"{YT} (receivables + inventory − payables) ÷ revenue",
    )
    src(
        "operating.nwc0",
        "derived",
        f"{YTQ if nwc_q else YT} receivables + inventory − payables, {qd if nwc_q else last_fye}; year 1 change = NWC % × revenue − this level",
    )
    src("operating.interest_net", "sourced", f"{YT} InterestExpense")
    src("operating.buyback_pct", "derived", f"{YT} share-count decline per year, last 2 years")
    src("operating.corp_pct", "assumption", "none (single segment)")
    src("terminal.g", "assumption", "3%, capped at rf − 1%")
    src("terminal.ronic", "assumption", "15% return on new invested capital")
    src("terminal.exit_multiple", "assumption", "12x EBITDA cross-check")
    src("timing.first_period", "derived", f"days from {today} to fiscal year end {next_fye}")
    src("timing.stub", "derived", f"share of the year after the last balance sheet ({bs_date})")
    src("capital.rf", "sourced", "^TNX 10-year Treasury yield, Yahoo chart API")
    src("capital.erp", "assumption", "Damodaran implied ERP, Jan-2026")
    src("capital.rm", "assumption", "rf + ERP")
    src("capital.crp", "assumption", "0 (US)")
    src("capital.beta_raw", "sourced", f"{YQ} summaryDetail.beta (5Y monthly)")
    src("capital.beta_custom", "derived", "copy of daily regression beta")
    src(
        "capital.kd_pretax",
        "derived",
        f"rf + {rating} spread; coverage = operating income ÷ interest expense",
    )
    src("capital.tax_marginal", "assumption", "21% US federal statutory rate")
    src("bridge.cash", "sourced", f"{YTQ} CashAndCashEquivalents, {bs_date}")
    src("bridge.core_debt", "sourced", f"{YTQ} TotalDebt − capital leases, {bs_date}")
    src("bridge.leases", "sourced", f"{YTQ} CapitalLeaseObligations")
    src("bridge.nci", "sourced", f"{YTQ} MinorityInterest")
    src("bridge.investments", "assumption", "not in feed; enter from the 10-K")
    for k in ("pe_fwd", "ev_ebitda", "ev_sales"):
        src(
            f"relative.weights.{k}",
            "assumption",
            "forward P/E only (sales ramping: trailing multiples left out)" if rampy else "40/40/20 blend",
        )
    for k in ("dcf", "mc", "comps", "regression"):
        src(f"verdict.weights.{k}", "assumption", "method weights")
    src("verdict.band", "assumption", "±10% treated as fairly valued")
    src("verdict.horizon_years", "assumption", "convergence horizon")
    src(
        "quality_data",
        "sourced",
        f"{YT}: revenue, cost of revenue, SG&A, receivables, current assets, net PP&E, total assets, current liabilities, long-term debt, D&A, net income (continuing), operating cash flow",
    )
    src(
        "model",
        "assumption",
        f'routed from Yahoo industry "{ind or "n/a"}"'
        + (
            " → excess-return model"
            if MODEL == "excess_return"
            else " → dividend discount model"
            if MODEL == "ddm"
            else " → FCFF DCF"
        )
        + "; change on the DES tab",
    )
    if FIN:
        for k in list(P):
            if fcff_only.match(k) or k.startswith("relative.weights."):
                del P[k]
        src("fin.bv0", "sourced", f"{YTQ} CommonStockEquity, {fc['bvDate']}")
        src("fin.tbv0", "sourced", f"{YTQ} TangibleBookValue")
        src(
            "fin.roe_ttm",
            "derived",
            f"{YTQ} net income to common (4 quarters) ÷ average common equity (5 quarter-ends)",
        )
        src(
            "fin.payout",
            "derived" if fc["payout"] is not None else "assumption",
            f"{YT} (dividends + buybacks) ÷ net income to common, 3-year average"
            if fc["payout"] is not None
            else "50% (no history)",
        )
        src("fin.dps0", "sourced", f"{YQ} summaryDetail.dividendRate (annualised)")
        src(
            "terminal.er_persistence",
            "assumption",
            "half of the long-run ROE spread over Ke survives in the terminal value",
        )
        for w in inp["relative"]["weights"]:
            src(f"relative.weights.{w}", "assumption", "50/50 blend")
        for sn in ["Bear", "Base", "Bull"]:
            if MODEL == "excess_return":
                src(
                    f"scenarios.{sn}.roe.0",
                    "derived",
                    "consensus EPS (0y) × shares ÷ book equity" if eps_y1 is not None else "trailing ROE",
                )
                src(
                    f"scenarios.{sn}.roe.1",
                    "assumption",
                    "midpoint of trailing ROE and the annual ROE average"
                    if sn == "Base"
                    else "lowest annual ROE"
                    if sn == "Bear"
                    else "trailing ROE held",
                )
            else:
                for k in range(4):
                    src(
                        f"scenarios.{sn}.dps_growth.{k}",
                        "derived" if sn == "Base" and fc["dpsG"] is not None else "assumption",
                        (
                            f"{YT} dividends per share (cash dividends ÷ diluted shares), 3-year CAGR"
                            if fc["dpsG"] is not None
                            else "terminal growth (no history)"
                        )
                        if sn == "Base"
                        else f"Base {'−2pp' if sn == 'Bear' else '+1pp'}",
                    )
        inp["notes"]["operating"] = (
            f"Excess-return model. ROE: trailing {fx(_z(fc['roeTTM']) * 100, 1)}%, annual history {', '.join(fx(x * 100, 1) + '%' for x in fc['roeHist'])}. Book grows by retained earnings: ROE × (1 − payout)."
            if MODEL == "excess_return"
            else f"Dividend discount model. Dividend per share {fx(fc['dps0'], 2) if fc['dps0'] is not None else 'n/a'}, 3-year growth {fx(fc['dpsG'] * 100, 1) + '%' if fc['dpsG'] is not None else 'n/a'}; (net income + D&A) covers dividends {fx(fc['coverage'], 2) + 'x' if fc['coverage'] is not None else 'n/a'}."
        )
        inp["notes"]["comps"] = (
            "Banks and insurers are compared on forward P/E and price-to-book; EV multiples mean nothing when debt is the product."
            if MODEL == "excess_return"
            else "REITs are compared on dividend yield and EV/EBITDA; Yahoo has no P/FFO."
        )
    return _clean(inp)


def _first(*xs):
    """JavaScript a ?? b ?? c."""
    for x in xs:
        if x is not None:
            return x
    return None


def from_fixture(fx_):
    """A saved snapshot (raw Yahoo shapes, or compact-v1) → the feeds dict build() takes."""
    cp = fx_.get("format") == "compact-v1"
    return {
        "ticker": fx_["ticker"],
        "today": fx_.get("today"),
        "tnx": fx_.get("tnx"),
        "summary": fx_["summary"],
        "annual": fx_["annual"] if cp else series(fx_["annual"]),
        "quarterly": fx_["quarterly"] if cp else series(fx_["quarterly"]),
        "stock": fx_["stock"] if cp else closes(fx_["stock"]),
        "index": fx_["index"] if cp else closes(fx_["index"]),
        "peers": fx_.get("peers"),
    }
