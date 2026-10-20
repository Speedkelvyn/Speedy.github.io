"""
valuation_v2.py -- segment-build DCF + relative valuation, any ticker.

Reads one inputs JSON (see dell_inputs.json). Mirrors web/model.js line
for line; tests/test_parity.py proves both give identical numbers.

    python valuation_v2.py dell_inputs.json        # prints + writes Excel

Pipeline (sell-side order):
  segments -> margins -> EBIT -> NOPAT -> FCFF -> WACC -> DCF (Gordon with
  RONIC, exit-multiple cross-check) -> equity bridge (core debt only; the
  financing arm valued at its net book) -> reverse DCF -> sensitivity ->
  Monte Carlo -> comps + P/E-growth regression -> verdict & expected alpha.
"""
from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pandas as pd

DEFAULT_DEFS = [
    {"key": "ai", "name": "AI-optimised servers", "margin": "ai", "report": "ISG"},
    {"key": "trad", "name": "Traditional servers & networking", "margin": "core", "report": "ISG"},
    {"key": "storage", "name": "Storage", "margin": "core", "report": "ISG"},
    {"key": "comm", "name": "CSG commercial", "margin": "csg", "report": "CSG"},
    {"key": "cons", "name": "CSG consumer", "margin": "csg", "report": "CSG"},
    {"key": "other", "name": "Other", "margin": None, "report": "Other"},
]
N, EXPLICIT = 10, 4


def model_of(inp):
    """"fcff" (default), "excess_return" (banks, insurers) or "ddm" (REITs)."""
    return inp.get("model") or "fcff"


def is_fin(inp):
    return model_of(inp) != "fcff"


def defs(inp):
    return inp.get("segment_defs") or DEFAULT_DEFS


def seg_keys(inp):
    return [d["key"] for d in defs(inp)]


def key_seg(inp):
    return (inp.get("ui") or {}).get("key_segment") or defs(inp)[0]["key"]


def key_margin(inp):
    return (inp.get("ui") or {}).get("key_margin") or next(d["margin"] for d in defs(inp) if d["margin"])


# ---------------------------------------------------------- cost of capital
def diluted_shares(inp):
    return inp["market"]["shares_basic"] + inp["market"]["dilution"]


def bottom_up_beta(inp):
    t = inp["capital"]["tax_marginal"]
    cols = inp["peers"]["columns"]
    ix = {c: i for i, c in enumerate(cols)}
    bu = []
    for tk in inp["peers"]["core_set"]:
        r = inp["peers"]["rows"][tk]
        b = r[ix["beta"]]                    # a peer with no published beta is left out, not set to 1
        bu.append(None if b is None else b / (1 + (1 - t) * r[ix["debt"]] / r[ix["mcap"]]))
    ok = [x for x in bu if x is not None]
    med = float(np.median(ok)) if ok else float("nan")
    E = inp["market"]["price"] * diluted_shares(inp)
    D = inp["bridge"]["core_debt"] + inp["bridge"]["leases"]
    return {"unlevered": bu, "median": med,
            "relevered": med * (1 + (1 - t) * D / E)}


def beta(inp):
    c = inp["capital"]
    blume = 0.67 * c["beta_raw"] + 0.33
    bu = bottom_up_beta(inp)
    src = c["beta_source"]
    daily = c.get("beta_daily")
    chosen = (c["beta_raw"] if src == "raw" else bu["relevered"] if src == "bottom_up"
              else c["beta_custom"] if src == "custom"
              else daily if src == "daily" and daily is not None
              else 0.67 * daily + 0.33 if src == "daily_adj" and daily is not None
              else c["beta_crsp_monthly"] if src == "crsp_monthly" and c.get("beta_crsp_monthly") is not None
              else 0.67 * c["beta_crsp_monthly"] + 0.33 if src == "crsp_monthly_adj" and c.get("beta_crsp_monthly") is not None
              else blume)
    return {"raw": c["beta_raw"], "blume": blume, "bottom_up": bu["relevered"],
            "unlevered_median": bu["median"], "daily": daily, "crsp": c.get("beta_crsp_monthly"), "chosen": chosen}


def market_risk_premium(c):
    """MRP = Rm - rf when an expected market return is given."""
    return c["rm"] - c["rf"] if c.get("rm") is not None else c["erp"]


def returns_from_prices(stock, index):
    """stock/index: lists of [date, close]; returns aligned simple returns."""
    m = dict(index)
    rows = sorted([r for r in stock if r[0] in m])
    rs = [rows[i][1] / rows[i - 1][1] - 1 for i in range(1, len(rows))]
    rm = [m[rows[i][0]] / m[rows[i - 1][0]] - 1 for i in range(1, len(rows))]
    return np.array(rs), np.array(rm), [r[0] for r in rows[1:]]


def capm_beta(rs, rm, periods_per_year=252):
    """OLS R_stock = a + b R_index (same regression as Excel's Data Analysis)."""
    rs, rm = np.asarray(rs, float), np.asarray(rm, float)
    n = rs.size
    mx, my = rm.mean(), rs.mean()
    sxx, syy = ((rm - mx) ** 2).sum(), ((rs - my) ** 2).sum()
    sxy = ((rm - mx) * (rs - my)).sum()
    b = sxy / sxx
    a = my - b * mx
    sse = ((rs - a - b * rm) ** 2).sum()
    s2 = sse / (n - 2)
    se_b, se_a = np.sqrt(s2 / sxx), np.sqrt(s2 * (1 / n + mx * mx / sxx))
    return {"beta": b, "alpha": a, "seBeta": se_b, "seAlpha": se_a, "tBeta": b / se_b,
            "tAlpha": a / se_a, "r2": 1 - sse / syy, "n": n,
            "alphaAnnual": (1 + a) ** periods_per_year - 1, "blume": 0.67 * b + 0.33,
            "ci95": [b - 1.96 * se_b, b + 1.96 * se_b],
            "volStock": np.sqrt(syy / (n - 1) * periods_per_year),
            "volMkt": np.sqrt(sxx / (n - 1) * periods_per_year)}


def wacc(inp):
    c, b = inp["capital"], beta(inp)
    mrp = market_risk_premium(c)
    ke = c["rf"] + b["chosen"] * mrp + c.get("crp", 0.0) + c.get("size_premium", 0.0)
    E = inp["market"]["price"] * diluted_shares(inp)
    D = inp["bridge"]["core_debt"] + inp["bridge"]["leases"]
    we, wd = E / (E + D), D / (E + D)
    kd = c["kd_pretax"] * (1 - c["tax_marginal"])
    return {"wacc": we * ke + wd * kd, "ke": ke, "kd_after": kd,
            "we": we, "wd": wd, "E": E, "D": D, "beta": b, "mrp": mrp, "rm": c["rf"] + mrp,
            "equityPremium": b["chosen"] * mrp}


# --------------------------------------------------------------- projection
def growth_path(explicit, gT):
    g = list(explicit[:EXPLICIT])
    last, fade = g[-1], N - 1 - EXPLICIT
    g += [last + (gT - last) * j / fade for j in range(1, fade + 1)]
    return g


def margin_path(m0, mLR):
    return [mLR if i >= EXPLICIT else m0 + (mLR - m0) * i / EXPLICIT
            for i in range(N)]


def project(inp, scen):
    op, gT = inp["operating"], inp["terminal"]["g"]
    SEGS, D = seg_keys(inp), defs(inp)
    base = inp.get("y1_segments") or inp["fy27_segments"]
    seg = {}
    for s in SEGS:
        g = growth_path(scen["growth"][s], gT)
        r = [base[s]]
        for i in range(N - 1):
            r.append(r[i] * (1 + g[i]))
        seg[s] = r
    mp = {k: margin_path(*v) for k, v in scen["margins"].items()}
    taxp = margin_path(op["tax0"], op["tax_lr"])
    rows, prev, shares = [], inp["rev_y0"], diluted_shares(inp)
    for i in range(N):
        rev = sum(seg[s][i] for s in SEGS)
        report_rev, report_oi, oi_sum = {}, {}, 0.0
        for d in D:
            oi = seg[d["key"]][i] * mp[d["margin"]][i] if d["margin"] else 0.0
            report_rev[d["report"]] = report_rev.get(d["report"], 0.0) + seg[d["key"]][i]
            report_oi[d["report"]] = report_oi.get(d["report"], 0.0) + oi
            oi_sum += oi
        corp = op["corp_pct"] * rev
        ebit = oi_sum - corp
        tax = taxp[i]
        nopat = ebit * (1 - tax)
        da, capex = op["da_pct"] * rev, op["capex_pct"] * rev
        # year 1 against the opening balance when it is known; afterwards NWC % × the change in revenue
        dnwc = op["nwc_pct"] * rev - op["nwc0"] if i == 0 and op.get("nwc0") is not None else op["nwc_pct"] * (rev - prev)
        fcff = nopat + da - capex - dnwc
        if i > 0:
            shares *= (1 - op["buyback_pct"])
        ni = (ebit - op["interest_net"]) * (1 - tax)
        ni_adj = (ebit + corp - op["interest_net"]) * (1 - tax)
        rows.append(dict(label=inp["meta"]["year_labels"][i], rev=rev,
                         growth=rev / prev - 1, seg={s: seg[s][i] for s in SEGS},
                         reportRev=report_rev, reportOI=report_oi, corp=corp, ebit=ebit,
                         margin=ebit / rev, tax=tax, nopat=nopat, da=da,
                         ebitda=ebit + da, capex=capex, dnwc=dnwc, fcff=fcff,
                         shares=shares, ni=ni, eps=ni / shares,
                         epsAdj=ni_adj / shares))
        prev = rev
    return rows


# --------------------------------------------------------------------- DCF
def dcf(inp, scen, wacc_override=None):
    if is_fin(inp):
        return fin_value(inp, scen, wacc_override)
    w = wacc(inp)["wacc"] if wacc_override is None else wacc_override
    tm = inp["terminal"]
    rows = project(inp, scen)
    tp = inp["timing"]
    fp, stub, mid = tp["first_period"], tp["stub"], tp["mid_year"]
    pv_sum = 0.0
    for i, r in enumerate(rows):
        frac = stub if i == 0 else 1.0
        t_end = fp + i
        t = max(t_end - 0.5 * frac if mid else t_end, 0.0)
        r.update(frac=frac, t=t, df=(1 + w) ** -t)
        r["pv"] = r["fcff"] * frac * r["df"]
        pv_sum += r["pv"]
    last, t_n = rows[-1], fp + N - 1
    if tm["g"] >= w:
        raise ValueError("terminal growth must be below WACC")
    tv_g = last["nopat"] * (1 + tm["g"]) * (1 - tm["g"] / tm["ronic"]) / (w - tm["g"])
    pv_g = tv_g * (1 + w) ** -((t_n - 0.5) if mid else t_n)
    tv_x = tm["exit_multiple"] * last["ebitda"]
    pv_x = tv_x * (1 + w) ** -t_n
    use_exit = tm["method"] == "exit"
    pv_tv = pv_x if use_exit else pv_g
    ev = pv_sum + pv_tv
    b = inp["bridge"]
    dfs_net = b["dfs_receivables"] + b["dfs_lease_equipment"] - b["dfs_debt"]
    adj = b["cash"] - b["core_debt"] - b["leases"] + dfs_net + b["investments"] - b["nci"]
    sh = diluted_shares(inp)
    return {"wacc": w, "rows": rows, "pvExplicit": pv_sum, "tvGordon": tv_g,
            "pvGordon": pv_g, "tvExit": tv_x, "pvExit": pv_x, "pvTV": pv_tv,
            "ev": ev, "dfsNet": dfs_net, "equity": ev + adj,
            "perShare": (ev + adj) / sh, "tvShare": pv_tv / ev,
            "impliedExitMultiple": tv_g / last["ebitda"],
            "impliedGFromExit": (tv_x * w - last["fcff"]) / (tv_x + last["fcff"]),
            "perShareGordon": (pv_sum + pv_g + adj) / sh,
            "perShareExit": (pv_sum + pv_x + adj) / sh}


# -------------------------------------------------------------- financials
# Discount rate is the cost of equity: for banks, insurers and REITs debt is raw material, not financing.
def roe_path(scen):
    return margin_path(scen["roe"][0], scen["roe"][1])


def dps_growth_path(explicit, gT):
    e = list(explicit[:EXPLICIT])
    return [e[i] if i < EXPLICIT else e[-1] + (gT - e[-1]) * (i - EXPLICIT + 1) / (N - EXPLICIT) for i in range(N)]


def fin_value(inp, scen, ke_override=None):
    r = wacc(inp)["ke"] if ke_override is None else ke_override
    tm = inp["terminal"]
    if tm["g"] >= r:
        raise ValueError("terminal growth must be below the cost of equity")
    tp = inp["timing"]
    fp, stub, mid = tp["first_period"], tp["stub"], tp["mid_year"]
    labels = (inp.get("meta") or {}).get("year_labels") or []
    sh, t_n = diluted_shares(inp), fp + N - 1
    t_tv = t_n - 0.5 if mid else t_n
    rows, pv_sum = [], 0.0
    lab = lambda i: labels[i] if i < len(labels) else f"Y{i + 1}"
    if model_of(inp) == "excess_return":
        # Equity = BV0 + sum PV[(ROE_t - Ke) x BV_{t-1}] + PV[terminal excess return]
        roe = roe_path(scen)
        p = scen["payout"] if scen.get("payout") is not None else inp["fin"]["payout"]
        bv = inp["fin"]["bv0"]
        for i in range(N):
            frac = stub if i == 0 else 1.0
            ni, charge = roe[i] * bv * frac, r * bv * frac
            er, dist = ni - charge, p * ni
            t = max(fp + i - 0.5 * frac if mid else fp + i, 0.0)
            df = (1 + r) ** -t
            rows.append({"label": lab(i), "roe": roe[i], "bvBegin": bv, "ni": ni, "charge": charge, "er": er,
                         "dist": dist, "bvEnd": bv + ni - dist, "frac": frac, "t": t, "df": df, "pv": er * df,
                         "epsAdj": ni / frac / sh, "bvps": (bv + ni - dist) / sh})
            pv_sum += er * df
            bv = bv + ni - dist
        persist = tm.get("er_persistence")
        persist = 1.0 if persist is None else persist
        roe_t = r + persist * (scen["roe"][1] - r)
        tv = (roe_t - r) * bv / (r - tm["g"])
        pv_tv = tv * (1 + r) ** -t_tv
        equity = inp["fin"]["bv0"] + pv_sum + pv_tv
        return {"model": "excess_return", "wacc": r, "ke": r, "rows": rows, "bv0": inp["fin"]["bv0"],
                "pvExplicit": pv_sum, "tv": tv, "pvTV": pv_tv, "roeTerminal": roe_t, "equity": equity,
                "perShare": equity / sh, "tvShare": pv_tv / equity, "justifiedPB": equity / inp["fin"]["bv0"],
                "bvps0": inp["fin"]["bv0"] / sh, "perShareGordon": equity / sh, "perShareExit": float("nan")}
    # DDM (per share): PV of dividends still to be paid + Gordon value of the year-10 dividend
    g = dps_growth_path(scen["dps_growth"], tm["g"])
    d = inp["fin"]["dps0"]
    for i in range(N):
        frac = min(1.0, max(0.0, fp)) if i == 0 else 1.0
        d = d * (1 + g[i])
        t = max(fp + i - 0.5 * frac if mid else fp + i, 0.0)
        df = (1 + r) ** -t
        rows.append({"label": lab(i), "g": g[i], "dps": d, "paid": d * frac, "frac": frac, "t": t, "df": df,
                     "pv": d * frac * df, "epsAdj": float("nan")})
        pv_sum += d * frac * df
    tv = d * (1 + tm["g"]) / (r - tm["g"])
    pv_tv = tv * (1 + r) ** -t_tv
    per = pv_sum + pv_tv
    return {"model": "ddm", "wacc": r, "ke": r, "rows": rows, "pvExplicit": pv_sum, "tv": tv, "pvTV": pv_tv,
            "equity": per * sh, "perShare": per, "tvShare": pv_tv / per, "impliedYield": inp["fin"]["dps0"] / per,
            "perShareGordon": per, "perShareExit": float("nan")}


# ------------------------------------------------------------------- knobs
def apply_knob(inp, scen, knob, x):
    I, S, ov = copy.deepcopy(inp), copy.deepcopy(scen), None
    if is_fin(inp):
        if knob == "roe_lr":
            S["roe"][1] = x
        elif knob == "roe_y1":
            S["roe"][0] = x
        elif knob == "roe_shift":
            S["roe"] = [v + x for v in S["roe"]]
        elif knob == "payout":
            S["payout"] = x
        elif knob == "persistence":
            I["terminal"]["er_persistence"] = x
        elif knob == "dps_shift":
            S["dps_growth"] = [v + x for v in S["dps_growth"]]
        elif knob == "dps_y1":
            S["dps_growth"][0] = x
        elif knob not in ("wacc", "g", "erp", "rm", "beta"):
            raise ValueError(f"knob {knob} does not apply to the {model_of(inp)} model")
        if knob in ("roe_lr", "roe_y1", "roe_shift", "payout", "persistence", "dps_shift", "dps_y1"):
            return I, S, ov
    alias = {"ai_shift": "seg_shift:" + key_seg(inp), "ai_fy28": "seg_fy1:" + key_seg(inp),
             "core_margin_lr": "margin_lr:" + key_margin(inp), "ai_margin_lr": "margin_lr:ai",
             "csg_margin_lr": "margin_lr:csg"}
    knob = alias.get(knob, knob)
    kind, _, arg = knob.partition(":")
    if kind == "seg_shift":
        S["growth"][arg] = [v + x for v in S["growth"][arg]]
        return I, S, ov
    if kind == "seg_fy1":
        S["growth"][arg][0] = x
        return I, S, ov
    if kind == "margin_lr":
        S["margins"][arg][1] = x
        return I, S, ov
    if knob == "wacc":
        ov = x
    elif knob == "g":
        I["terminal"]["g"] = x
    elif knob == "exit_multiple":
        I["terminal"]["exit_multiple"] = x
    elif knob == "all_growth_shift":
        for s in seg_keys(inp):
            S["growth"][s] = [v + x for v in S["growth"][s]]
    elif knob == "margin_shift":
        for k in S["margins"]:
            S["margins"][k][1] += x
    elif knob == "nwc_pct":
        I["operating"]["nwc_pct"] = x
    elif knob == "capex_pct":
        I["operating"]["capex_pct"] = x
    elif knob == "erp":
        if I["capital"].get("rm") is not None:
            I["capital"]["rm"] = I["capital"]["rf"] + x
        else:
            I["capital"]["erp"] = x
    elif knob == "rm":
        I["capital"]["rm"] = x
    elif knob == "beta":
        I["capital"]["beta_source"] = "custom"
        I["capital"]["beta_custom"] = x
    elif knob == "ronic":
        I["terminal"]["ronic"] = x
    else:
        raise ValueError(knob)
    return I, S, ov


def value_at(inp, scen, knob, x):
    I, S, ov = apply_knob(inp, scen, knob, x)
    return dcf(I, S, ov)["perShare"]


def reverse(inp, scen, knob, lo, hi, price=None):
    """Bisection to 1e-12, identical to the JS solver."""
    p = inp["market"]["price"] if price is None else price
    f = lambda x: value_at(inp, scen, knob, x) - p
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(hi - lo) < 1e-12:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def sensitivity(inp, scen, k1, v1, k2, v2):
    out = np.full((len(v1), len(v2)), np.nan)
    for i, a in enumerate(v1):
        for j, b in enumerate(v2):
            I, S, o1 = apply_knob(inp, scen, k1, a)
            I, S, o2 = apply_knob(I, S, k2, b)
            ov = o2 if o2 is not None else o1
            try:
                out[i, j] = dcf(I, S, ov)["perShare"]
            except ValueError:
                pass
    return pd.DataFrame(out, index=pd.Index(v1, name=k1), columns=pd.Index(v2, name=k2))


def monte_carlo(inp, scen, n=5000, seed=7, sd_ai=0.12, sd_core=0.02, sd_csg=0.01,
                sd_allg=0.02, wacc_lo=-0.01, wacc_hi=0.015, g_lo=0.02, g_hi=0.035):
    """Same shock design as the JS engine (numpy RNG, so draws differ)."""
    if is_fin(inp):
        return monte_carlo_fin(inp, scen, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    w0 = wacc(inp)["wacc"]
    ks, km, SEGS = key_seg(inp), key_margin(inp), seg_keys(inp)
    vals = []
    for _ in range(n):
        I, S = copy.deepcopy(inp), copy.deepcopy(scen)
        d_ai, d_core, d_csg, d_all = rng.normal(0, [sd_ai, sd_core, sd_csg, sd_allg])
        S["growth"][ks][0] += d_ai
        for s in SEGS:
            S["growth"][s] = [v + d_all for v in S["growth"][s]]
        for g in S["margins"]:
            S["margins"][g][1] += d_core if g == km else d_core / 2 if g == ks else d_csg
        w = rng.triangular(w0 + wacc_lo, w0, w0 + wacc_hi)
        I["terminal"]["g"] = rng.triangular(g_lo, inp["terminal"]["g"], g_hi)
        if w - I["terminal"]["g"] < 0.01:
            continue
        vals.append(dcf(I, S, w)["perShare"])
    v = np.array(vals)
    p = inp["market"]["price"]
    return {"values": v, "n": v.size, "mean": v.mean(), "median": np.median(v),
            **{f"p{q}": np.percentile(v, q) for q in (5, 10, 25, 75, 90, 95)},
            "probAbove": float((v > p).mean())}


def monte_carlo_fin(inp, scen, n=5000, seed=7, sd_roe_lr=0.015, sd_roe1=0.01, sd_dps=0.01,
                    ke_lo=-0.01, ke_hi=0.015, g_lo=0.015, g_hi=0.035):
    """Financials: shock long-run ROE (or dividend growth), year-1 ROE, the cost of equity and terminal g."""
    rng = np.random.default_rng(seed)
    k0, er = wacc(inp)["ke"], model_of(inp) == "excess_return"
    vals = []
    for _ in range(n):
        I, S = copy.deepcopy(inp), copy.deepcopy(scen)
        a, b = rng.normal(0, 1, 2)
        if er:
            S["roe"][1] += sd_roe_lr * a
            S["roe"][0] += sd_roe1 * b
        else:
            S["dps_growth"] = [v + sd_dps * a for v in S["dps_growth"]]
        r = rng.triangular(k0 + ke_lo, k0, k0 + ke_hi)
        I["terminal"]["g"] = rng.triangular(g_lo, min(max(inp["terminal"]["g"], g_lo), g_hi), g_hi)
        if r - I["terminal"]["g"] < 0.01:
            continue
        vals.append(fin_value(I, S, r)["perShare"])
    v = np.array(vals)
    p = inp["market"]["price"]
    return {"values": v, "n": v.size, "mean": v.mean(), "median": np.median(v),
            **{f"p{q}": np.percentile(v, q) for q in (5, 10, 25, 75, 90, 95)},
            "probAbove": float((v > p).mean())}


# ---------------------------------------------------------------- relative
def peer_frame(inp, tickers=None):
    cols = inp["peers"]["columns"]
    tks = tickers or inp["peers"]["core_set"]
    return pd.DataFrame({tk: inp["peers"]["rows"][tk] for tk in tks}, index=cols).T


def _stat(x, how):
    v = np.array([a for a in x if a is not None and np.isfinite(a) and a > 0], float)
    if v.size == 0:
        nan = float("nan")
        return {"k": nan, "n": 0, "q1": nan, "q3": nan, "min": nan, "max": nan}
    k = (v.size / (1 / v).sum() if how == "harmonic" else v.mean() if how == "mean"
         else np.median(v))
    return {"k": k, "n": v.size, "q1": np.percentile(v, 25), "q3": np.percentile(v, 75),
            "min": v.min(), "max": v.max()}


def relative(inp, tickers=None, how=None, weights=None):
    how = how or inp["relative"]["stat"]
    weights = weights or inp["relative"]["weights"]
    P = peer_frame(inp, tickers)
    m, sh = inp["market"], diluted_shares(inp)
    td = m.get("total_debt_incl_dfs")
    nd = td + inp["bridge"]["leases"] - inp["bridge"]["cash"] if _ok(td) else float("nan")
    ev = m["price"] * sh + nd
    nan = float("nan")
    fin = inp.get("fin") or {}
    bvps = fin["bv0"] / sh if _ok(fin.get("bv0")) else None
    dps0 = fin.get("dps0")
    all_ = {
        "pe_fwd": (m.get("eps_ntm"), lambda k: k * m["eps_ntm"]),
        "pb": (bvps, lambda k: k * bvps),
        "div_yield": (dps0, lambda k: dps0 / k),
        "ev_ebitda": (m.get("ebitda_ttm"), lambda k: (k * m["ebitda_ttm"] - nd) / sh),
        "ev_sales": (m.get("revenue_ttm"), lambda k: (k * m["revenue_ttm"] - nd) / sh),
    }
    # only the multiples that carry a weight for this company (and exist in the peer table) are computed
    spec = {k: v for k, v in all_.items() if weights.get(k, 0) > 0 and k in inp["peers"]["columns"]}
    own_of = {"pe_fwd": lambda mt: m["price"] / mt, "pb": lambda mt: m["price"] / mt,
              "div_yield": lambda mt: mt / m["price"]}
    table, acc, ws = {}, 0.0, 0.0
    for key, (metric, to_price) in spec.items():
        st = _stat(P[key].tolist(), how)
        # a multiple is used only if the company's own metric is positive and peers give a finite multiple
        usable = _ok(metric) and metric > 0 and np.isfinite(st["k"]) and (not key.startswith("ev_") or _ok(nd))
        own = (own_of[key](metric) if key in own_of else ev / metric) if usable else nan
        st.update(own=own, premium=own / st["k"] - 1 if usable else nan, implied=to_price(st["k"]) if usable else nan,
                  impliedQ1=to_price(st["q1"]) if usable else nan, impliedQ3=to_price(st["q3"]) if usable else nan,
                  weight=weights.get(key, 0.0) if usable else 0.0, usable=usable)
        table[key] = st
        if st["weight"] > 0:
            acc += st["weight"] * st["implied"]
            ws += st["weight"]
    return {"peers": P, "table": pd.DataFrame(table).T, "blended": acc / ws if ws > 0 else nan,
            "netDebtAll": nd, "evDell": ev}


def reg_min_r2(inp):
    m = (inp.get("relative") or {}).get("reg_min_r2")
    return 0.3 if m is None else m


def _reg_gate(inp, r2, value):
    """A peer regression that explains little of the peer spread is reported but not used in the blend."""
    weak = not (r2 >= reg_min_r2(inp))
    return {"impliedRaw": value, "implied": float("nan") if weak else value, "weak": weak, "minR2": reg_min_r2(inp)}


def regression_pb(inp, tickers=None):
    """Banks: cross-sectional OLS of P/B on ROE; implied price = fitted P/B x own book per share."""
    P = peer_frame(inp, tickers or list(inp["peers"]["rows"]))
    ok = [(float(a), float(b)) for a, b in zip(P["roe"], P["pb"]) if _ok(a) and a > 0 and _ok(b) and b > 0]
    roe, bvps = inp["fin"]["roe_ttm"], inp["fin"]["bv0"] / diluted_shares(inp)
    nan = float("nan")
    if len(ok) < 3:
        return {"a": nan, "b": nan, "r2": nan, "n": len(ok), "dellGrowth": roe, "pe": nan, "implied": nan, "kind": "pb_roe"}
    x, y = np.array([a for a, _ in ok]), np.array([b for _, b in ok])
    mx, my = x.mean(), y.mean()
    sxy, sxx, syy = ((x - mx) * (y - my)).sum(), ((x - mx) ** 2).sum(), ((y - my) ** 2).sum()
    b = sxy / sxx
    a = my - b * mx
    pb = a + b * roe
    return {"a": a, "b": b, "r2": sxy * sxy / (sxx * syy), "n": x.size, "dellGrowth": roe, "pe": pb,
            **_reg_gate(inp, sxy * sxy / (sxx * syy), pb * bvps), "kind": "pb_roe"}


def regression(inp, dell_growth, tickers=None):
    if model_of(inp) == "excess_return":
        return regression_pb(inp, tickers)
    P = peer_frame(inp, tickers or list(inp["peers"]["rows"]))
    P = P[(P["eps_g3y"].astype(float) > 0) & (P["pe_fwd"].astype(float) > 0)]
    x, y = P["eps_g3y"].astype(float).values, P["pe_fwd"].astype(float).values
    b, a = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    pe = a + b * dell_growth
    return {"a": a, "b": b, "r2": r2, "n": x.size, "dellGrowth": dell_growth,
            "pe": pe, **_reg_gate(inp, r2, pe * inp["market"]["eps_ntm"])}


# ----------------------------------------------------------------- verdict
def _ok(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def _get(o, path):
    for k in path.split("."):
        if o is None:
            return None
        o = o[int(k)] if isinstance(o, list) else o.get(k)
    return o


def check_inputs(inp):
    """Fail-closed checks. A blocking item stops the call; a warning is shown next to it."""
    blocking, warnings = [], []

    def need(path, label, test=None, why=""):
        v = _get(inp, path)
        if not _ok(v):
            blocking.append({"path": path, "msg": f"{label} is missing"})
        elif test and not test(v):
            blocking.append({"path": path, "msg": f"{label} {why}"})

    need("market.price", "Share price", lambda v: v > 0, "must be above zero")
    need("market.shares_basic", "Shares outstanding", lambda v: v > 0, "must be above zero")
    need("market.dilution", "Dilution", lambda v: v >= 0, "cannot be negative")
    fin, mdl = is_fin(inp), model_of(inp)
    if mdl not in ("fcff", "excess_return", "ddm"):
        blocking.append({"path": "model", "msg": f'Unknown valuation model "{mdl}"'})
    if not fin:
        need("rev_y0", "Last fiscal-year revenue", lambda v: v > 0, "must be above zero")
        for k in seg_keys(inp):
            need(f"y1_segments.{k}", f"Next-year revenue ({k})", lambda v: v >= 0, "cannot be negative")
        need("bridge.cash", "Cash", lambda v: v >= 0, "cannot be negative")
        need("bridge.core_debt", "Debt", lambda v: v >= 0, "cannot be negative")
        for k in ["leases", "investments", "nci", "dfs_receivables", "dfs_lease_equipment", "dfs_debt"]:
            if k in inp["bridge"]:
                need(f"bridge.{k}", f"Bridge item {k}")
    elif mdl == "excess_return":
        need("fin.bv0", "Common book equity", lambda v: v > 0, "must be above zero (negative book breaks an excess-return model)")
        need("fin.payout", "Payout ratio (dividends + buybacks)", lambda v: 0 <= v <= 1.5, "is outside 0–150%")
        if inp["terminal"].get("er_persistence") is not None:
            need("terminal.er_persistence", "Excess-return persistence", lambda v: 0 <= v <= 1, "must be between 0 and 1")
    elif mdl == "ddm":
        need("fin.dps0", "Dividend per share", lambda v: v > 0, "must be above zero for a dividend model")
    c = inp["capital"]
    need("capital.rf", "Risk-free rate (10Y yield)", lambda v: -0.01 < v < 0.2, "is outside −1% to 20%")
    if c.get("rm") is not None:
        need("capital.rm", "Expected market return", lambda v: _ok(c["rf"]) and v > c["rf"], "must exceed the risk-free rate")
    else:
        need("capital.erp", "Equity risk premium", lambda v: v > 0, "must be above zero")
    need("capital.kd_pretax", "Pre-tax cost of debt", lambda v: 0 <= v < 0.3, "is outside 0–30%")
    need("capital.tax_marginal", "Marginal tax rate", lambda v: 0 <= v < 0.6, "is outside 0–60%")
    if not fin:
        for k in ["tax0", "tax_lr", "da_pct", "capex_pct", "nwc_pct"]:
            need(f"operating.{k}", f"Operating input {k}")
    need("terminal.g", "Terminal growth", lambda v: v < 0.06, "is above 6%, faster than the economy forever")
    parts = (["roe"] if mdl == "excess_return" else ["dps_growth"]) if fin else ["growth", "margins"]
    for sn, sc in inp["scenarios"].items():
        for part in parts:
            val = sc.get(part)
            if val is None or isinstance(val, list):
                arr = val or []
                if not arr:
                    blocking.append({"path": f"scenarios.{sn}.{part}", "msg": f"{sn} {part} is missing"})
                for i, v in enumerate(arr):
                    if not _ok(v):
                        blocking.append({"path": f"scenarios.{sn}.{part}.{i}", "msg": f"{sn} {part} (item {i + 1}) is missing"})
            else:
                for g, arr in val.items():
                    for i, v in enumerate(arr):
                        if not _ok(v):
                            blocking.append({"path": f"scenarios.{sn}.{part}.{g}.{i}", "msg": f"{sn} {part} ({g}, item {i + 1}) is missing"})
    meta = inp.get("meta") or {}
    if meta.get("financial_currency") and meta.get("currency") and meta["financial_currency"] != meta["currency"]:
        blocking.append({"path": "meta.financial_currency", "msg": f"Statements are in {meta['financial_currency']} but the price is in {meta['currency']}. Convert the statements or value the local listing."})
    dq = inp.get("data_quality") or {}
    blocking += list(dq.get("blocking") or [])
    warnings += list(dq.get("warnings") or [])
    if not blocking:
        b = beta(inp)
        if not _ok(b["chosen"]) or b["chosen"] < -1 or b["chosen"] > 5:
            blocking.append({"path": "capital.beta_source", "msg": "Beta is missing or outside −1 to 5"})
        src = c.get("beta_source")
        want_daily, want_crsp = src in ("daily", "daily_adj"), src in ("crsp_monthly", "crsp_monthly_adj")
        if (want_daily and c.get("beta_daily") is None) or (want_crsp and c.get("beta_crsp_monthly") is None):
            warnings.append({"path": "capital.beta_source", "msg": f'Beta source "{src}" has no data; the Blume-adjusted feed beta is used instead'})
        st = c.get("beta_daily_stats")
        if want_daily and st and _ok(st.get("n")) and st["n"] < 250:
            warnings.append({"path": "capital.beta_daily", "msg": f"Daily beta uses only {st['n']} days"})
        if want_daily and st and _ok(st.get("r2")) and st["r2"] < 0.05:
            warnings.append({"path": "capital.beta_daily", "msg": f"Daily beta R² is {st['r2']:.3f}; the market explains almost none of the stock's moves"})
        W = wacc(inp)
        if fin:
            if not (W["ke"] > inp["terminal"]["g"] + 0.005):
                blocking.append({"path": "terminal.g", "msg": f"Cost of equity ({W['ke'] * 100:.2f}%) must exceed terminal growth by at least 0.5pp"})
        else:
            if inp["terminal"].get("method") != "exit" and not (W["wacc"] > inp["terminal"]["g"] + 0.005):
                blocking.append({"path": "terminal.g", "msg": f"WACC ({W['wacc'] * 100:.2f}%) must exceed terminal growth by at least 0.5pp"})
            if _ok(inp["terminal"].get("ronic")) and inp["terminal"]["ronic"] <= inp["terminal"]["g"]:
                warnings.append({"path": "terminal.ronic", "msg": "RONIC at or below terminal growth makes growth destroy value"})
    if meta.get("balance_sheet_date") and meta.get("valuation_date"):
        age = (pd.Timestamp(meta["valuation_date"]) - pd.Timestamp(meta["balance_sheet_date"])).days
        if age > 200:
            warnings.append({"path": "meta.balance_sheet_date", "msg": f"Balance sheet is {age} days old"})
    return {"blocking": blocking, "warnings": warnings, "ok": not blocking}


# ---------------------------------------------------------- earnings quality
# Beneish (1999) 8-variable M-score; coefficients from the unweighted probit.
BENEISH = {"c": -4.84, "DSRI": 0.92, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892, "DEPI": 0.115,
           "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327}
BENEISH_MANIP = {"DSRI": 1.465, "GMI": 1.193, "AQI": 1.254, "SGI": 1.607, "DEPI": 1.077, "SGAI": 1.041,
                 "LVGI": 1.111, "TATA": 0.031}
EQ_LIMITS = {"mRed": -1.78, "mAmber": -2.22, "accrRed": 0.10, "accrAmber": 0.05, "ccRed": 0.5, "ccAmber": 0.8}


def beneish_year(q, t):
    def at(k, i):
        a = q.get(k)
        return a[i] if a is not None and i < len(a) and _ok(a[i]) else None

    def ratio(f):
        try:
            x = f()
        except (TypeError, ZeroDivisionError):
            return None
        return x if _ok(x) else None

    s_ = lambda i: at("sales", i)
    idx = {
        "DSRI": ratio(lambda: (at("rec", t) / s_(t)) / (at("rec", t - 1) / s_(t - 1))),
        "GMI": ratio(lambda: ((s_(t - 1) - at("cogs", t - 1)) / s_(t - 1)) / ((s_(t) - at("cogs", t)) / s_(t))),
        "AQI": ratio(lambda: (1 - (at("ca", t) + at("ppe", t)) / at("ta", t)) / (1 - (at("ca", t - 1) + at("ppe", t - 1)) / at("ta", t - 1))),
        "SGI": ratio(lambda: s_(t) / s_(t - 1)),
        "DEPI": ratio(lambda: (at("dep", t - 1) / (at("dep", t - 1) + at("ppe", t - 1))) / (at("dep", t) / (at("dep", t) + at("ppe", t)))),
        "SGAI": ratio(lambda: (at("sga", t) / s_(t)) / (at("sga", t - 1) / s_(t - 1))),
        "LVGI": ratio(lambda: ((at("cl", t) + at("ltd", t)) / at("ta", t)) / ((at("cl", t - 1) + at("ltd", t - 1)) / at("ta", t - 1))),
        "TATA": ratio(lambda: (at("ni", t) - at("cfo", t)) / at("ta", t)),
    }
    neutral = [k for k in idx if k != "TATA" and idx[k] is None]
    used = {k: (1.0 if k in neutral else v) for k, v in idx.items()}
    if idx["TATA"] is None or len(neutral) > 2:
        m = float("nan")
    else:
        m = BENEISH["c"]
        for k in used:
            m += BENEISH[k] * used[k]
    return {"label": q["labels"][t], "idx": idx, "neutral": neutral, "m": m}


def earnings_quality(inp):
    q = inp.get("quality_data")
    if model_of(inp) == "excess_return":
        return {"grade": "n/a", "reason": "Not meaningful for banks and insurers: no cost of sales, and operating cash flow includes lending and deposit flows", "years": [], "flags": []}
    if not q or not q.get("labels") or len(q["labels"]) < 2:
        return {"grade": "n/a", "reason": "No statement history for an earnings-quality check", "years": [], "flags": []}
    n = len(q["labels"])
    years = [beneish_year(q, t) for t in range(1, n)]
    last = years[-1]

    def v(k, i):
        a = q.get(k)
        return a[i] if a is not None and _ok(a[i]) else None

    nan = float("nan")
    accruals = ((v("ni", n - 1) - v("cfo", n - 1)) / ((v("ta", n - 1) + v("ta", n - 2)) / 2)
                if None not in (v("ni", n - 1), v("cfo", n - 1), v("ta", n - 1), v("ta", n - 2)) else nan)
    k = min(3, n)

    def sum_of(key):
        xs = [v(key, i) for i in range(n - k, n)]
        return nan if None in xs else sum(xs)

    s_ni, s_cfo = sum_of("ni"), sum_of("cfo")
    cash_conv = s_cfo / s_ni if s_ni > 0 else nan
    L, flags = EQ_LIMITS, []
    if np.isfinite(last["m"]):
        if last["m"] > L["mRed"]:
            flags.append({"level": "red", "key": "m", "msg": f"Beneish M-score {last['m']:.2f} is above −1.78, the level Beneish found for likely manipulators"})
        elif last["m"] > L["mAmber"]:
            flags.append({"level": "amber", "key": "m", "msg": f"Beneish M-score {last['m']:.2f} is between −2.22 and −1.78 (grey zone)"})
    if last["neutral"]:
        flags.append({"level": "info", "key": "neutral", "msg": f"M-score inputs missing ({', '.join(last['neutral'])}): set to 1, which is neutral"})
    if np.isfinite(accruals):
        if accruals > L["accrRed"]:
            flags.append({"level": "red", "key": "accruals", "msg": f"Accruals are {accruals * 100:.1f}% of assets: earnings run well ahead of cash"})
        elif accruals > L["accrAmber"]:
            flags.append({"level": "amber", "key": "accruals", "msg": f"Accruals are {accruals * 100:.1f}% of assets"})
    if np.isfinite(cash_conv):
        if cash_conv < L["ccRed"]:
            flags.append({"level": "red", "key": "cash", "msg": f"Operating cash flow is only {cash_conv * 100:.0f}% of net income over {k} years"})
        elif cash_conv < L["ccAmber"]:
            flags.append({"level": "amber", "key": "cash", "msg": f"Operating cash flow is {cash_conv * 100:.0f}% of net income over {k} years"})
    reds = sum(f["level"] == "red" for f in flags)
    ambers = sum(f["level"] == "amber" for f in flags)
    tested = np.isfinite(last["m"]) or np.isfinite(accruals) or np.isfinite(cash_conv)
    grade = "n/a" if not tested else "D" if reds >= 2 else "C" if reds == 1 or ambers >= 2 else "B" if ambers == 1 else "A"
    return {"grade": grade, "years": years, "latest": last, "accruals": accruals, "cashConv": cash_conv,
            "cashYears": k, "flags": flags, "limits": L, "coef": BENEISH, "manip": BENEISH_MANIP}


def verdict(inp, parts):
    Q = check_inputs(inp)
    w, band, p = inp["verdict"]["weights"], inp["verdict"]["band"], inp["market"]["price"]
    live = [k for k in w if w[k] > 0]
    dropped = [k for k in live if not (parts.get(k) is not None and np.isfinite(parts[k]))]
    warnings = Q["warnings"] + [{"path": f"verdict.weights.{k}", "msg": f"{k} gave no value and is left out of the blend"} for k in dropped]
    if not Q["ok"] or len(dropped) == len(live):
        nan = float("nan")
        return {"fair": nan, "upside": nan, "call": "NOT VALUED", "expectedReturn": nan, "requiredReturn": nan,
                "alpha": nan, "divYield": nan, "warnings": warnings, "dropped": dropped,
                "blocking": Q["blocking"] if not Q["ok"] else [{"path": "verdict.weights", "msg": "No valuation method produced a value"}]}
    use = {k: v for k, v in parts.items() if k in w and v is not None and np.isfinite(v)}
    # earnings quality gates the call: D withholds it, C marks it low confidence (switch off with verdict.quality_gate = false)
    EQ, gate = earnings_quality(inp), inp["verdict"].get("quality_gate") is not False
    if EQ["grade"] in ("C", "D"):
        warnings.append({"path": "quality_data", "msg": f"Earnings quality {EQ['grade']}: " + "; ".join(f["msg"] for f in EQ["flags"] if f["level"] != "info")})
    fv = sum(w[k] * use[k] for k in use) / sum(w[k] for k in use)
    up = fv / p - 1
    dy = inp["market"]["dividend_ps"] / p
    er = (fv / p) ** (1 / inp["verdict"]["horizon_years"]) - 1 + dy
    ke = wacc(inp)["ke"]
    raw = "UNDERVALUED" if up > band else "OVERVALUED" if up < -band else "FAIRLY VALUED"
    gated = gate and EQ["grade"] == "D"
    return {"fair": fv, "upside": up, "call": "NO CALL" if gated else raw, "rawCall": raw, "gated": gated,
            "confidence": "low" if gate and EQ["grade"] == "C" else "normal",
            "quality": {"grade": EQ["grade"], "flags": EQ["flags"]}, "expectedReturn": er,
            "requiredReturn": ke, "alpha": er - ke, "divYield": dy, "blocking": [], "warnings": warnings, "dropped": dropped}


def pvgo(inp):
    ke, e1, p = wacc(inp)["ke"], inp["market"]["eps_ntm"], inp["market"]["price"]
    return {"noGrowth": e1 / ke, "pvgo": p - e1 / ke, "share": (p - e1 / ke) / p}


def run_all(inp, scen_name="Base", mc_n=5000):
    Q = check_inputs(inp)
    if not Q["ok"]:
        return {"checks": Q, "verdict": verdict(inp, {})}
    scen = inp["scenarios"][scen_name]
    base = dcf(inp, scen)
    rows = base["rows"]
    sg = inp.get("relative", {}).get("subject_growth")
    growth = sg if sg is not None and np.isfinite(sg) else (rows[3]["epsAdj"] / rows[0]["epsAdj"]) ** (1 / 3) - 1
    rel = relative(inp)
    reg = {"implied": float("nan"), "n": 0} if model_of(inp) == "ddm" else regression(inp, growth)
    mc = monte_carlo(inp, scen, n=mc_n)
    v = verdict(inp, {"dcf": base["perShare"], "mc": mc["median"],
                      "comps": rel["blended"], "regression": reg["implied"]})
    if reg.get("weak") and np.isfinite(reg.get("r2", float("nan"))):
        v["warnings"] = [w for w in v["warnings"] if w["path"] != "verdict.weights.regression"]
        v["warnings"].append({"path": "relative.reg_min_r2", "msg": f"Peer regression R² {reg['r2']:.2f} is below {reg['minR2']:.2f}; it is shown but left out of the blend"})
    return {"checks": Q, "W": wacc(inp), "base": base,
            "scen3": {k: dcf(inp, s) for k, s in inp["scenarios"].items()},
            "rel": rel, "reg": reg, "mc": mc, "verdict": v, "pvgo": pvgo(inp)}


def to_excel(inp, R, path):
    with pd.ExcelWriter(path) as xw:
        v, W = R["verdict"], R["W"]
        pd.Series({"Price": inp["market"]["price"], "Fair value": v["fair"],
                   "Upside": v["upside"], "Call": v["call"],
                   "Expected 1y return": v["expectedReturn"], "Required (Ke)": v["requiredReturn"],
                   "Expected alpha": v["alpha"], "WACC": W["wacc"], "Ke": W["ke"],
                   "Beta used": W["beta"]["chosen"], "DCF base": R["base"]["perShare"],
                   "MC median": R["mc"]["median"], "P(value>price)": R["mc"]["probAbove"],
                   "Comps blend": R["rel"]["blended"], "Regression P/E value": R["reg"]["implied"]}
                  ).to_frame("value").to_excel(xw, sheet_name="Summary")
        for k, d in R["scen3"].items():
            flat = [{**{kk: vv for kk, vv in r.items() if not isinstance(vv, dict)}, **r["seg"]} for r in d["rows"]]
            pd.DataFrame(flat).set_index("label").T.to_excel(xw, sheet_name=f"DCF {k}")
        R["rel"]["table"].to_excel(xw, sheet_name="Comps")
        R["rel"]["peers"].to_excel(xw, sheet_name="Peers")
        scen = inp["scenarios"]["Base"]
        w0 = W["wacc"]
        sensitivity(inp, scen, "wacc", [round(w0 + d, 4) for d in (-.02, -.01, 0, .01, .02)],
                    "g", [.02, .025, .03, .035, .04]).to_excel(xw, sheet_name="Sens WACC x g")
        pd.DataFrame(json.loads(json.dumps(inp["sources"])), index=["source"]).T.to_excel(
            xw, sheet_name="Sources")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dell_inputs.json"
    inp = json.load(open(path))
    R = run_all(inp)
    if not R["checks"]["ok"]:
        print(f"{inp['meta']['ticker']}  NOT VALUED — fix these inputs first:")
        for b in R["checks"]["blocking"]:
            print(f"  [{b['path']}] {b['msg']}")
        sys.exit(2)
    for w in R["verdict"]["warnings"]:
        print(f"  warning [{w['path']}] {w['msg']}")
    W, v = R["W"], R["verdict"]
    print(f"{inp['meta']['ticker']}  price {inp['market']['price']:.2f}")
    print(f"WACC {W['wacc']:.4%}  Ke {W['ke']:.4%}  beta {W['beta']['chosen']:.3f}")
    for k, d in R["scen3"].items():
        print(f"  DCF {k:<5} {d['perShare']:8.2f}   (exit-multiple check {d['perShareExit']:.2f})")
    print(f"  Monte Carlo median {R['mc']['median']:.2f}  P(>price) {R['mc']['probAbove']:.1%}")
    print(f"  Comps blend {R['rel']['blended']:.2f}   P/E-growth regression {R['reg']['implied']:.2f}")
    print(f"VERDICT {v['call']}  fair {v['fair']:.2f}  upside {v['upside']:.1%}  "
          f"expected alpha {v['alpha']:.1%}")
    to_excel(inp, R, f"{inp['meta']['ticker']}_valuation_v2.xlsx")
