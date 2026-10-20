/* research.js — turn Yahoo Finance's raw JSON feeds into a terminal inputs file.
 * Pure functions (no DOM, no network); the page fetches, this builds.
 * Every input written gets a provenance tag in inp.provenance[path] =
 *   [kind, source]   kind: "sourced" (read from a feed), "derived" (computed
 *   only from sourced numbers, formula given), "assumption" (a modelling choice).
 */
(function (root) {
  "use strict";
  const Y = {
    crumb: "https://query1.finance.yahoo.com/v1/test/getcrumb",
    modules:
      "price,summaryDetail,defaultKeyStatistics,financialData,earningsTrend,recommendationTrend,upgradeDowngradeHistory,assetProfile",
    summary(t, crumb) {
      return `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(t)}?modules=${this.modules}&crumb=${encodeURIComponent(crumb)}`;
    },
    peerSummary(t, crumb) {
      return `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(t)}?modules=price,summaryDetail,defaultKeyStatistics,financialData,earningsTrend&crumb=${encodeURIComponent(crumb)}`;
    },
    TYPES: [
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
    ],
    timeseries(t, period) {
      return `https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/${encodeURIComponent(t)}?type=${this.TYPES.map((x) => period + x).join(",")}&period1=1420070400&period2=1900000000`;
    },
    chart(t, range) {
      return `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(t)}?range=${range || "2y"}&interval=1d`;
    },
    recs(t) {
      return `https://query2.finance.yahoo.com/v6/finance/recommendationsbysymbol/${encodeURIComponent(t)}`;
    },
  };

  // Damodaran synthetic rating, large non-financial firms (interest coverage → rating, spread)
  const RATINGS = [
    [8.5, "AAA", 0.0059],
    [6.5, "AA", 0.007],
    [5.5, "A+", 0.0092],
    [4.25, "A", 0.0107],
    [3.0, "A-", 0.0121],
    [2.5, "BBB", 0.0147],
    [2.25, "BB+", 0.0193],
    [2.0, "BB", 0.0236],
    [1.75, "B+", 0.0285],
    [1.5, "B", 0.0347],
    [1.25, "B-", 0.0437],
    [0.8, "CCC", 0.0811],
    [0.65, "CC", 0.1],
    [0.2, "C", 0.14],
    [-1e9, "D", 0.19],
  ];
  const synthetic = (cov) => RATINGS.find(([lo]) => cov >= lo);

  const raw = (o) => (o && typeof o === "object" && "raw" in o ? o.raw : typeof o === "number" ? o : null);
  const num = (x) => (typeof x === "number" && isFinite(x) ? x : null);

  // ---- parse the browser's get_page_text output (header, "---", body) ----
  function bodyOf(text) {
    let b = String(text);
    const i = b.indexOf("\n---\n");
    if (i >= 0) b = b.slice(i + 5);
    const j = b.indexOf("\n\nTab Context:");
    if (j >= 0) b = b.slice(0, j);
    return b.trim();
  }
  function jsonOf(text) {
    const b = bodyOf(text),
      s = b.indexOf("{"),
      e = b.lastIndexOf("}");
    if (s < 0 || e < s) throw new Error("no JSON in page");
    return JSON.parse(b.slice(s, e + 1));
  }

  // ---- feed shapes → plain series ----
  function series(tsResult) {
    const out = {};
    for (const x of tsResult || []) {
      const t = x.meta && x.meta.type && x.meta.type[0];
      if (!t) continue;
      out[t.replace(/^(annual|quarterly)/, "")] = (x[t] || [])
        .filter(Boolean)
        .map((e) => [e.asOfDate, e.reportedValue ? e.reportedValue.raw : null])
        .filter((e) => e[1] != null)
        .sort((a, b) => (a[0] < b[0] ? -1 : 1));
    }
    return out;
  }
  function closes(chartResult) {
    const r = chartResult,
      adj =
        r.indicators.adjclose && r.indicators.adjclose[0]
          ? r.indicators.adjclose[0].adjclose
          : r.indicators.quote[0].close;
    return r.timestamp
      .map((ts, i) => [new Date(ts * 1000).toISOString().slice(0, 10), adj[i]])
      .filter((x) => x[1] != null);
  }
  const last = (s) => (s && s.length ? s[s.length - 1][1] : null);
  const lastDate = (s) => (s && s.length ? s[s.length - 1][0] : null);
  const at = (s, k) => (s && s.length >= k ? s[s.length - k][1] : null);
  const sumLast = (s, k) => (s && s.length >= k ? s.slice(-k).reduce((a, x) => a + x[1], 0) : null);
  const fy = (d) => "FY" + String(new Date(d).getUTCFullYear() % 100).padStart(2, "0");

  /**
   * Build an inputs object from parsed feeds.
   * f = { ticker, summary: quoteSummary.result[0], annual: series(), quarterly: series(),
   *       stock: closes(), index: closes(), tnx: number (percent), peers: {T: quoteSummary.result[0]},
   *       today: "YYYY-MM-DD", erp: decimal }
   */
  function build(f, M) {
    const S = f.summary,
      pr = S.price || {},
      sd = S.summaryDetail || {},
      ks = S.defaultKeyStatistics || {},
      fd = S.financialData || {},
      A = f.annual,
      Q = f.quarterly;
    const P = {}; // provenance
    const src = (path, kind, s) => {
      P[path] = [kind, s];
    };
    const YQ = "Yahoo Finance quoteSummary",
      YT = "Yahoo Finance fundamentals (annual)",
      YTQ = "Yahoo Finance fundamentals (quarterly)";
    const today = f.today || new Date().toISOString().slice(0, 10);
    const M_ = 1e6;
    // data-quality log: blocking items stop the call until fixed; warnings mark silent fallbacks
    const DQ = { blocking: [], warnings: [] };
    // model routing by Yahoo industry: banks/insurers → excess-return, REITs → dividend discount, else FCFF
    const prof = S.assetProfile || {},
      ind = prof.industry || "";
    const lev =
      last(Q.TotalAssets) && last(Q.CommonStockEquity)
        ? last(Q.TotalAssets) / last(Q.CommonStockEquity)
        : null;
    const MODEL =
      f.model ||
      (/^REIT/.test(ind)
        ? "ddm"
        : /^(Banks|Insurance - |Capital Markets|Mortgage Finance|Financial Conglomerates)/.test(ind) ||
            (ind === "Credit Services" && lev != null && lev > 6)
          ? "excess_return"
          : "fcff");
    const FIN = MODEL !== "fcff";
    // FCFF-only fallbacks are not reported for financials (those inputs are not used)
    const fcffOnly = /^(y1_segments|scenarios\.|operating\.|capital\.kd_pretax|bridge\.)/;
    const warn = (path, msg) => {
      if (!(FIN && fcffOnly.test(path))) DQ.warnings.push({ path, msg });
    };
    const block = (path, msg) => DQ.blocking.push({ path, msg });
    const num = (x) => (typeof x === "number" && Number.isFinite(x) ? x : null);

    const price = raw(pr.regularMarketPrice);
    const mcap = raw(pr.marketCap);
    let shares = raw(ks.impliedSharesOutstanding) || raw(ks.sharesOutstanding);
    if (mcap && price && (!shares || Math.abs((shares * price) / mcap - 1) > 0.05)) shares = mcap / price;
    shares = shares ? shares / M_ : null;
    const qDil = last(Q.DilutedAverageShares),
      qOrd = last(Q.OrdinarySharesNumber);
    const dilution = qDil && qOrd && qDil > qOrd ? (qDil - qOrd) / M_ : 0;

    const rev0 = num(last(A.TotalRevenue)) != null ? last(A.TotalRevenue) / M_ : null;
    const lastFYE = lastDate(A.TotalRevenue);
    let nextFYE = raw(ks.nextFiscalYearEnd)
      ? new Date(raw(ks.nextFiscalYearEnd) * 1000).toISOString().slice(0, 10)
      : null;
    if (!nextFYE || nextFYE <= today) {
      const d = new Date(lastFYE);
      d.setUTCFullYear(d.getUTCFullYear() + 1);
      nextFYE = d.toISOString().slice(0, 10);
    }
    while (nextFYE <= today) {
      const d = new Date(nextFYE);
      d.setUTCFullYear(d.getUTCFullYear() + 1);
      nextFYE = d.toISOString().slice(0, 10);
    }
    const y0 = new Date(nextFYE).getUTCFullYear();
    const labels = Array.from({ length: 10 }, (_, i) => "FY" + String((y0 + i) % 100).padStart(2, "0"));
    const bsDate = lastDate(Q.TotalDebt) || lastDate(Q.CashAndCashEquivalents) || lastFYE;
    const days = (a, b) => (new Date(b) - new Date(a)) / 86400000;
    const firstPeriod = days(today, nextFYE) / 365.25;
    const stub = Math.min(1, Math.max(0.02, days(bsDate, nextFYE) / 365.25));

    // consensus
    const tr = (S.earningsTrend && S.earningsTrend.trend) || [];
    const T0 = tr.find((t) => t.period === "0y") || {},
      T1 = tr.find((t) => t.period === "+1y") || {};
    const revY1 = raw(T0.revenueEstimate && T0.revenueEstimate.avg),
      revY2 = raw(T1.revenueEstimate && T1.revenueEstimate.avg);
    const epsY1 = raw(T0.earningsEstimate && T0.earningsEstimate.avg),
      epsY2avg = raw(T1.earningsEstimate && T1.earningsEstimate.avg);
    const g1eps = raw(T1.growth);
    // Yahoo's +1y EPS average can lag its own growth field; keep both and flag the gap
    const epsY2fromG = epsY1 != null && g1eps != null ? epsY1 * (1 + g1eps) : null;
    const epsConflict = epsY2avg != null && epsY2fromG != null && Math.abs(epsY2avg / epsY2fromG - 1) > 0.03;
    const histG =
      A.TotalRevenue && A.TotalRevenue.length >= 2 ? at(A.TotalRevenue, 1) / at(A.TotalRevenue, 2) - 1 : null;
    if (!revY1)
      warn(
        "y1_segments.total",
        histG != null
          ? "No consensus revenue; next year uses last year's growth"
          : "No consensus revenue and under two years of history",
      );
    const y1 = revY1 ? revY1 / M_ : rev0 != null && histG != null ? rev0 * (1 + histG) : null;
    const y2 = revY2 ? revY2 / M_ : y1 != null && histG != null ? y1 * (1 + Math.min(histG, 0.1)) : null;
    if (revY1 && !revY2)
      warn(
        "scenarios.Base.growth.total.0",
        "No consensus for the second year; growth uses history capped at 10%",
      );
    const gNext = y2 / y1 - 1;

    // margins
    const ttmRev = sumLast(Q.TotalRevenue, 4),
      ttmOI = sumLast(Q.OperatingIncome, 4);
    let mTTM =
      ttmRev && ttmOI != null
        ? ttmOI / ttmRev
        : last(A.OperatingIncome) != null && last(A.TotalRevenue)
          ? last(A.OperatingIncome) / last(A.TotalRevenue)
          : null;
    if (!(ttmRev && ttmOI != null))
      warn(
        "scenarios.Base.margins.total.0",
        "Fewer than 4 quarters of operating income; the last annual margin is used",
      );
    if (mTTM != null && mTTM < 0)
      warn(
        "scenarios.Base.margins.total.0",
        "Operating margin is negative; an FCFF DCF on today's margin is unreliable",
      );
    const hist = (A.TotalRevenue || [])
      .map(([d, r]) => {
        const oi = (A.OperatingIncome || []).find((x) => x[0] === d);
        return oi ? oi[1] / r : null;
      })
      .filter((x) => x != null);
    const mAvg = hist.length ? hist.reduce((a, b) => a + b, 0) / hist.length : mTTM;
    const mMin = hist.length ? Math.min(...hist) : mTTM != null ? mTTM * 0.8 : null;
    const baseLR = mTTM != null && mAvg != null ? (mTTM + mAvg) / 2 : null;
    // a perpetual loss values a going concern below zero; when the worst year is a loss, Bear fades to the through-cycle average instead
    const bearLR = mMin != null && mMin <= 0 && mAvg != null && mAvg > 0 ? mAvg : mMin;
    if (bearLR !== mMin)
      warn(
        "scenarios.Bear.margins.total.1",
        `Worst annual margin ${(mMin * 100).toFixed(1)}% is a loss; Bear fades to the ${hist.length}-year average ${(mAvg * 100).toFixed(1)}% instead`,
      );

    const pre = last(A.PretaxIncome),
      taxp = last(A.TaxProvision);
    const effRaw = pre > 0 && taxp != null ? taxp / pre : null;
    const eff = effRaw != null ? Math.min(0.3, Math.max(0.1, effRaw)) : 0.21;
    if (effRaw == null)
      warn("operating.tax0", "No positive pre-tax income; tax rate set to the 21% statutory rate");
    else if (effRaw !== eff)
      warn(
        "operating.tax0",
        `Effective tax rate ${(effRaw * 100).toFixed(1)}% was clamped to ${(eff * 100).toFixed(0)}%`,
      );
    const avg3 = (s, absv) => {
      const v = (s || []).slice(-3).map((x) => (absv ? Math.abs(x[1]) : x[1]));
      return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
    };
    const rev3 = avg3(A.TotalRevenue);
    const daPct =
      avg3(A.ReconciledDepreciation) != null && rev3 ? avg3(A.ReconciledDepreciation) / rev3 : null;
    const capexPct =
      avg3(A.CapitalExpenditure, true) != null && rev3 ? avg3(A.CapitalExpenditure, true) / rev3 : null;
    // working capital: latest quarter's balances against that quarter's annualised revenue (a balance reflects the sales running at that date)
    const qd = lastDate(Q.TotalRevenue);
    const qParts = qd
      ? [Q.AccountsReceivable, Q.Inventory, Q.AccountsPayable].map((s_) => {
          const x = (s_ || []).find((e) => e[0] === qd);
          return x ? x[1] : null;
        })
      : [null, null, null];
    const nwcQ = qParts.every((x) => x != null);
    const nwcParts = nwcQ ? qParts : [last(A.AccountsReceivable), last(A.Inventory), last(A.AccountsPayable)];
    if (nwcParts.some((x) => x == null))
      warn(
        "operating.nwc_pct",
        "Receivables, inventory or payables missing in the feed; the missing item counts as zero",
      );
    const nwcLevel = (nwcParts[0] || 0) + (nwcParts[1] || 0) - (nwcParts[2] || 0);
    const nwcRev = nwcQ ? 4 * last(Q.TotalRevenue) : rev0 != null ? rev0 * M_ : null;
    const nwcPct = nwcRev ? nwcLevel / nwcRev : null;
    const nwc0 = nwcLevel / M_;
    const intExp = last(A.InterestExpense) != null ? Math.abs(last(A.InterestExpense)) / M_ : 0;
    if (last(A.InterestExpense) == null)
      warn(
        "capital.kd_pretax",
        "No interest expense in the feed; coverage looks unlimited and the cost of debt gets the AAA spread",
      );
    // year-1 margin from consensus EPS: EBIT = EPS × diluted shares ÷ (1 − tax) + interest, so the model's year-1 EPS equals consensus
    const shDil = shares != null ? shares + dilution : null;
    const mCons =
      epsY1 != null && epsY1 > 0 && revY1 && y1 && shDil ? ((epsY1 * shDil) / (1 - eff) + intExp) / y1 : null;
    if (mCons != null && mTTM != null && Math.abs(mCons - mTTM) > 0.1)
      warn(
        "scenarios.Base.margins.total.0",
        `Consensus EPS implies a ${(mCons * 100).toFixed(1)}% operating margin next year against ${(mTTM * 100).toFixed(1)}% trailing; Base year 1 uses consensus`,
      );
    const m1 = {
      Base: mCons ?? mTTM,
      Bear: mCons != null && mTTM != null ? Math.min(mCons, mTTM) : mTTM,
      Bull: mCons != null && mTTM != null ? Math.max(mCons, mTTM) : mTTM,
    };
    // trailing EV/EBITDA and EV/Sales mislead when sales are ramping hard: lean on forward P/E instead
    const ramp = ttmRev && last(Q.TotalRevenue) != null ? (4 * last(Q.TotalRevenue)) / ttmRev : null;
    const rampy = !FIN && ramp != null && ramp > 1.3;
    if (rampy)
      warn(
        "relative.weights",
        `Latest quarter annualised is ${ramp.toFixed(2)}x trailing revenue; trailing EV/EBITDA and EV/Sales understate the business, so comps use forward P/E only`,
      );
    const sh = A.OrdinarySharesNumber || [];
    const buyback =
      sh.length >= 3 ? Math.max(0, 1 - Math.pow(sh[sh.length - 1][1] / sh[sh.length - 3][1], 1 / 2)) : 0;

    // cost of capital
    const rf = num(f.tnx) != null ? f.tnx / 100 : null;
    const erp = f.erp != null ? f.erp : 0.0423;
    let bd = null,
      betaData = null;
    if (M && f.stock && f.index) {
      const r = M.returnsFromPrices(f.stock, f.index);
      let rs = r.rs,
        rm = r.rm,
        dates = r.dates;
      if (rs.length > 504) {
        rs = rs.slice(-504);
        rm = rm.slice(-504);
        dates = dates.slice(-504);
      }
      if (rs.length >= 60) {
        bd = M.capmBeta(rs, rm);
        bd.window = `${dates[0]} to ${dates[dates.length - 1]}`;
        bd.frequency = "daily (Yahoo adjusted closes)";
        const step = Math.max(1, Math.floor(rs.length / 600));
        bd.points = rs.map((y, i) => [rm[i], y]).filter((_, i) => i % step === 0);
        bd.xmin = Math.min(...rm);
        bd.xmax = Math.max(...rm);
        // the aligned closes behind the regression, so a spreadsheet can rerun it (SLOPE of daily returns)
        const im = new Map(f.index),
          al = f.stock
            .filter(([d]) => im.has(d))
            .sort((x, y) => (x[0] < y[0] ? -1 : 1))
            .slice(-(rs.length + 1));
        betaData = {
          dates: al.map((x) => x[0]),
          stock: al.map((x) => x[1]),
          index: al.map((x) => im.get(x[0])),
        };
      }
    }
    const ebitL = num(last(A.OperatingIncome)) != null ? last(A.OperatingIncome) / M_ : null;
    const cov = intExp > 0 && ebitL != null ? ebitL / intExp : 99;
    const [, rating, spread] = synthetic(cov);

    // debt and cash never default to zero: a missing figure blocks the call until it is entered
    const cashRaw = last(Q.CashAndCashEquivalents) ?? last(A.CashAndCashEquivalents) ?? raw(fd.totalCash);
    const debtRaw = last(Q.TotalDebt) ?? last(A.TotalDebt) ?? raw(fd.totalDebt);
    const cash = num(cashRaw) != null ? cashRaw / M_ : null;
    const totalDebt = num(debtRaw) != null ? debtRaw / M_ : null;
    const leases = (last(Q.CapitalLeaseObligations) ?? 0) / M_;
    const nci = (last(Q.MinorityInterest) ?? last(A.MinorityInterest) ?? 0) / M_;
    if (last(Q.TotalDebt) == null && debtRaw != null)
      warn(
        "bridge.core_debt",
        `Debt is from the ${last(A.TotalDebt) != null ? "last annual report" : "financialData summary"}, not the latest quarter`,
      );
    if (last(Q.CashAndCashEquivalents) == null && cashRaw != null)
      warn("bridge.cash", "Cash is not from the latest quarter");
    // currency: Yahoo reports statements in the filing currency (e.g. TWD for TSM) even when the quote is USD
    const finCcy = fd.financialCurrency || null,
      pxCcy = pr.currency || null;

    // ---------------- financials (banks, insurers, REITs) ----------------
    const finCalc = {};
    if (FIN) {
      const cseQ = Q.CommonStockEquity || Q.StockholdersEquity,
        cseA = A.CommonStockEquity || A.StockholdersEquity;
      const bvRaw = last(cseQ) ?? last(cseA);
      finCalc.bv0 = num(bvRaw) != null ? bvRaw / M_ : null;
      finCalc.bvDate = lastDate(cseQ) || lastDate(cseA);
      finCalc.tbv0 = last(Q.TangibleBookValue) != null ? last(Q.TangibleBookValue) / M_ : null;
      const niQ = Q.NetIncomeCommonStockholders || Q.NetIncome;
      const niTTM = sumLast(niQ, 4) ?? last(A.NetIncomeCommonStockholders) ?? last(A.NetIncome);
      const eqWin = (cseQ || []).slice(-5).map((x) => x[1]);
      const avgEq = eqWin.length ? eqWin.reduce((a, b) => a + b, 0) / eqWin.length : bvRaw;
      finCalc.roeTTM = niTTM != null && avgEq > 0 ? niTTM / avgEq : null;
      // annual ROE = net income to common / average of opening and closing common equity
      const niA = A.NetIncomeCommonStockholders || A.NetIncome || [];
      finCalc.roeHist = niA
        .map(([d, v]) => {
          const i = (cseA || []).findIndex((e) => e[0] === d);
          return i > 0 ? v / ((cseA[i][1] + cseA[i - 1][1]) / 2) : null;
        })
        .filter((x) => x != null);
      const rh = finCalc.roeHist,
        roeAvg = rh.length ? rh.reduce((a, b) => a + b, 0) / rh.length : finCalc.roeTTM;
      // year-1 ROE from consensus EPS where available
      finCalc.roe1 = epsY1 != null && shares && finCalc.bv0 ? (epsY1 * shares) / finCalc.bv0 : finCalc.roeTTM;
      finCalc.roeLR = finCalc.roeTTM != null && roeAvg != null ? (finCalc.roeTTM + roeAvg) / 2 : null;
      finCalc.roeMin = rh.length ? Math.min(...rh, finCalc.roeTTM ?? Infinity) : finCalc.roeTTM;
      // total payout (dividends + buybacks) / net income to common, 3-year average
      const pay = niA
        .slice(-3)
        .map(([d, v]) => {
          const dv = (A.CashDividendsPaid || []).find((e) => e[0] === d),
            bb = (A.RepurchaseOfCapitalStock || []).find((e) => e[0] === d);
          return v > 0 && dv ? (Math.abs(dv[1]) + Math.abs(bb ? bb[1] : 0)) / v : null;
        })
        .filter((x) => x != null);
      finCalc.payout = pay.length
        ? Math.min(1.2, Math.max(0, pay.reduce((a, b) => a + b, 0) / pay.length))
        : null;
      // dividend per share history: cash dividends / diluted average shares
      const dps = (A.CashDividendsPaid || [])
        .map(([d, v]) => {
          const s_ = (A.DilutedAverageShares || []).find((e) => e[0] === d);
          return s_ ? [d, Math.abs(v) / s_[1]] : null;
        })
        .filter(Boolean);
      finCalc.dpsHist = dps;
      const dg =
        dps.length >= 4
          ? Math.pow(dps[dps.length - 1][1] / dps[dps.length - 4][1], 1 / 3) - 1
          : dps.length >= 2
            ? dps[dps.length - 1][1] / dps[dps.length - 2][1] - 1
            : null;
      finCalc.dpsG = dg != null ? Math.min(0.1, Math.max(-0.05, dg)) : null;
      const dr = raw(sd.dividendRate) ?? raw(sd.trailingAnnualDividendRate);
      finCalc.dps0 = dr > 0 ? dr : null;
      const ffo =
        last(A.NetIncome) != null && last(A.ReconciledDepreciation) != null
          ? last(A.NetIncome) + last(A.ReconciledDepreciation)
          : null;
      finCalc.coverage =
        ffo != null && last(A.CashDividendsPaid) ? ffo / Math.abs(last(A.CashDividendsPaid)) : null;
      if (finCalc.bv0 == null)
        DQ.blocking.push({
          path: "fin.bv0",
          msg: "Common book equity is missing from the feed; enter it from the latest 10-Q",
        });
      if (MODEL === "excess_return") {
        if (finCalc.payout == null) warn("fin.payout", "No dividend or buyback history; payout set to 50%");
        if (epsY1 == null) warn("scenarios.Base.roe.0", "No consensus EPS; year-1 ROE uses the trailing ROE");
        if (rh.length < 2)
          warn(
            "scenarios.Base.roe.1",
            "Under two years of ROE history; the long-run ROE leans on the trailing year",
          );
        if (finCalc.tbv0 != null && finCalc.bv0 && finCalc.tbv0 / finCalc.bv0 < 0.6)
          warn(
            "fin.tbv0",
            `Tangible book is only ${((finCalc.tbv0 / finCalc.bv0) * 100).toFixed(0)}% of book; goodwill-heavy balance sheet`,
          );
      } else {
        if (finCalc.dps0 == null)
          DQ.blocking.push({
            path: "fin.dps0",
            msg: "No dividend rate in the feed; a dividend model needs one",
          });
        if (finCalc.dpsG == null)
          warn(
            "scenarios.Base.dps_growth.0",
            "Under two years of dividend history; dividend growth set to terminal growth",
          );
        if (finCalc.coverage != null && finCalc.coverage < 1)
          warn(
            "fin.coverage",
            `Dividends exceed net income + D&A (cover ${finCalc.coverage.toFixed(2)}x); the dividend may not be sustainable`,
          );
        warn(
          "fin.dps0",
          "REIT value uses dividends; FFO/AFFO are not in Yahoo's feed. Check the payout against AFFO in the supplement",
        );
      }
    }

    // both financial scenario sets, so the DES tab can switch between excess return and DDM
    const finScen = {},
      finRelW = { excess_return: { pe_fwd: 0.5, pb: 0.5 }, ddm: { div_yield: 0.5, ev_ebitda: 0.5 } },
      finVerW = {
        excess_return: { dcf: 0.4, mc: 0.2, comps: 0.3, regression: 0.1 },
        ddm: { dcf: 0.5, mc: 0.2, comps: 0.3, regression: 0 },
      };
    if (FIN) {
      finScen.excess_return = {
        Bear: { roe: [finCalc.roe1, finCalc.roeMin] },
        Base: { roe: [finCalc.roe1, finCalc.roeLR] },
        Bull: {
          roe: [
            finCalc.roe1,
            finCalc.roeLR != null && finCalc.roeTTM != null ? Math.max(finCalc.roeTTM, finCalc.roeLR) : null,
          ],
        },
      };
      const g0 = finCalc.dpsG ?? Math.min(0.03, Math.max(0.01, (rf ?? 0.04) - 0.01));
      finScen.ddm = {
        Bear: { dps_growth: [g0 - 0.02, g0 - 0.02, g0 - 0.02, g0 - 0.02] },
        Base: { dps_growth: [g0, g0, g0, g0] },
        Bull: { dps_growth: [g0 + 0.01, g0 + 0.01, g0 + 0.01, g0 + 0.01] },
      };
    }
    // peers
    const cols = [
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
    ];
    const rows = {};
    for (const [t, ps] of Object.entries(f.peers || {})) {
      if (!ps) continue;
      const ppr = ps.price || {},
        psd = ps.summaryDetail || {},
        pks = ps.defaultKeyStatistics || {},
        pfd = ps.financialData || {};
      const ptr = ((ps.earningsTrend && ps.earningsTrend.trend) || []).find((x) => x.period === "+1y");
      rows[t] = [
        ppr.longName || ppr.shortName || t,
        raw(ppr.regularMarketPrice),
        (raw(ppr.marketCap) || 0) / M_,
        (raw(pks.enterpriseValue) || 0) / M_,
        raw(psd.beta) ?? null,
        raw(psd.forwardPE) ?? raw(pks.forwardPE),
        raw(pks.enterpriseToEbitda),
        raw(pks.enterpriseToRevenue),
        ptr ? raw(ptr.growth) : null,
        (raw(pfd.totalDebt) || 0) / M_,
        (raw(pfd.totalRevenue) || 0) / M_,
        raw(pks.priceToBook) ?? null,
        raw(psd.dividendYield) ?? null,
        raw(pfd.returnOnEquity) ?? null,
      ];
    }
    const peerList = Object.keys(rows);
    if (peerList.length < 3)
      warn(
        "peers.core_set",
        `Only ${peerList.length} peer${peerList.length === 1 ? "" : "s"} with data; comps are weak`,
      );
    const missingPeers = Object.entries(f.peers || {})
      .filter(([, v]) => !v)
      .map(([t]) => t);
    if (missingPeers.length) warn("peers.core_set", `No data returned for ${missingPeers.join(", ")}`);
    if (!bd)
      warn(
        "capital.beta_daily",
        "Not enough daily prices for a regression beta; the Blume-adjusted 5-year beta is used",
      );
    // a regression that explains almost nothing is not a usable beta: fall back to Yahoo's 5-year monthly beta (Blume-adjusted)
    const weakBeta = bd && bd.r2 < 0.05 && raw(sd.beta) != null;
    if (weakBeta)
      warn(
        "capital.beta_source",
        `Daily regression beta ${bd.beta.toFixed(2)} explains only ${(bd.r2 * 100).toFixed(1)}% of the stock's moves (R²); using Yahoo's 5-year monthly beta ${raw(sd.beta).toFixed(2)}, Blume-adjusted. Switch on the BETA tab`,
      );

    const scen = (dg, mY1, mLR) => ({
      growth: {
        total: [
          gNext + dg,
          (gNext + dg) * 0.75 + 0.01,
          (gNext + dg) * 0.5 + 0.015,
          (gNext + dg) * 0.35 + 0.02,
        ],
      },
      margins: { total: [mY1, mLR] },
    });
    const ud = ((S.upgradeDowngradeHistory && S.upgradeDowngradeHistory.history) || [])
      .slice(0, 8)
      .map((h) => [
        h.firm,
        h.toGrade + (h.action === "down" ? " (downgrade)" : h.action === "up" ? " (upgrade)" : ""),
        h.currentPriceTarget || null,
        new Date(h.epochGradeDate * 1000).toISOString().slice(0, 10),
      ]);
    const rt = (S.recommendationTrend && S.recommendationTrend.trend && S.recommendationTrend.trend[0]) || {};
    const divRate = raw(sd.dividendRate) ?? raw(sd.trailingAnnualDividendRate) ?? 0;

    const annualLabels = (A.TotalRevenue || []).map(([d]) => fy(d));
    const pick = (s) =>
      (A.TotalRevenue || []).map(([d]) => {
        const x = (s || []).find((e) => e[0] === d);
        return x ? x[1] / M_ : null;
      });
    const qlab = (Q.TotalRevenue || []).slice(-8).map(([d]) => d.slice(0, 7));
    const qpick = (s) =>
      (Q.TotalRevenue || []).slice(-8).map(([d]) => {
        const x = (s || []).find((e) => e[0] === d);
        return x ? x[1] / M_ : null;
      });

    const inp = {
      meta: {
        ticker: f.ticker,
        name: pr.longName || pr.shortName || f.ticker,
        exchange: pr.exchangeName || "",
        currency: pxCcy || "USD",
        financial_currency: finCcy || pxCcy || "USD",
        units: "millions except per-share",
        valuation_date: today,
        price_date: today,
        fiscal_year_end_next: nextFYE,
        balance_sheet_date: bsDate,
        year_labels: labels,
        sector: (S.assetProfile || {}).sector || "",
        industry: (S.assetProfile || {}).industry || "",
        built: "live research (Yahoo Finance feeds via the Claude app browser)",
      },
      market: {
        price,
        prev_close: raw(pr.regularMarketPreviousClose) ?? price,
        shares_basic: shares,
        dilution,
        dividend_ps: divRate,
        high_52w: raw(sd.fiftyTwoWeekHigh) ?? price,
        low_52w: raw(sd.fiftyTwoWeekLow) ?? price,
        eps_ntm: raw(ks.forwardEps) ?? epsY2fromG ?? epsY1,
        ebitda_ttm: num(raw(fd.ebitda)) != null ? raw(fd.ebitda) / M_ : null,
        revenue_ttm: num(raw(fd.totalRevenue)) != null ? raw(fd.totalRevenue) / M_ : null,
        total_debt_incl_dfs: totalDebt,
      },
      model: MODEL,
      rev_y0: rev0,
      segment_defs: [{ key: "total", name: "Total revenue", margin: "total", report: "Company" }],
      margin_groups: { total: "Company EBIT margin" },
      ui: {
        key_segment: "total",
        key_margin: "total",
        key_segment_label: "Revenue",
        key_margin_label: "EBIT",
      },
      y1_segments: { total: y1 },
      scenarios: FIN
        ? finScen[MODEL]
        : {
            Bear: scen(-0.05, m1.Bear, bearLR),
            Base: scen(0, m1.Base, baseLR),
            Bull: scen(0.04, m1.Bull, Math.max(mTTM, baseLR)),
          },
      fin: FIN
        ? {
            bv0: finCalc.bv0,
            tbv0: finCalc.tbv0,
            bv_date: finCalc.bvDate,
            roe_ttm: finCalc.roeTTM,
            roe_hist: finCalc.roeHist,
            payout: finCalc.payout ?? 0.5,
            dps0: finCalc.dps0,
            dps_hist: finCalc.dpsHist,
            dps_cagr: finCalc.dpsG,
            dividend_cover: finCalc.coverage,
          }
        : undefined,
      operating: {
        corp_pct: 0,
        tax0: eff,
        tax_lr: Math.max(eff, 0.18),
        da_pct: daPct,
        capex_pct: capexPct,
        nwc_pct: nwcPct,
        nwc0,
        interest_net: intExp,
        buyback_pct: buyback,
      },
      terminal: {
        method: "gordon",
        g: rf != null ? Math.min(0.03, Math.max(0.01, rf - 0.01)) : 0.025,
        ronic: 0.15,
        exit_multiple: 12.0,
        ...(FIN ? { er_persistence: 0.5 } : {}),
      },
      timing: { first_period: firstPeriod, stub, mid_year: true },
      capital: {
        rf,
        erp,
        rm: rf != null ? rf + erp : null,
        crp: 0,
        beta_raw: raw(sd.beta) ?? (bd ? bd.beta : 1),
        beta_source: bd && !weakBeta ? "daily_adj" : "blume",
        beta_custom: bd ? bd.beta : (raw(sd.beta) ?? 1),
        beta_daily: bd ? bd.beta : null,
        beta_daily_stats: bd,
        beta_data: betaData,
        kd_pretax: rf != null ? rf + spread : null,
        tax_marginal: 0.21,
        size_premium: 0,
      },
      bridge: {
        cash,
        core_debt: totalDebt != null ? totalDebt - leases : null,
        leases,
        dfs_receivables: 0,
        dfs_lease_equipment: 0,
        dfs_debt: 0,
        investments: 0,
        nci,
      },
      consensus: {
        eps: { [labels[0]]: epsY1, [labels[1]]: epsConflict ? epsY2fromG : (epsY2avg ?? epsY2fromG) },
        revenue: { [labels[0]]: revY1 ? revY1 / M_ : null, [labels[1]]: revY2 ? revY2 / M_ : null },
        targets: {
          low: raw(fd.targetLowPrice),
          mean: raw(fd.targetMeanPrice),
          median: raw(fd.targetMedianPrice),
          high: raw(fd.targetHighPrice),
          n: raw(fd.numberOfAnalystOpinions) || 0,
        },
        ratings: {
          strong_buy: rt.strongBuy || 0,
          buy: rt.buy || 0,
          hold: rt.hold || 0,
          sell: rt.sell || 0,
          strong_sell: rt.strongSell || 0,
        },
        recent: ud,
        note:
          "Built from Yahoo Finance consensus. Base-case growth starts at consensus revenue for the next two fiscal years and fades after that." +
          (epsConflict
            ? ` Yahoo's ${labels[1]} EPS average (${epsY2avg.toFixed(2)}) disagrees with its own growth figure (implies ${epsY2fromG.toFixed(2)}); the growth-implied figure is used.`
            : ""),
      },
      peers: {
        columns: cols,
        rows,
        core_set: peerList,
        note: "Peer list starts from Yahoo's related-symbols feed (stocks viewed together), not an industry screen. Edit it to true comparables.",
      },
      relative: {
        weights: FIN
          ? finRelW[MODEL]
          : rampy
            ? { pe_fwd: 1, ev_ebitda: 0, ev_sales: 0 }
            : { pe_fwd: 0.4, ev_ebitda: 0.4, ev_sales: 0.2 },
        stat: "median",
        subject_growth: g1eps,
      },
      verdict: {
        weights: FIN ? finVerW[MODEL] : { dcf: 0.4, mc: 0.2, comps: 0.3, regression: 0.1 },
        band: 0.1,
        horizon_years: 1.0,
      },
      alt_models: FIN
        ? Object.fromEntries(
            ["excess_return", "ddm"]
              .filter((k) => k !== MODEL && (k !== "ddm" || finCalc.dps0 != null))
              .map((k) => [
                k,
                { scenarios: finScen[k], relative_weights: finRelW[k], verdict_weights: finVerW[k] },
              ]),
          )
        : undefined,
      history: {
        annual: {
          labels: annualLabels,
          revenue: pick(A.TotalRevenue),
          op_income: pick(A.OperatingIncome),
          net_income: pick(A.NetIncome),
          eps_diluted: (A.TotalRevenue || []).map(([d]) => {
            const x = (A.DilutedEPS || []).find((e) => e[0] === d);
            return x ? x[1] : null;
          }),
          cfo: pick(A.OperatingCashFlow),
          capex: pick(A.CapitalExpenditure).map((x) => (x == null ? null : Math.abs(x))),
          fcf: pick(A.FreeCashFlow),
        },
        annual_names: {
          revenue: "Revenue",
          op_income: "Operating income",
          net_income: "Net income",
          eps_diluted: "Diluted EPS",
          cfo: "Cash from operations",
          capex: "Capex",
          fcf: "Free cash flow",
        },
        annual_note:
          "Yahoo Finance standardised statements. Operating income can differ from the company's GAAP line where Yahoo normalises unusual items.",
        quarters: { labels: qlab, revenue: qpick(Q.TotalRevenue), op_income: qpick(Q.OperatingIncome) },
        quarter_names: { revenue: "Revenue", op_income: "Operating income" },
        quarter_margins: [["Operating margin", "op_income", "revenue"]],
        prices: (f.stock || []).slice(-85),
      },
      // statement lines for the earnings-quality check (Beneish M-score, accruals, cash conversion), $m
      quality_data: (() => {
        const ds = (A.TotalRevenue || []).map(([d]) => d);
        const col = (k) =>
          ds.map((d) => {
            const x = (A[k] || []).find((e) => e[0] === d);
            return x ? x[1] / M_ : null;
          });
        return {
          labels: ds.map(fy),
          dates: ds,
          sales: col("TotalRevenue"),
          cogs: col("CostOfRevenue"),
          sga: col("SellingGeneralAndAdministration"),
          rec: col("AccountsReceivable"),
          ca: col("CurrentAssets"),
          ppe: col("NetPPE"),
          ta: col("TotalAssets"),
          cl: col("CurrentLiabilities"),
          ltd: col("LongTermDebt"),
          dep: col("ReconciledDepreciation"),
          ni: A.NetIncomeContinuousOperations ? col("NetIncomeContinuousOperations") : col("NetIncome"),
          cfo: col("OperatingCashFlow"),
        };
      })(),
      bridge_names: {
        cash: "Cash & equivalents",
        core_debt: "Total debt ex leases",
        leases: "Capital leases",
        investments: "Investments",
        nci: "Non-controlling interests",
        dfs_receivables: "Captive-finance receivables",
        dfs_lease_equipment: "Captive-finance lease equipment",
        dfs_debt: "Captive-finance debt",
      },
      bridge_note:
        "Captive-finance arms and non-operating investments are not in Yahoo's feed. Check the 10-K and enter them if the company has them.",
      notes: {
        kd: `(rf + ${rating} spread, coverage ${cov.toFixed(1)}x)`,
        erp: "Damodaran implied ERP, Jan-2026 (edit to update)",
        rm:
          rf != null
            ? `Rm = 10Y yield + ERP (${(rf * 100).toFixed(2)}% + ${(erp * 100).toFixed(2)}%). Replace with your own expected S&P 500 return.`
            : "10-year yield missing: enter it and set Rm.",
        segments: "Single revenue line. Split into reported segments for a sell-side-grade model.",
        operating: `Margins: TTM ${(mTTM * 100).toFixed(1)}%, ${hist.length}-year average ${(mAvg * 100).toFixed(1)}%, low ${(mMin * 100).toFixed(1)}%. D&A and capex are 3-year averages of sales.`,
        comps: "All multiples come from the same vendor, so definitions match across peers.",
      },
      sources: {
        "Price, market cap, shares, beta (5Y), forward P/E, EV multiples, debt, cash, EBITDA, revenue (TTM), targets": `Yahoo Finance quoteSummary, ${today}`,
        "Annual and quarterly statements": `Yahoo Finance fundamentals-timeseries, ${today}`,
        "Daily prices and beta regression": `Yahoo Finance chart API (adjusted closes) for ${f.ticker} and ^GSPC, ${bd ? bd.window : "n/a"}`,
        "Risk-free rate": `^TNX (10-year Treasury yield) ${f.tnx != null ? f.tnx.toFixed(3) + "%" : "n/a"}, ${today}`,
        "Equity risk premium": "Damodaran implied ERP, Jan-2026 (assumption, editable)",
        "Cost of debt": `Synthetic rating ${rating} from interest coverage ${cov.toFixed(1)}x (Damodaran large-firm table)`,
        Peers: `Yahoo related symbols: ${peerList.join(", ") || "none"}`,
      },
      provenance: P,
      data_quality: DQ,
    };
    // unit check: shares × price should match the quoted market cap
    if (mcap && price && shares && Math.abs((shares * M_ * price) / mcap - 1) > 0.02)
      warn("market.shares_basic", "Shares × price differs from the quoted market cap by more than 2%");

    // ---------------- provenance ----------------
    src("market.price", "sourced", `${YQ} price.regularMarketPrice`);
    src(
      "market.shares_basic",
      "sourced",
      `${YQ} defaultKeyStatistics.impliedSharesOutstanding (checked against marketCap ÷ price)`,
    );
    src(
      "market.dilution",
      dilution ? "derived" : "assumption",
      dilution
        ? `${YTQ}: diluted average shares − ordinary shares, latest quarter`
        : "no dilution found in feed",
    );
    src("market.dividend_ps", "sourced", `${YQ} summaryDetail.dividendRate`);
    src("market.eps_ntm", "sourced", `${YQ} defaultKeyStatistics.forwardEps`);
    src("market.ebitda_ttm", "sourced", `${YQ} financialData.ebitda (TTM)`);
    src("market.revenue_ttm", "sourced", `${YQ} financialData.totalRevenue (TTM)`);
    src("market.total_debt_incl_dfs", "sourced", `${YTQ} TotalDebt, ${bsDate}`);
    src("rev_y0", "sourced", `${YT} TotalRevenue, ${lastFYE}`);
    src(
      "y1_segments.total",
      revY1 ? "sourced" : "derived",
      revY1 ? `${YQ} earningsTrend 0y revenue estimate (consensus)` : "last revenue × last year's growth",
    );
    for (const s of ["Bear", "Base", "Bull"]) {
      src(
        `scenarios.${s}.growth.total.0`,
        s === "Base" && revY2 ? "sourced" : "assumption",
        s === "Base"
          ? `consensus: ${labels[1]} ÷ ${labels[0]} revenue estimate − 1`
          : `Base ${s === "Bear" ? "−5pp" : "+4pp"}`,
      );
      for (const k of [1, 2, 3])
        src(`scenarios.${s}.growth.total.${k}`, "assumption", "fade: g×0.75+1%, g×0.5+1.5%, g×0.35+2%");
      src(
        `scenarios.${s}.margins.total.0`,
        "derived",
        mCons == null
          ? `TTM operating margin ${YTQ} (4 quarters)`
          : s === "Base"
            ? `${YQ} consensus ${labels[0]} EPS × diluted shares ÷ (1 − tax) + interest, ÷ consensus revenue (consensus EPS may be on an adjusted basis)`
            : `${s === "Bear" ? "lower" : "higher"} of consensus-implied and TTM operating margin`,
      );
      src(
        `scenarios.${s}.margins.total.1`,
        "assumption",
        s === "Base"
          ? "midpoint of TTM and multi-year average margin"
          : s === "Bear"
            ? bearLR !== mMin
              ? "multi-year average margin (the lowest year is a loss)"
              : "lowest margin in the annual history"
            : "TTM margin held",
      );
    }
    src("operating.tax0", "derived", `${YT} TaxProvision ÷ PretaxIncome (clamped 10–30%)`);
    src("operating.tax_lr", "assumption", "max(effective rate, 18%)");
    src("operating.da_pct", "derived", `${YT} D&A ÷ revenue, 3-year average`);
    src("operating.capex_pct", "derived", `${YT} capex ÷ revenue, 3-year average`);
    src(
      "operating.nwc_pct",
      "derived",
      nwcQ
        ? `${YTQ} (receivables + inventory − payables) ÷ (4 × revenue), quarter to ${qd}`
        : `${YT} (receivables + inventory − payables) ÷ revenue`,
    );
    src(
      "operating.nwc0",
      "derived",
      `${nwcQ ? YTQ : YT} receivables + inventory − payables, ${nwcQ ? qd : lastFYE}; year 1 change = NWC % × revenue − this level`,
    );
    src("operating.interest_net", "sourced", `${YT} InterestExpense`);
    src("operating.buyback_pct", "derived", `${YT} share-count decline per year, last 2 years`);
    src("operating.corp_pct", "assumption", "none (single segment)");
    src("terminal.g", "assumption", "3%, capped at rf − 1%");
    src("terminal.ronic", "assumption", "15% return on new invested capital");
    src("terminal.exit_multiple", "assumption", "12x EBITDA cross-check");
    src("timing.first_period", "derived", `days from ${today} to fiscal year end ${nextFYE}`);
    src("timing.stub", "derived", `share of the year after the last balance sheet (${bsDate})`);
    src("capital.rf", "sourced", "^TNX 10-year Treasury yield, Yahoo chart API");
    src("capital.erp", "assumption", "Damodaran implied ERP, Jan-2026");
    src("capital.rm", "assumption", "rf + ERP");
    src("capital.crp", "assumption", "0 (US)");
    src("capital.beta_raw", "sourced", `${YQ} summaryDetail.beta (5Y monthly)`);
    src("capital.beta_custom", "derived", "copy of daily regression beta");
    src(
      "capital.kd_pretax",
      "derived",
      `rf + ${rating} spread; coverage = operating income ÷ interest expense`,
    );
    src("capital.tax_marginal", "assumption", "21% US federal statutory rate");
    src("bridge.cash", "sourced", `${YTQ} CashAndCashEquivalents, ${bsDate}`);
    src("bridge.core_debt", "sourced", `${YTQ} TotalDebt − capital leases, ${bsDate}`);
    src("bridge.leases", "sourced", `${YTQ} CapitalLeaseObligations`);
    src("bridge.nci", "sourced", `${YTQ} MinorityInterest`);
    src("bridge.investments", "assumption", "not in feed; enter from the 10-K");
    for (const k of ["pe_fwd", "ev_ebitda", "ev_sales"])
      src(
        `relative.weights.${k}`,
        "assumption",
        rampy ? "forward P/E only (sales ramping: trailing multiples left out)" : "40/40/20 blend",
      );
    for (const k of ["dcf", "mc", "comps", "regression"])
      src(`verdict.weights.${k}`, "assumption", "method weights");
    src("verdict.band", "assumption", "±10% treated as fairly valued");
    src("verdict.horizon_years", "assumption", "convergence horizon");
    src(
      "quality_data",
      "sourced",
      `${YT}: revenue, cost of revenue, SG&A, receivables, current assets, net PP&E, total assets, current liabilities, long-term debt, D&A, net income (continuing), operating cash flow`,
    );
    src(
      "model",
      "assumption",
      `routed from Yahoo industry "${ind || "n/a"}"${MODEL === "excess_return" ? " → excess-return model" : MODEL === "ddm" ? " → dividend discount model" : " → FCFF DCF"}; change on the DES tab`,
    );
    if (FIN) {
      for (const k of Object.keys(P)) if (fcffOnly.test(k) || k.startsWith("relative.weights.")) delete P[k];
      src("fin.bv0", "sourced", `${YTQ} CommonStockEquity, ${finCalc.bvDate}`);
      src("fin.tbv0", "sourced", `${YTQ} TangibleBookValue`);
      src(
        "fin.roe_ttm",
        "derived",
        `${YTQ} net income to common (4 quarters) ÷ average common equity (5 quarter-ends)`,
      );
      src(
        "fin.payout",
        finCalc.payout != null ? "derived" : "assumption",
        finCalc.payout != null
          ? `${YT} (dividends + buybacks) ÷ net income to common, 3-year average`
          : "50% (no history)",
      );
      src("fin.dps0", "sourced", `${YQ} summaryDetail.dividendRate (annualised)`);
      src(
        "terminal.er_persistence",
        "assumption",
        "half of the long-run ROE spread over Ke survives in the terminal value",
      );
      for (const w of Object.keys(inp.relative.weights))
        src(`relative.weights.${w}`, "assumption", "50/50 blend");
      for (const sn of ["Bear", "Base", "Bull"]) {
        if (MODEL === "excess_return") {
          src(
            `scenarios.${sn}.roe.0`,
            epsY1 != null ? "derived" : "derived",
            epsY1 != null ? "consensus EPS (0y) × shares ÷ book equity" : "trailing ROE",
          );
          src(
            `scenarios.${sn}.roe.1`,
            "assumption",
            sn === "Base"
              ? "midpoint of trailing ROE and the annual ROE average"
              : sn === "Bear"
                ? "lowest annual ROE"
                : "trailing ROE held",
          );
        } else {
          for (const k of [0, 1, 2, 3])
            src(
              `scenarios.${sn}.dps_growth.${k}`,
              sn === "Base" && finCalc.dpsG != null ? "derived" : "assumption",
              sn === "Base"
                ? finCalc.dpsG != null
                  ? `${YT} dividends per share (cash dividends ÷ diluted shares), 3-year CAGR`
                  : "terminal growth (no history)"
                : `Base ${sn === "Bear" ? "−2pp" : "+1pp"}`,
            );
        }
      }
      inp.notes.operating =
        MODEL === "excess_return"
          ? `Excess-return model. ROE: trailing ${(finCalc.roeTTM * 100).toFixed(1)}%, annual history ${finCalc.roeHist.map((x) => (x * 100).toFixed(1) + "%").join(", ")}. Book grows by retained earnings: ROE × (1 − payout).`
          : `Dividend discount model. Dividend per share ${finCalc.dps0 != null ? finCalc.dps0.toFixed(2) : "n/a"}, 3-year growth ${finCalc.dpsG != null ? (finCalc.dpsG * 100).toFixed(1) + "%" : "n/a"}; (net income + D&A) covers dividends ${finCalc.coverage != null ? finCalc.coverage.toFixed(2) + "x" : "n/a"}.`;
      inp.notes.comps =
        MODEL === "excess_return"
          ? "Banks and insurers are compared on forward P/E and price-to-book; EV multiples mean nothing when debt is the product."
          : "REITs are compared on dividend yield and EV/EBITDA; Yahoo has no P/FFO.";
    }
    return inp;
  }

  const api = { Y, bodyOf, jsonOf, series, closes, build, synthetic };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.VResearch = api;
})(this);
