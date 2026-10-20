/* model.js — valuation engine shared by the terminal UI and the tests.
 * Pure functions, no DOM. Mirrors valuation_v2.py line for line; the
 * parity test (tests/test_parity.py) checks both give identical numbers.
 * Units: USD millions except per-share; rates as decimals.
 */
(function (root) {
  "use strict";
  // default segment map (Dell); any ticker supplies its own inp.segment_defs
  const DEFAULT_DEFS = [
    { key: "ai", name: "AI-optimised servers", margin: "ai", report: "ISG" },
    { key: "trad", name: "Traditional servers & networking", margin: "core", report: "ISG" },
    { key: "storage", name: "Storage", margin: "core", report: "ISG" },
    { key: "comm", name: "CSG commercial", margin: "csg", report: "CSG" },
    { key: "cons", name: "CSG consumer", margin: "csg", report: "CSG" },
    { key: "other", name: "Other", margin: null, report: "Other" },
  ];
  const defs = (inp) => inp.segment_defs || DEFAULT_DEFS;
  const okNum = (x) => typeof x === "number" && Number.isFinite(x);
  const segKeys = (inp) => defs(inp).map((d) => d.key);
  const keySeg = (inp) => (inp.ui && inp.ui.key_segment) || defs(inp)[0].key;
  const keyMargin = (inp) => (inp.ui && inp.ui.key_margin) || defs(inp).find((d) => d.margin).margin;
  const N = 10; // FY27..FY36
  // valuation model: "fcff" (default), "excess_return" (banks, insurers), "ddm" (REITs)
  const modelOf = (inp) => inp.model || "fcff";
  const isFin = (inp) => modelOf(inp) !== "fcff";
  const EXPLICIT = 4; // FY28..FY31 growth given per segment

  const sum = (a) => a.reduce((s, x) => s + x, 0);
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const median = (a) => {
    const s = [...a].sort((x, y) => x - y),
      m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  const quantile = (a, q) => {
    // linear interpolation (numpy default)
    const s = [...a].sort((x, y) => x - y),
      p = (s.length - 1) * q;
    const lo = Math.floor(p),
      hi = Math.ceil(p);
    return s[lo] + (s[hi] - s[lo]) * (p - lo);
  };

  // ------------------------------------------------------------ cost of capital
  function beta(inp) {
    const c = inp.capital;
    const blume = 0.67 * c.beta_raw + 0.33;
    const bu = bottomUpBeta(inp);
    const daily = c.beta_daily != null ? c.beta_daily : null;
    const chosen =
      c.beta_source === "raw"
        ? c.beta_raw
        : c.beta_source === "bottom_up"
          ? bu.relevered
          : c.beta_source === "custom"
            ? c.beta_custom
            : c.beta_source === "daily" && daily != null
              ? daily
              : c.beta_source === "daily_adj" && daily != null
                ? 0.67 * daily + 0.33
                : c.beta_source === "crsp_monthly" && c.beta_crsp_monthly != null
                  ? c.beta_crsp_monthly
                  : c.beta_source === "crsp_monthly_adj" && c.beta_crsp_monthly != null
                    ? 0.67 * c.beta_crsp_monthly + 0.33
                    : blume;
    return {
      raw: c.beta_raw,
      blume,
      bottom_up: bu.relevered,
      unlevered_median: bu.median,
      daily,
      crsp: c.beta_crsp_monthly != null ? c.beta_crsp_monthly : null,
      chosen,
    };
  }

  function bottomUpBeta(inp) {
    const t = inp.capital.tax_marginal,
      cols = inp.peers.columns;
    const ix = (k) => cols.indexOf(k);
    const bu = [];
    for (const tk of inp.peers.core_set) {
      const r = inp.peers.rows[tk];
      const b = r[ix("beta")],
        d = r[ix("debt")],
        e = r[ix("mcap")];
      bu.push(b == null ? null : b / (1 + ((1 - t) * d) / e)); // a peer with no published beta is left out, not set to 1
    }
    const ok = bu.filter((x) => x != null);
    const med = ok.length ? median(ok) : NaN;
    const E = inp.market.price * dilutedShares(inp);
    const D = inp.bridge.core_debt + inp.bridge.leases;
    return { unlevered: bu, median: med, relevered: med * (1 + ((1 - t) * D) / E) };
  }

  function dilutedShares(inp) {
    return inp.market.shares_basic + inp.market.dilution;
  }

  function wacc(inp) {
    const c = inp.capital,
      b = beta(inp);
    // Ke = rf + beta x (Rm - rf) + country risk premium (+ optional size premium)
    const mrp = marketRiskPremium(c);
    const ke = c.rf + b.chosen * mrp + (c.crp || 0) + (c.size_premium || 0);
    const E = inp.market.price * dilutedShares(inp);
    const D = inp.bridge.core_debt + inp.bridge.leases;
    const we = E / (E + D),
      wd = D / (E + D);
    const kd = c.kd_pretax * (1 - c.tax_marginal);
    return {
      wacc: we * ke + wd * kd,
      ke,
      kd_after: kd,
      we,
      wd,
      E,
      D,
      beta: b,
      mrp,
      rm: c.rf + mrp,
      equityPremium: b.chosen * mrp,
    };
  }

  // MRP = Rm - rf when an expected market return is given; otherwise the stored premium
  function marketRiskPremium(c) {
    return c.rm != null ? c.rm - c.rf : c.erp;
  }

  // CAPM regression on aligned price series: [[date, close], ...] for stock and index
  function returnsFromPrices(stock, index) {
    const m = new Map(index.map(([d, p]) => [d, p]));
    const rows = stock.filter(([d]) => m.has(d)).sort((a, b) => (a[0] < b[0] ? -1 : 1));
    const rs = [],
      rm = [],
      dates = [];
    for (let i = 1; i < rows.length; i++) {
      const [d0, s0] = rows[i - 1],
        [d1, s1] = rows[i];
      rs.push(s1 / s0 - 1);
      rm.push(m.get(d1) / m.get(d0) - 1);
      dates.push(d1);
    }
    return { rs, rm, dates };
  }
  function capmBeta(rs, rm, periodsPerYear) {
    const n = rs.length,
      ppy = periodsPerYear || 252;
    const mx = sum(rm) / n,
      my = sum(rs) / n;
    let sxy = 0,
      sxx = 0,
      syy = 0;
    for (let i = 0; i < n; i++) {
      sxy += (rm[i] - mx) * (rs[i] - my);
      sxx += (rm[i] - mx) ** 2;
      syy += (rs[i] - my) ** 2;
    }
    const b = sxy / sxx,
      a = my - b * mx;
    let sse = 0;
    for (let i = 0; i < n; i++) sse += (rs[i] - a - b * rm[i]) ** 2;
    const s2 = sse / (n - 2),
      seB = Math.sqrt(s2 / sxx),
      seA = Math.sqrt(s2 * (1 / n + (mx * mx) / sxx));
    const r2 = 1 - sse / syy;
    return {
      beta: b,
      alpha: a,
      seBeta: seB,
      seAlpha: seA,
      tBeta: b / seB,
      tAlpha: a / seA,
      r2,
      n,
      alphaAnnual: Math.pow(1 + a, ppy) - 1,
      blume: 0.67 * b + 0.33,
      ci95: [b - 1.96 * seB, b + 1.96 * seB],
      volStock: Math.sqrt((syy / (n - 1)) * ppy),
      volMkt: Math.sqrt((sxx / (n - 1)) * ppy),
    };
  }

  // ------------------------------------------------------------ projection
  function growthPath(explicit, gT) {
    // FY28..FY31 explicit; FY32..FY36 fade linearly from FY31 rate to gT
    const g = explicit.slice(0, EXPLICIT);
    const last = g[EXPLICIT - 1];
    const fadeYears = N - 1 - EXPLICIT; // 5
    for (let j = 1; j <= fadeYears; j++) g.push(last + ((gT - last) * j) / fadeYears);
    return g; // length 9 (FY28..FY36)
  }

  function marginPath(m0, mLR) {
    // FY27 = m0, linear to mLR at FY31 (index 4), flat after
    const out = [];
    for (let i = 0; i < N; i++) out.push(i >= EXPLICIT ? mLR : m0 + ((mLR - m0) * i) / EXPLICIT);
    return out;
  }

  function project(inp, scen) {
    const op = inp.operating,
      gT = inp.terminal.g;
    const SEGS = segKeys(inp),
      D = defs(inp);
    const seg = {};
    for (const s of SEGS) {
      const g = growthPath(scen.growth[s], gT);
      const r = [(inp.y1_segments || inp.fy27_segments)[s]];
      for (let i = 0; i < N - 1; i++) r.push(r[i] * (1 + g[i]));
      seg[s] = r;
    }
    const mp = {};
    for (const g of Object.keys(scen.margins)) mp[g] = marginPath(scen.margins[g][0], scen.margins[g][1]);
    const taxP = marginPath(op.tax0, op.tax_lr);
    const rows = [];
    let prevRev = inp.rev_y0;
    let shares = dilutedShares(inp);
    for (let i = 0; i < N; i++) {
      const rev = sum(SEGS.map((s) => seg[s][i]));
      const segRev = {},
        reportRev = {},
        reportOI = {};
      let segOIsum = 0;
      for (const d of D) {
        const oi = d.margin ? seg[d.key][i] * mp[d.margin][i] : 0;
        segRev[d.key] = seg[d.key][i];
        reportRev[d.report] = (reportRev[d.report] || 0) + seg[d.key][i];
        reportOI[d.report] = (reportOI[d.report] || 0) + oi;
        segOIsum += oi;
      }
      const corp = op.corp_pct * rev;
      const ebit = segOIsum - corp;
      const tax = taxP[i];
      const nopat = ebit * (1 - tax);
      const da = op.da_pct * rev,
        capex = op.capex_pct * rev;
      // year 1 against the opening balance when it is known; afterwards NWC % × the change in revenue
      const dnwc = i === 0 && op.nwc0 != null ? op.nwc_pct * rev - op.nwc0 : op.nwc_pct * (rev - prevRev);
      const fcff = nopat + da - capex - dnwc;
      if (i > 0) shares = shares * (1 - op.buyback_pct);
      const ni = (ebit - op.interest_net) * (1 - tax);
      const niAdj = (ebit + corp - op.interest_net) * (1 - tax);
      rows.push({
        label: inp.meta.year_labels[i],
        rev,
        growth: rev / prevRev - 1,
        seg: segRev,
        reportRev,
        reportOI,
        corp,
        ebit,
        margin: ebit / rev,
        tax,
        nopat,
        da,
        ebitda: ebit + da,
        capex,
        dnwc,
        fcff,
        shares,
        ni,
        eps: ni / shares,
        epsAdj: niAdj / shares,
      });
      prevRev = rev;
    }
    return rows;
  }

  // ------------------------------------------------------------ DCF
  function dcf(inp, scen, override) {
    if (isFin(inp)) return finValue(inp, scen, override);
    const w = override && override.wacc != null ? override.wacc : wacc(inp).wacc;
    const tm = Object.assign({}, inp.terminal, override && override.terminal);
    const rows = project(inp, scen);
    const { first_period: fp, stub, mid_year: mid } = inp.timing;
    let pvSum = 0;
    rows.forEach((r, i) => {
      const frac = i === 0 ? stub : 1;
      const tEnd = fp + i;
      const t = Math.max(mid ? tEnd - 0.5 * frac : tEnd, 0);
      r.frac = frac;
      r.t = t;
      r.df = Math.pow(1 + w, -t);
      r.pv = r.fcff * frac * r.df;
      pvSum += r.pv;
    });
    const last = rows[N - 1],
      tEndN = fp + N - 1;
    if (tm.g >= w) throw new Error("terminal growth must be below WACC");
    const tvGordon = (last.nopat * (1 + tm.g) * (1 - tm.g / tm.ronic)) / (w - tm.g);
    const pvGordon = tvGordon * Math.pow(1 + w, -(mid ? tEndN - 0.5 : tEndN));
    const tvExit = tm.exit_multiple * last.ebitda;
    const pvExit = tvExit * Math.pow(1 + w, -tEndN);
    const useExit = tm.method === "exit";
    const pvTV = useExit ? pvExit : pvGordon;
    const ev = pvSum + pvTV;
    const b = inp.bridge;
    const dfsNet = b.dfs_receivables + b.dfs_lease_equipment - b.dfs_debt;
    const equity = ev + b.cash - b.core_debt - b.leases + dfsNet + b.investments - b.nci;
    const sh = dilutedShares(inp);
    // implied cross-checks
    // g that makes a plain Gordon TV on FCFF_N equal the exit-multiple TV
    const impliedG = (tvExit * w - last.fcff) / (tvExit + last.fcff);
    return {
      wacc: w,
      rows,
      pvExplicit: pvSum,
      tvGordon,
      pvGordon,
      tvExit,
      pvExit,
      pvTV,
      tv: useExit ? tvExit : tvGordon,
      ev,
      dfsNet,
      equity,
      perShare: equity / sh,
      tvShare: pvTV / ev,
      impliedExitMultiple: tvGordon / last.ebitda,
      impliedGFromExit: impliedG,
      perShareGordon: (pvSum + pvGordon + equity - ev) / sh,
      perShareExit: (pvSum + pvExit + equity - ev) / sh,
    };
  }

  // ------------------------------------------------------------ financials
  // Discount rate is the cost of equity: for banks, insurers and REITs debt is raw material, not financing.
  // ROE path: year-1 ROE fades linearly to the long-run ROE by year 5, then flat (same shape as margins).
  function roePath(scen) {
    return marginPath(scen.roe[0], scen.roe[1]);
  }
  // dividend growth: 4 explicit years, then linear fade to terminal g by year 10
  function dpsGrowthPath(explicit, gT) {
    const e = explicit.slice(0, EXPLICIT),
      out = [];
    for (let i = 0; i < N; i++)
      out.push(
        i < EXPLICIT
          ? e[i]
          : e[EXPLICIT - 1] + ((gT - e[EXPLICIT - 1]) * (i - EXPLICIT + 1)) / (N - EXPLICIT),
      );
    return out;
  }
  function finValue(inp, scen, override) {
    const r = override && override.wacc != null ? override.wacc : wacc(inp).ke;
    const tm = Object.assign({}, inp.terminal, override && override.terminal);
    if (tm.g >= r) throw new Error("terminal growth must be below the cost of equity");
    const { first_period: fp, stub, mid_year: mid } = inp.timing;
    const labels = (inp.meta && inp.meta.year_labels) || [];
    const sh = dilutedShares(inp),
      tEndN = fp + N - 1;
    const tTV = mid ? tEndN - 0.5 : tEndN;
    const rows = [];
    let pvSum = 0;
    if (modelOf(inp) === "excess_return") {
      // Equity = BV0 + sum PV[(ROE_t - Ke) x BV_{t-1}] + PV[terminal excess return]
      const roe = roePath(scen),
        p = scen.payout != null ? scen.payout : inp.fin.payout;
      let bv = inp.fin.bv0;
      for (let i = 0; i < N; i++) {
        const frac = i === 0 ? stub : 1;
        const ni = roe[i] * bv * frac,
          charge = r * bv * frac,
          er = ni - charge,
          dist = p * ni;
        const t = Math.max(mid ? fp + i - 0.5 * frac : fp + i, 0),
          df = Math.pow(1 + r, -t);
        rows.push({
          label: labels[i] || "Y" + (i + 1),
          roe: roe[i],
          bvBegin: bv,
          ni,
          charge,
          er,
          dist,
          bvEnd: bv + ni - dist,
          frac,
          t,
          df,
          pv: er * df,
          epsAdj: ni / frac / sh,
          bvps: (bv + ni - dist) / sh,
        });
        pvSum += er * df;
        bv = bv + ni - dist;
      }
      // terminal: ROE keeps only part of its spread over Ke forever (persistence 1 = all of it, 0 = none)
      const persist = tm.er_persistence != null ? tm.er_persistence : 1;
      const roeT = r + persist * (scen.roe[1] - r);
      const tv = ((roeT - r) * bv) / (r - tm.g),
        pvTV = tv * Math.pow(1 + r, -tTV);
      const equity = inp.fin.bv0 + pvSum + pvTV;
      return {
        model: "excess_return",
        wacc: r,
        ke: r,
        rows,
        bv0: inp.fin.bv0,
        pvExplicit: pvSum,
        tv,
        pvTV,
        roeTerminal: roeT,
        equity,
        perShare: equity / sh,
        tvShare: pvTV / equity,
        justifiedPB: equity / inp.fin.bv0,
        bvps0: inp.fin.bv0 / sh,
        perShareGordon: equity / sh,
        perShareExit: NaN,
      };
    }
    // DDM (per share): PV of dividends still to be paid + Gordon value of year-10 dividend
    const g = dpsGrowthPath(scen.dps_growth, tm.g);
    let d = inp.fin.dps0;
    for (let i = 0; i < N; i++) {
      const frac = i === 0 ? Math.min(1, Math.max(0, fp)) : 1;
      d = d * (1 + g[i]);
      const t = Math.max(mid ? fp + i - 0.5 * frac : fp + i, 0),
        df = Math.pow(1 + r, -t);
      rows.push({
        label: labels[i] || "Y" + (i + 1),
        g: g[i],
        dps: d,
        paid: d * frac,
        frac,
        t,
        df,
        pv: d * frac * df,
        epsAdj: NaN,
      });
      pvSum += d * frac * df;
    }
    const tv = (d * (1 + tm.g)) / (r - tm.g),
      pvTV = tv * Math.pow(1 + r, -tTV);
    const per = pvSum + pvTV;
    return {
      model: "ddm",
      wacc: r,
      ke: r,
      rows,
      pvExplicit: pvSum,
      tv,
      pvTV,
      equity: per * sh,
      perShare: per,
      tvShare: pvTV / per,
      impliedYield: inp.fin.dps0 / per,
      perShareGordon: per,
      perShareExit: NaN,
    };
  }

  // ------------------------------------------------------------ solvers
  function bisect(f, lo, hi, tol) {
    let flo = f(lo),
      fhi = f(hi);
    if (flo * fhi > 0) return NaN;
    for (let k = 0; k < 200; k++) {
      const mid = (lo + hi) / 2,
        fm = f(mid);
      if (Math.abs(hi - lo) < (tol || 1e-12)) return mid;
      if (flo * fm <= 0) {
        hi = mid;
        fhi = fm;
      } else {
        lo = mid;
        flo = fm;
      }
    }
    return (lo + hi) / 2;
  }

  // apply a named knob to (inp, scen) copies
  function applyKnob(inp, scen, knob, x) {
    const I = clone(inp),
      S = clone(scen),
      SEGS = segKeys(inp);
    if (isFin(inp)) {
      switch (knob) {
        case "roe_lr":
          S.roe[1] = x;
          return { I, S, ov: null };
        case "roe_y1":
          S.roe[0] = x;
          return { I, S, ov: null };
        case "roe_shift":
          S.roe = S.roe.map((v) => v + x);
          return { I, S, ov: null };
        case "payout":
          S.payout = x;
          return { I, S, ov: null };
        case "persistence":
          I.terminal.er_persistence = x;
          return { I, S, ov: null };
        case "dps_shift":
          S.dps_growth = S.dps_growth.map((v) => v + x);
          return { I, S, ov: null };
        case "dps_y1":
          S.dps_growth[0] = x;
          return { I, S, ov: null };
        case "wacc":
        case "g":
        case "erp":
        case "rm":
        case "beta":
          break;
        default:
          throw new Error("knob " + knob + " does not apply to the " + modelOf(inp) + " model");
      }
    }
    // generic knobs: seg_shift:<key>, seg_fy1:<key>, margin_lr:<group>; legacy Dell aliases map onto them
    const alias = {
      ai_shift: "seg_shift:" + keySeg(inp),
      ai_fy28: "seg_fy1:" + keySeg(inp),
      core_margin_lr: "margin_lr:" + keyMargin(inp),
      ai_margin_lr: "margin_lr:ai",
      csg_margin_lr: "margin_lr:csg",
    };
    knob = alias[knob] || knob;
    const [kind, arg] = knob.split(":");
    if (kind === "seg_shift") {
      S.growth[arg] = S.growth[arg].map((v) => v + x);
      return { I, S, ov: null };
    }
    if (kind === "seg_fy1") {
      S.growth[arg][0] = x;
      return { I, S, ov: null };
    }
    if (kind === "margin_lr") {
      S.margins[arg][1] = x;
      return { I, S, ov: null };
    }
    switch (knob) {
      case "wacc":
        return { I, S, ov: { wacc: x } };
      case "g":
        I.terminal.g = x;
        break;
      case "exit_multiple":
        I.terminal.exit_multiple = x;
        break;
      case "all_growth_shift":
        for (const s of SEGS) S.growth[s] = S.growth[s].map((v) => v + x);
        break;
      case "margin_shift":
        for (const k of Object.keys(S.margins)) S.margins[k][1] += x;
        break;
      case "nwc_pct":
        I.operating.nwc_pct = x;
        break;
      case "capex_pct":
        I.operating.capex_pct = x;
        break;
      case "erp":
        if (I.capital.rm != null) I.capital.rm = I.capital.rf + x;
        else I.capital.erp = x;
        break;
      case "rm":
        I.capital.rm = x;
        break;
      case "beta":
        I.capital.beta_source = "custom";
        I.capital.beta_custom = x;
        break;
      case "ronic":
        I.terminal.ronic = x;
        break;
      default:
        throw new Error("unknown knob " + knob);
    }
    return { I, S, ov: null };
  }

  function valueAt(inp, scen, knob, x) {
    const { I, S, ov } = applyKnob(inp, scen, knob, x);
    return dcf(I, S, ov).perShare;
  }

  function reverse(inp, scen, knob, lo, hi, price) {
    const p = price == null ? inp.market.price : price;
    return bisect((x) => valueAt(inp, scen, knob, x) - p, lo, hi, 1e-12);
  }

  function sensitivity(inp, scen, k1, v1, k2, v2) {
    return v1.map((a) =>
      v2.map((b) => {
        const A = applyKnob(inp, scen, k1, a);
        const B = applyKnob(A.I, A.S, k2, b);
        const ov = Object.assign({}, A.ov || {}, B.ov || {});
        try {
          return dcf(B.I, B.S, Object.keys(ov).length ? ov : null).perShare;
        } catch (e) {
          return NaN;
        }
      }),
    );
  }

  function tornado(inp, scen, specs) {
    const base = dcf(inp, scen).perShare;
    return specs
      .map(([label, knob, lo, hi]) => ({
        label,
        lo: valueAt(inp, scen, knob, lo),
        hi: valueAt(inp, scen, knob, hi),
        base,
      }))
      .sort((a, b) => Math.abs(b.hi - b.lo) - Math.abs(a.hi - a.lo));
  }

  // ------------------------------------------------------------ Monte Carlo
  function rng(seed) {
    // mulberry32 + Box-Muller
    let a = seed >>> 0;
    const u = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    const n = () => {
      if (spare != null) {
        const s = spare;
        spare = null;
        return s;
      }
      let x, y, r;
      do {
        x = 2 * u() - 1;
        y = 2 * u() - 1;
        r = x * x + y * y;
      } while (r >= 1 || r === 0);
      const m = Math.sqrt((-2 * Math.log(r)) / r);
      spare = y * m;
      return x * m;
    };
    const tri = (lo, mode, hi) => {
      const p = u(),
        c = (mode - lo) / (hi - lo);
      return p < c
        ? lo + Math.sqrt(p * (hi - lo) * (mode - lo))
        : hi - Math.sqrt((1 - p) * (hi - lo) * (hi - mode));
    };
    return { u, n, tri };
  }

  function monteCarlo(inp, scen, opts) {
    if (isFin(inp)) return monteCarloFin(inp, scen, opts);
    const o = Object.assign(
      {
        n: 5000,
        seed: 7,
        sdAI: 0.12,
        sdCore: 0.02,
        sdCSG: 0.01,
        sdAllG: 0.02,
        waccLo: -0.01,
        waccHi: 0.015,
        gLo: 0.02,
        gHi: 0.035,
      },
      opts || {},
    );
    const R = rng(o.seed),
      w0 = wacc(inp).wacc,
      ks = keySeg(inp),
      km = keyMargin(inp),
      SEGS = segKeys(inp);
    const vals = [];
    for (let k = 0; k < o.n; k++) {
      const I = clone(inp),
        S = clone(scen);
      const dAI = o.sdAI * R.n(),
        dCore = o.sdCore * R.n(),
        dCSG = o.sdCSG * R.n(),
        dAll = o.sdAllG * R.n();
      S.growth[ks][0] += dAI;
      for (const s of SEGS) S.growth[s] = S.growth[s].map((v) => v + dAll);
      for (const g of Object.keys(S.margins))
        S.margins[g][1] += g === km ? dCore : g === ks ? dCore / 2 : dCSG;
      const w = R.tri(w0 + o.waccLo, w0, w0 + o.waccHi);
      I.terminal.g = R.tri(o.gLo, inp.terminal.g, o.gHi);
      if (w - I.terminal.g < 0.01) continue;
      vals.push(dcf(I, S, { wacc: w }).perShare);
    }
    vals.sort((a, b) => a - b);
    const p = inp.market.price;
    return {
      values: vals,
      n: vals.length,
      mean: sum(vals) / vals.length,
      median: quantile(vals, 0.5),
      p5: quantile(vals, 0.05),
      p10: quantile(vals, 0.1),
      p25: quantile(vals, 0.25),
      p75: quantile(vals, 0.75),
      p90: quantile(vals, 0.9),
      p95: quantile(vals, 0.95),
      probAbove: vals.filter((v) => v > p).length / vals.length,
    };
  }

  // financials: shock long-run ROE (or dividend growth), year-1 ROE, the cost of equity and terminal g
  function monteCarloFin(inp, scen, opts) {
    const o = Object.assign(
      {
        n: 5000,
        seed: 7,
        sdRoeLR: 0.015,
        sdRoe1: 0.01,
        sdDps: 0.01,
        keLo: -0.01,
        keHi: 0.015,
        gLo: 0.015,
        gHi: 0.035,
      },
      opts || {},
    );
    const R = rng(o.seed),
      k0 = wacc(inp).ke,
      er = modelOf(inp) === "excess_return";
    const vals = [];
    for (let k = 0; k < o.n; k++) {
      const I = clone(inp),
        S = clone(scen);
      const a = R.n(),
        b = R.n();
      if (er) {
        S.roe[1] += o.sdRoeLR * a;
        S.roe[0] += o.sdRoe1 * b;
      } else S.dps_growth = S.dps_growth.map((v) => v + o.sdDps * a);
      const r = R.tri(k0 + o.keLo, k0, k0 + o.keHi);
      I.terminal.g = R.tri(o.gLo, Math.min(Math.max(inp.terminal.g, o.gLo), o.gHi), o.gHi);
      if (r - I.terminal.g < 0.01) continue;
      vals.push(finValue(I, S, { wacc: r }).perShare);
    }
    vals.sort((x, y) => x - y);
    const p = inp.market.price;
    return {
      values: vals,
      n: vals.length,
      mean: sum(vals) / vals.length,
      median: quantile(vals, 0.5),
      p5: quantile(vals, 0.05),
      p10: quantile(vals, 0.1),
      p25: quantile(vals, 0.25),
      p75: quantile(vals, 0.75),
      p90: quantile(vals, 0.9),
      p95: quantile(vals, 0.95),
      probAbove: vals.filter((v) => v > p).length / vals.length,
    };
  }

  // ------------------------------------------------------------ relative
  function peerFrame(inp, set) {
    const cols = inp.peers.columns;
    return (set || inp.peers.core_set).map((tk) => {
      const r = inp.peers.rows[tk],
        o = { ticker: tk };
      cols.forEach((c, i) => (o[c] = r[i]));
      return o;
    });
  }

  function stat(xs, how) {
    const v = xs.filter((x) => x != null && isFinite(x) && x > 0);
    if (!v.length) return { k: NaN, n: 0 };
    const k =
      how === "harmonic"
        ? v.length / sum(v.map((x) => 1 / x))
        : how === "mean"
          ? sum(v) / v.length
          : median(v);
    return {
      k,
      n: v.length,
      q1: quantile(v, 0.25),
      q3: quantile(v, 0.75),
      min: Math.min(...v),
      max: Math.max(...v),
    };
  }

  function relative(inp, opts) {
    const o = Object.assign(
      { set: null, stat: inp.relative.stat, weights: inp.relative.weights },
      opts || {},
    );
    const peers = peerFrame(inp, o.set);
    const m = inp.market,
      sh = dilutedShares(inp);
    // EV-based: Dell EV on the same basis as data vendors (all debt incl. DFS, less cash)
    const netDebtAll = m.total_debt_incl_dfs + inp.bridge.leases - inp.bridge.cash;
    const evDell = m.price * sh + netDebtAll;
    const bvps = inp.fin && okNum(inp.fin.bv0) ? inp.fin.bv0 / sh : null,
      dps0 = inp.fin ? inp.fin.dps0 : null;
    const all = {
      pe_fwd: { metric: m.eps_ntm, own: m.price / m.eps_ntm, toPrice: (k) => k * m.eps_ntm },
      pb: { metric: bvps, own: m.price / bvps, toPrice: (k) => k * bvps },
      div_yield: { metric: dps0, own: dps0 / m.price, toPrice: (k) => dps0 / k },
      ev_ebitda: {
        metric: m.ebitda_ttm,
        own: evDell / m.ebitda_ttm,
        toPrice: (k) => (k * m.ebitda_ttm - netDebtAll) / sh,
      },
      ev_sales: {
        metric: m.revenue_ttm,
        own: evDell / m.revenue_ttm,
        toPrice: (k) => (k * m.revenue_ttm - netDebtAll) / sh,
      },
    };
    // only the multiples that carry a weight for this company (and exist in the peer table) are computed
    const spec = {};
    for (const k of Object.keys(all))
      if ((o.weights[k] || 0) > 0 && inp.peers.columns.includes(k)) spec[k] = all[k];
    const table = {};
    let wsum = 0,
      acc = 0;
    for (const key of Object.keys(spec)) {
      const st = stat(
        peers.map((p) => p[key]),
        o.stat,
      );
      const s = spec[key];
      // a multiple is used only if the company's own metric is positive and peers give a finite multiple
      const usable =
        okNum(s.metric) &&
        s.metric > 0 &&
        Number.isFinite(st.k) &&
        (!key.startsWith("ev_") || okNum(m.total_debt_incl_dfs));
      table[key] = Object.assign(st, {
        own: usable ? s.own : NaN,
        premium: usable ? s.own / st.k - 1 : NaN,
        implied: usable ? s.toPrice(st.k) : NaN,
        impliedQ1: usable ? s.toPrice(st.q1) : NaN,
        impliedQ3: usable ? s.toPrice(st.q3) : NaN,
        weight: usable ? o.weights[key] || 0 : 0,
        usable,
      });
      if (table[key].weight > 0) {
        wsum += table[key].weight;
        acc += table[key].weight * table[key].implied;
      }
    }
    return { peers, table, blended: wsum > 0 ? acc / wsum : NaN, netDebtAll, evDell };
  }

  // a peer regression that explains little of the peer spread is reported but not used in the blend
  const regMinR2 = (inp) => (inp.relative && inp.relative.reg_min_r2 != null ? inp.relative.reg_min_r2 : 0.3);
  function regGate(inp, r2, value) {
    const weak = !(r2 >= regMinR2(inp));
    return { impliedRaw: value, implied: weak ? NaN : value, weak, minR2: regMinR2(inp) };
  }
  function regression(inp, dellGrowth, set) {
    if (modelOf(inp) === "excess_return") return regressionPB(inp, set);
    // cross-sectional OLS: fwd P/E = a + b * EPS growth (peers with g > 0)
    const P = peerFrame(inp, set || Object.keys(inp.peers.rows)).filter((p) => p.eps_g3y > 0 && p.pe_fwd > 0);
    const x = P.map((p) => p.eps_g3y),
      y = P.map((p) => p.pe_fwd),
      n = x.length;
    const mx = sum(x) / n,
      my = sum(y) / n;
    let sxy = 0,
      sxx = 0,
      syy = 0;
    for (let i = 0; i < n; i++) {
      sxy += (x[i] - mx) * (y[i] - my);
      sxx += (x[i] - mx) ** 2;
      syy += (y[i] - my) ** 2;
    }
    const b = sxy / sxx,
      a = my - b * mx;
    const r2 = (sxy * sxy) / (sxx * syy);
    const pe = a + b * dellGrowth;
    return {
      a,
      b,
      r2,
      n,
      points: P.map((p) => ({ t: p.ticker, x: p.eps_g3y, y: p.pe_fwd })),
      dellGrowth,
      pe,
      ...regGate(inp, r2, pe * inp.market.eps_ntm),
    };
  }

  // banks: cross-sectional OLS of P/B on ROE (Wilcox); implied price = fitted P/B x own book per share
  function regressionPB(inp, set) {
    const P = peerFrame(inp, set || Object.keys(inp.peers.rows)).filter(
      (p) => okNum(p.roe) && p.roe > 0 && okNum(p.pb) && p.pb > 0,
    );
    const x = P.map((p) => p.roe),
      y = P.map((p) => p.pb),
      n = x.length;
    const roe = inp.fin.roe_ttm,
      bvps = inp.fin.bv0 / dilutedShares(inp);
    if (n < 3)
      return {
        a: NaN,
        b: NaN,
        r2: NaN,
        n,
        points: [],
        dellGrowth: roe,
        pe: NaN,
        implied: NaN,
        kind: "pb_roe",
      };
    const mx = sum(x) / n,
      my = sum(y) / n;
    let sxy = 0,
      sxx = 0,
      syy = 0;
    for (let i = 0; i < n; i++) {
      sxy += (x[i] - mx) * (y[i] - my);
      sxx += (x[i] - mx) ** 2;
      syy += (y[i] - my) ** 2;
    }
    const b = sxy / sxx,
      a = my - b * mx,
      pb = a + b * roe;
    return {
      a,
      b,
      r2: (sxy * sxy) / (sxx * syy),
      n,
      points: P.map((p) => ({ t: p.ticker, x: p.roe, y: p.pb })),
      dellGrowth: roe,
      pe: pb,
      ...regGate(inp, (sxy * sxy) / (sxx * syy), pb * bvps),
      kind: "pb_roe",
    };
  }

  // ------------------------------------------------------------ earnings quality
  // Beneish (1999) 8-variable M-score; coefficients from the unweighted probit.
  const BENEISH = {
    c: -4.84,
    DSRI: 0.92,
    GMI: 0.528,
    AQI: 0.404,
    SGI: 0.892,
    DEPI: 0.115,
    SGAI: -0.172,
    TATA: 4.679,
    LVGI: -0.327,
  };
  // sample means of the manipulators in Beneish (1999), shown as reference points
  const BENEISH_MANIP = {
    DSRI: 1.465,
    GMI: 1.193,
    AQI: 1.254,
    SGI: 1.607,
    DEPI: 1.077,
    SGAI: 1.041,
    LVGI: 1.111,
    TATA: 0.031,
  };
  const EQ_LIMITS = { mRed: -1.78, mAmber: -2.22, accrRed: 0.1, accrAmber: 0.05, ccRed: 0.5, ccAmber: 0.8 };
  function beneishYear(q, t) {
    const at = (k, i) => (q[k] && okNum(q[k][i]) ? q[k][i] : null);
    const ratio = (f) => {
      try {
        const x = f();
        return okNum(x) ? x : null;
      } catch (e) {
        return null;
      }
    };
    const s = (i) => at("sales", i);
    const idx = {
      DSRI: ratio(() => at("rec", t) / s(t) / (at("rec", t - 1) / s(t - 1))),
      GMI: ratio(() => (s(t - 1) - at("cogs", t - 1)) / s(t - 1) / ((s(t) - at("cogs", t)) / s(t))),
      AQI: ratio(
        () =>
          (1 - (at("ca", t) + at("ppe", t)) / at("ta", t)) /
          (1 - (at("ca", t - 1) + at("ppe", t - 1)) / at("ta", t - 1)),
      ),
      SGI: ratio(() => s(t) / s(t - 1)),
      DEPI: ratio(
        () =>
          at("dep", t - 1) /
          (at("dep", t - 1) + at("ppe", t - 1)) /
          (at("dep", t) / (at("dep", t) + at("ppe", t))),
      ),
      SGAI: ratio(() => at("sga", t) / s(t) / (at("sga", t - 1) / s(t - 1))),
      LVGI: ratio(
        () =>
          (at("cl", t) + at("ltd", t)) /
          at("ta", t) /
          ((at("cl", t - 1) + at("ltd", t - 1)) / at("ta", t - 1)),
      ),
      TATA: ratio(() => (at("ni", t) - at("cfo", t)) / at("ta", t)),
    };
    // a missing input makes a null, and null * coefficient must not count as 0: missing indices are set to 1 (neutral)
    // and listed; TATA (accruals) has no neutral value, so without it there is no M-score
    const neutral = Object.keys(idx).filter((k) => k !== "TATA" && idx[k] == null);
    const used = Object.assign({}, idx);
    for (const k of neutral) used[k] = 1;
    let m = idx.TATA == null || neutral.length > 2 ? NaN : BENEISH.c;
    if (isFinite(m)) for (const k of Object.keys(used)) m += BENEISH[k] * used[k];
    return { label: q.labels[t], idx, neutral, m };
  }
  function earningsQuality(inp) {
    const q = inp.quality_data;
    if (modelOf(inp) === "excess_return")
      return {
        grade: "n/a",
        reason:
          "Not meaningful for banks and insurers: no cost of sales, and operating cash flow includes lending and deposit flows",
        years: [],
        flags: [],
      };
    if (!q || !q.labels || q.labels.length < 2)
      return {
        grade: "n/a",
        reason: "No statement history for an earnings-quality check",
        years: [],
        flags: [],
      };
    const n = q.labels.length,
      years = [];
    for (let t = 1; t < n; t++) years.push(beneishYear(q, t));
    const last = years[years.length - 1];
    const v = (k, i) => (q[k] && okNum(q[k][i]) ? q[k][i] : null);
    // Sloan (1996) balance of accruals: (net income − operating cash flow) ÷ average total assets
    const accruals =
      v("ni", n - 1) != null && v("cfo", n - 1) != null && v("ta", n - 1) != null && v("ta", n - 2) != null
        ? (v("ni", n - 1) - v("cfo", n - 1)) / ((v("ta", n - 1) + v("ta", n - 2)) / 2)
        : NaN;
    const k = Math.min(3, n),
      sumOf = (key) => {
        let a = 0;
        for (let i = n - k; i < n; i++) {
          if (v(key, i) == null) return NaN;
          a += v(key, i);
        }
        return a;
      };
    const sNi = sumOf("ni"),
      sCfo = sumOf("cfo");
    const cashConv = sNi > 0 ? sCfo / sNi : NaN;
    const flags = [],
      L = EQ_LIMITS;
    if (isFinite(last.m)) {
      if (last.m > L.mRed)
        flags.push({
          level: "red",
          key: "m",
          msg: `Beneish M-score ${last.m.toFixed(2)} is above −1.78, the level Beneish found for likely manipulators`,
        });
      else if (last.m > L.mAmber)
        flags.push({
          level: "amber",
          key: "m",
          msg: `Beneish M-score ${last.m.toFixed(2)} is between −2.22 and −1.78 (grey zone)`,
        });
    }
    if (last.neutral.length)
      flags.push({
        level: "info",
        key: "neutral",
        msg: `M-score inputs missing (${last.neutral.join(", ")}): set to 1, which is neutral`,
      });
    if (isFinite(accruals)) {
      if (accruals > L.accrRed)
        flags.push({
          level: "red",
          key: "accruals",
          msg: `Accruals are ${(accruals * 100).toFixed(1)}% of assets: earnings run well ahead of cash`,
        });
      else if (accruals > L.accrAmber)
        flags.push({
          level: "amber",
          key: "accruals",
          msg: `Accruals are ${(accruals * 100).toFixed(1)}% of assets`,
        });
    }
    if (isFinite(cashConv)) {
      if (cashConv < L.ccRed)
        flags.push({
          level: "red",
          key: "cash",
          msg: `Operating cash flow is only ${(cashConv * 100).toFixed(0)}% of net income over ${k} years`,
        });
      else if (cashConv < L.ccAmber)
        flags.push({
          level: "amber",
          key: "cash",
          msg: `Operating cash flow is ${(cashConv * 100).toFixed(0)}% of net income over ${k} years`,
        });
    }
    const reds = flags.filter((f) => f.level === "red").length,
      ambers = flags.filter((f) => f.level === "amber").length;
    const tested = isFinite(last.m) || isFinite(accruals) || isFinite(cashConv);
    const grade = !tested
      ? "n/a"
      : reds >= 2
        ? "D"
        : reds === 1 || ambers >= 2
          ? "C"
          : ambers === 1
            ? "B"
            : "A";
    return {
      grade,
      years,
      latest: last,
      accruals,
      cashConv,
      cashYears: k,
      flags,
      limits: L,
      coef: BENEISH,
      manip: BENEISH_MANIP,
    };
  }

  // ------------------------------------------------------------ verdict
  // ---- fail-closed input checks: a blocking item stops the call; a warning is shown next to it
  const getPath = (o, p) => p.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
  function checkInputs(inp) {
    const blocking = [],
      warnings = [];
    const need = (path, label, test, why) => {
      const v = getPath(inp, path);
      if (!okNum(v)) blocking.push({ path, msg: `${label} is missing` });
      else if (test && !test(v)) blocking.push({ path, msg: `${label} ${why}` });
    };
    need("market.price", "Share price", (v) => v > 0, "must be above zero");
    need("market.shares_basic", "Shares outstanding", (v) => v > 0, "must be above zero");
    need("market.dilution", "Dilution", (v) => v >= 0, "cannot be negative");
    const fin = isFin(inp),
      mdl = modelOf(inp);
    if (!["fcff", "excess_return", "ddm"].includes(mdl))
      blocking.push({ path: "model", msg: `Unknown valuation model "${mdl}"` });
    if (!fin) {
      need("rev_y0", "Last fiscal-year revenue", (v) => v > 0, "must be above zero");
      for (const k of segKeys(inp))
        need(`y1_segments.${k}`, `Next-year revenue (${k})`, (v) => v >= 0, "cannot be negative");
      need("bridge.cash", "Cash", (v) => v >= 0, "cannot be negative");
      need("bridge.core_debt", "Debt", (v) => v >= 0, "cannot be negative");
      for (const k of ["leases", "investments", "nci", "dfs_receivables", "dfs_lease_equipment", "dfs_debt"])
        if (inp.bridge[k] !== undefined) need(`bridge.${k}`, `Bridge item ${k}`);
    } else if (mdl === "excess_return") {
      need(
        "fin.bv0",
        "Common book equity",
        (v) => v > 0,
        "must be above zero (negative book breaks an excess-return model)",
      );
      need(
        "fin.payout",
        "Payout ratio (dividends + buybacks)",
        (v) => v >= 0 && v <= 1.5,
        "is outside 0–150%",
      );
      if (inp.terminal.er_persistence != null)
        need(
          "terminal.er_persistence",
          "Excess-return persistence",
          (v) => v >= 0 && v <= 1,
          "must be between 0 and 1",
        );
    } else if (mdl === "ddm") {
      need("fin.dps0", "Dividend per share", (v) => v > 0, "must be above zero for a dividend model");
    }
    need("capital.rf", "Risk-free rate (10Y yield)", (v) => v > -0.01 && v < 0.2, "is outside −1% to 20%");
    if (inp.capital.rm != null)
      need(
        "capital.rm",
        "Expected market return",
        (v) => v > inp.capital.rf,
        "must exceed the risk-free rate",
      );
    else need("capital.erp", "Equity risk premium", (v) => v > 0, "must be above zero");
    need("capital.kd_pretax", "Pre-tax cost of debt", (v) => v >= 0 && v < 0.3, "is outside 0–30%");
    need("capital.tax_marginal", "Marginal tax rate", (v) => v >= 0 && v < 0.6, "is outside 0–60%");
    if (!fin)
      for (const k of ["tax0", "tax_lr", "da_pct", "capex_pct", "nwc_pct"])
        need(`operating.${k}`, `Operating input ${k}`);
    need("terminal.g", "Terminal growth", (v) => v < 0.06, "is above 6%, faster than the economy forever");
    const parts = fin ? (mdl === "excess_return" ? ["roe"] : ["dps_growth"]) : ["growth", "margins"];
    for (const [sn, sc] of Object.entries(inp.scenarios))
      for (const part of parts) {
        if (Array.isArray(sc[part]) || sc[part] == null) {
          const arr = sc[part] || [];
          if (!arr.length)
            blocking.push({ path: `scenarios.${sn}.${part}`, msg: `${sn} ${part} is missing` });
          arr.forEach((v, i) => {
            if (!okNum(v))
              blocking.push({
                path: `scenarios.${sn}.${part}.${i}`,
                msg: `${sn} ${part} (item ${i + 1}) is missing`,
              });
          });
        } else
          for (const [g, arr] of Object.entries(sc[part]))
            arr.forEach((v, i) => {
              if (!okNum(v))
                blocking.push({
                  path: `scenarios.${sn}.${part}.${g}.${i}`,
                  msg: `${sn} ${part} (${g}, item ${i + 1}) is missing`,
                });
            });
      }
    const meta = inp.meta || {};
    if (meta.financial_currency && meta.currency && meta.financial_currency !== meta.currency)
      blocking.push({
        path: "meta.financial_currency",
        msg: `Statements are in ${meta.financial_currency} but the price is in ${meta.currency}. Convert the statements or value the local listing.`,
      });
    const dq = inp.data_quality || {};
    for (const b of dq.blocking || []) blocking.push(b);
    for (const w of dq.warnings || []) warnings.push(w);
    if (!blocking.length) {
      const b = beta(inp),
        c = inp.capital;
      if (!okNum(b.chosen) || b.chosen < -1 || b.chosen > 5)
        blocking.push({ path: "capital.beta_source", msg: "Beta is missing or outside −1 to 5" });
      const wantDaily = c.beta_source === "daily" || c.beta_source === "daily_adj";
      const wantCrsp = c.beta_source === "crsp_monthly" || c.beta_source === "crsp_monthly_adj";
      if ((wantDaily && c.beta_daily == null) || (wantCrsp && c.beta_crsp_monthly == null))
        warnings.push({
          path: "capital.beta_source",
          msg: `Beta source "${c.beta_source}" has no data; the Blume-adjusted feed beta is used instead`,
        });
      const st = c.beta_daily_stats;
      if (wantDaily && st && okNum(st.n) && st.n < 250)
        warnings.push({ path: "capital.beta_daily", msg: `Daily beta uses only ${st.n} days` });
      if (wantDaily && st && okNum(st.r2) && st.r2 < 0.05)
        warnings.push({
          path: "capital.beta_daily",
          msg: `Daily beta R² is ${st.r2.toFixed(3)}; the market explains almost none of the stock's moves`,
        });
      const W = wacc(inp);
      if (fin) {
        if (!(W.ke > inp.terminal.g + 0.005))
          blocking.push({
            path: "terminal.g",
            msg: `Cost of equity (${(W.ke * 100).toFixed(2)}%) must exceed terminal growth by at least 0.5pp`,
          });
      } else {
        if (inp.terminal.method !== "exit" && !(W.wacc > inp.terminal.g + 0.005))
          blocking.push({
            path: "terminal.g",
            msg: `WACC (${(W.wacc * 100).toFixed(2)}%) must exceed terminal growth by at least 0.5pp`,
          });
        if (okNum(inp.terminal.ronic) && inp.terminal.ronic <= inp.terminal.g)
          warnings.push({
            path: "terminal.ronic",
            msg: "RONIC at or below terminal growth makes growth destroy value",
          });
      }
    }
    if (meta.balance_sheet_date && meta.valuation_date) {
      const age = (new Date(meta.valuation_date) - new Date(meta.balance_sheet_date)) / 86400000;
      if (age > 200)
        warnings.push({
          path: "meta.balance_sheet_date",
          msg: `Balance sheet is ${Math.round(age)} days old`,
        });
    }
    return { blocking, warnings, ok: blocking.length === 0 };
  }

  function verdict(inp, parts) {
    const Q = checkInputs(inp);
    const w = inp.verdict.weights,
      band = inp.verdict.band,
      p = inp.market.price;
    const dropped = Object.keys(w).filter((k) => w[k] > 0 && !(parts[k] != null && isFinite(parts[k])));
    const warnings = Q.warnings.concat(
      dropped.map((k) => ({
        path: `verdict.weights.${k}`,
        msg: `${k} gave no value and is left out of the blend`,
      })),
    );
    if (!Q.ok || dropped.length === Object.keys(w).filter((k) => w[k] > 0).length)
      return {
        fair: NaN,
        upside: NaN,
        call: "NOT VALUED",
        expectedReturn: NaN,
        requiredReturn: NaN,
        alpha: NaN,
        divYield: NaN,
        blocking: Q.ok
          ? [{ path: "verdict.weights", msg: "No valuation method produced a value" }]
          : Q.blocking,
        warnings,
        dropped,
      };
    let acc = 0,
      ws = 0;
    for (const k of Object.keys(w))
      if (parts[k] != null && isFinite(parts[k])) {
        acc += w[k] * parts[k];
        ws += w[k];
      }
    // earnings quality gates the call: D withholds it, C marks it low confidence (switch off with verdict.quality_gate = false)
    const EQ = earningsQuality(inp),
      gate = inp.verdict.quality_gate !== false;
    if (EQ.grade === "C" || EQ.grade === "D")
      warnings.push({
        path: "quality_data",
        msg: `Earnings quality ${EQ.grade}: ${EQ.flags
          .filter((f) => f.level !== "info")
          .map((f) => f.msg)
          .join("; ")}`,
      });
    const fv = acc / ws,
      up = fv / p - 1;
    const h = inp.verdict.horizon_years;
    const dy = inp.market.dividend_ps / p;
    const expRet = Math.pow(fv / p, 1 / h) - 1 + dy;
    const ke = wacc(inp).ke;
    const raw = up > band ? "UNDERVALUED" : up < -band ? "OVERVALUED" : "FAIRLY VALUED";
    const gated = gate && EQ.grade === "D";
    return {
      fair: fv,
      upside: up,
      call: gated ? "NO CALL" : raw,
      rawCall: raw,
      gated,
      confidence: gate && EQ.grade === "C" ? "low" : "normal",
      quality: { grade: EQ.grade, flags: EQ.flags },
      expectedReturn: expRet,
      requiredReturn: ke,
      alpha: expRet - ke,
      divYield: dy,
      blocking: [],
      warnings,
      dropped,
    };
  }

  function pvgo(inp) {
    const ke = wacc(inp).ke,
      e1 = inp.market.eps_ntm,
      p = inp.market.price;
    return { noGrowth: e1 / ke, pvgo: p - e1 / ke, share: (p - e1 / ke) / p };
  }

  // run everything for a scenario name
  function runAll(inp, scenName, mcOpts) {
    const Q = checkInputs(inp);
    if (!Q.ok) return { checks: Q, verdict: verdict(inp, {}) };
    const scen = inp.scenarios[scenName];
    const W = wacc(inp);
    const base = dcf(inp, scen);
    const scen3 = {};
    for (const k of Object.keys(inp.scenarios)) scen3[k] = dcf(inp, inp.scenarios[k]);
    const rel = relative(inp);
    const rows = base.rows;
    const sg = inp.relative && inp.relative.subject_growth;
    const growth = sg != null && isFinite(sg) ? sg : Math.pow(rows[3].epsAdj / rows[0].epsAdj, 1 / 3) - 1;
    const reg = modelOf(inp) === "ddm" ? { implied: NaN, n: 0 } : regression(inp, growth);
    const mc = monteCarlo(inp, scen, mcOpts);
    const v = verdict(inp, {
      dcf: base.perShare,
      mc: mc.median,
      comps: rel.blended,
      regression: reg.implied,
    });
    if (reg.weak && isFinite(reg.r2))
      (v.warnings = v.warnings.filter((w) => w.path !== "verdict.weights.regression")),
        v.warnings.push({
          path: "relative.reg_min_r2",
          msg: `Peer regression R² ${reg.r2.toFixed(2)} is below ${reg.minR2.toFixed(2)}; it is shown but left out of the blend`,
        });
    return { checks: Q, W, base, scen3, rel, reg, mc, verdict: v, pvgo: pvgo(inp) };
  }

  const api = {
    marketRiskPremium,
    returnsFromPrices,
    capmBeta,
    DEFAULT_DEFS,
    defs,
    segKeys,
    keySeg,
    keyMargin,
    N,
    beta,
    bottomUpBeta,
    dilutedShares,
    wacc,
    growthPath,
    marginPath,
    project,
    dcf,
    reverse,
    sensitivity,
    tornado,
    monteCarlo,
    relative,
    regression,
    checkInputs,
    earningsQuality,
    beneishYear,
    BENEISH,
    verdict,
    pvgo,
    runAll,
    valueAt,
    applyKnob,
    stat,
    median,
    quantile,
    clone,
    regMinR2,
    finValue,
    roePath,
    dpsGrowthPath,
    modelOf,
    isFin,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.VModel = api;
})(this);
