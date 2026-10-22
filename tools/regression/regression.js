/* regression.js: OLS, logit and probit for the Speedy Regression tool.
 *
 * No DOM access, so the same file runs in the browser (window.RegressionCore) and in Node, where
 * tests/test_regression_js.py checks every number against statsmodels.
 *
 * Numerics: regressors are centred and scaled before the normal equations are solved, so variables such as
 * a year (about 2000) or a market value (about 1e12) cannot make the problem ill-conditioned; coefficients and
 * covariances are mapped back to the original units afterwards. Logit and probit are fitted by Newton-Raphson
 * with step halving, with the observed information matrix, which is what Stata and statsmodels report.
 * Memory and time are linear in the number of observations.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RegressionCore = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const COLLINEAR_TOL = 1e-9; // relative size of a pivot below which a column counts as collinear
  const MISSING = /^(na|n\/a|#n\/a|nan|null|none|\.)$/i;
  const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

  // ------------------------------------------------------------------------------ reading values
  /** null = missing, NaN = text that is not a number, otherwise the number. */
  function toNumber(v) {
    if (v === null || v === undefined) return null;
    if (typeof v === "number") return Number.isFinite(v) ? v : null;
    if (typeof v === "boolean") return v ? 1 : 0;
    const s = String(v).trim();
    if (s === "" || MISSING.test(s)) return null;
    if (/^(true|false)$/i.test(s)) return s.toLowerCase() === "true" ? 1 : 0;
    return NUMBER.test(s) ? Number(s) : NaN;
  }

  /** Split the columns into numeric ones (usable in a model) and text ones. */
  function classifyColumns(rows, headers) {
    const numeric = [];
    const text = [];
    for (const h of headers) {
      let seen = 0;
      let isText = false;
      for (const r of rows) {
        const v = toNumber(r[h]);
        if (v === null) continue;
        if (Number.isNaN(v)) {
          isText = true;
          break;
        }
        seen++;
      }
      (isText || seen === 0 ? text : numeric).push(h);
    }
    return { numeric, text };
  }

  /** Complete-case design: rows with a missing value in Y or any X are dropped, never replaced by zero. */
  function designMatrix(rows, yName, xNames) {
    const y = [];
    const X = [];
    const missingBy = {};
    const used = [];
    for (const name of [yName, ...xNames]) missingBy[name] = 0;
    rows.forEach((r, idx) => {
      const vals = [yName, ...xNames].map((name) => toNumber(r[name]));
      let ok = true;
      vals.forEach((v, i) => {
        if (v === null) {
          missingBy[i === 0 ? yName : xNames[i - 1]]++;
          ok = false;
        } else if (Number.isNaN(v)) {
          throw new Error(
            `"${i === 0 ? yName : xNames[i - 1]}" contains text, so it cannot be used as a number.`,
          );
        }
      });
      if (!ok) return;
      y.push(vals[0]);
      X.push(vals.slice(1));
      used.push(idx);
    });
    return { y, X, nTotal: rows.length, nUsed: y.length, nDropped: rows.length - y.length, missingBy, used };
  }

  // ------------------------------------------------------------------------------ linear algebra
  function zeros(r, c) {
    return Array.from({ length: r }, () => new Float64Array(c));
  }

  /** Cholesky factor of a symmetric matrix; `bad` is the first column that is (almost) linearly dependent. */
  function cholesky(A, tol) {
    const p = A.length;
    const L = zeros(p, p);
    for (let j = 0; j < p; j++) {
      let s = A[j][j];
      for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
      if (!(s > tol * A[j][j])) return { bad: j };
      L[j][j] = Math.sqrt(s);
      for (let i = j + 1; i < p; i++) {
        let t = A[i][j];
        for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
        L[i][j] = t / L[j][j];
      }
    }
    return { L };
  }

  function cholSolve(L, b) {
    const p = L.length;
    const u = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * u[k];
      u[i] = s / L[i][i];
    }
    const x = new Float64Array(p);
    for (let i = p - 1; i >= 0; i--) {
      let s = u[i];
      for (let k = i + 1; k < p; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }

  function cholInverse(L) {
    const p = L.length;
    const inv = zeros(p, p);
    for (let c = 0; c < p; c++) {
      const e = new Float64Array(p);
      e[c] = 1;
      const col = cholSolve(L, e);
      for (let r = 0; r < p; r++) inv[r][c] = col[r];
    }
    return inv;
  }

  function matMul(A, B) {
    const out = zeros(A.length, B[0].length);
    for (let i = 0; i < A.length; i++)
      for (let k = 0; k < B.length; k++) {
        const a = A[i][k];
        if (a === 0) continue;
        for (let j = 0; j < B[0].length; j++) out[i][j] += a * B[k][j];
      }
    return out;
  }

  function transpose(A) {
    const out = zeros(A[0].length, A.length);
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) out[j][i] = A[i][j];
    return out;
  }

  // ------------------------------------------------------------------------------ set-up shared by all models
  /**
   * Centre and scale the k regressors, append the intercept as the last column and factorise Z'Z.
   * Returns Z (n rows of length k+1), the map A with beta = A theta, and the Cholesky factor.
   */
  function prepare(X, names) {
    const n = X.length;
    const k = names.length;
    const p = k + 1;
    if (n <= p) throw new Error(`Not enough observations (${n}) for ${p} coefficients.`);
    const mean = new Float64Array(k);
    const sd = new Float64Array(k);
    for (const row of X) for (let j = 0; j < k; j++) mean[j] += row[j] / n;
    for (const row of X) for (let j = 0; j < k; j++) sd[j] += (row[j] - mean[j]) ** 2 / n;
    for (let j = 0; j < k; j++) {
      sd[j] = Math.sqrt(sd[j]);
      if (!(sd[j] > 1e-12 * Math.max(1, Math.abs(mean[j])))) {
        throw new Error(`"${names[j]}" is constant, so it duplicates the intercept. Remove it.`);
      }
    }
    const Z = X.map((row) => {
      const z = new Float64Array(p);
      for (let j = 0; j < k; j++) z[j] = (row[j] - mean[j]) / sd[j];
      z[k] = 1;
      return z;
    });
    const ZtZ = zeros(p, p);
    for (const z of Z) for (let a = 0; a < p; a++) for (let b = 0; b <= a; b++) ZtZ[a][b] += z[a] * z[b];
    for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) ZtZ[a][b] = ZtZ[b][a];
    const chol = cholesky(ZtZ, COLLINEAR_TOL);
    if (chol.bad !== undefined) {
      throw new Error(
        `"${names[chol.bad]}" is (almost) a linear combination of the variables above it ` +
          `(perfect multicollinearity). Remove one of them.`,
      );
    }
    const A = zeros(p, p);
    for (let j = 0; j < k; j++) {
      A[j][j] = 1 / sd[j];
      A[k][j] = -mean[j] / sd[j];
    }
    A[k][k] = 1;
    return { Z, A, chol: chol.L, ZtZ, n, p, k };
  }

  /** Coefficient table in original units from theta (standardised) and its covariance. */
  function table(names, theta, V, A, pvalue, crit) {
    const p = theta.length;
    const beta = new Float64Array(p);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) beta[i] += A[i][j] * theta[j];
    const Vb = matMul(matMul(A, V), transpose(A));
    return [...names, "_cons"].map((name, i) => {
      const se = Math.sqrt(Vb[i][i]);
      const s = beta[i] / se;
      return {
        name,
        coef: beta[i],
        se,
        stat: s,
        p: pvalue(s),
        lo: beta[i] - crit * se,
        hi: beta[i] + crit * se,
      };
    });
  }

  function create(jStat) {
    const normCdf = (z) => jStat.normal.cdf(z, 0, 1);
    const normInv = (q) => jStat.normal.inv(q, 0, 1);

    // ---------------------------------------------------------------------------- OLS
    function ols({ y, X, names, robust = false }) {
      const { Z, A, chol, n, p, k } = prepare(X, names);
      const ZtY = new Float64Array(p);
      for (let i = 0; i < n; i++) for (let a = 0; a < p; a++) ZtY[a] += Z[i][a] * y[i];
      const theta = cholSolve(chol, ZtY);
      const bread = cholInverse(chol);
      const yHat = new Float64Array(n);
      const residuals = new Float64Array(n);
      let ssRes = 0;
      let ybar = 0;
      for (let i = 0; i < n; i++) ybar += y[i] / n;
      let ssTot = 0;
      for (let i = 0; i < n; i++) {
        let f = 0;
        for (let a = 0; a < p; a++) f += Z[i][a] * theta[a];
        yHat[i] = f;
        residuals[i] = y[i] - f;
        ssRes += residuals[i] ** 2;
        ssTot += (y[i] - ybar) ** 2;
      }
      if (!(ssTot > 1e-20 * n * Math.max(1, ybar * ybar))) {
        throw new Error("The dependent variable is constant, so there is nothing to explain.");
      }
      const dfModel = k;
      const dfResid = n - p;
      const msResid = ssRes / dfResid;
      let V;
      if (robust) {
        // HC1: bread * sum(e_i^2 z_i z_i') * bread * n / (n - p), as in Stata's ", robust"
        const meat = zeros(p, p);
        for (let i = 0; i < n; i++) {
          const e2 = residuals[i] ** 2;
          for (let a = 0; a < p; a++) for (let b = 0; b <= a; b++) meat[a][b] += e2 * Z[i][a] * Z[i][b];
        }
        for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) meat[a][b] = meat[b][a];
        V = matMul(matMul(bread, meat), bread);
        for (const row of V) for (let b = 0; b < p; b++) row[b] *= n / dfResid;
      } else {
        V = bread.map((row) => row.map((v) => v * msResid));
      }
      // F test that all slopes are zero: classical from sums of squares, robust as a Wald statistic
      let fStat;
      if (robust) {
        const Vs = zeros(k, k);
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) Vs[a][b] = V[a][b];
        const cs = cholesky(Vs, 1e-14);
        if (cs.bad !== undefined) throw new Error("The robust covariance matrix is singular.");
        const sol = cholSolve(cs.L, theta.subarray(0, k));
        let w = 0;
        for (let a = 0; a < k; a++) w += theta[a] * sol[a];
        fStat = w / k;
      } else {
        fStat = (ssTot - ssRes) / dfModel / msResid;
      }
      const fP = Math.max(0, 1 - jStat.centralF.cdf(fStat, dfModel, dfResid));
      const r2 = 1 - ssRes / ssTot;
      const crit = jStat.studentt.inv(0.975, dfResid);
      const pvalue = (t) => 2 * (1 - jStat.studentt.cdf(Math.abs(t), dfResid));
      return {
        model: "ols",
        robust,
        n,
        p,
        dfModel,
        dfResid,
        ssModel: ssTot - ssRes,
        ssResid: ssRes,
        ssTotal: ssTot,
        fStat,
        fP,
        r2,
        adjR2: 1 - ((1 - r2) * (n - 1)) / dfResid,
        rootMSE: Math.sqrt(msResid),
        coefficients: table(names, theta, V, A, pvalue, crit),
        statName: "t",
        y: Array.from(y),
        yHat: Array.from(yHat),
        residuals: Array.from(residuals),
        warnings: [],
      };
    }

    // ---------------------------------------------------------------------------- logit and probit
    // Stable pieces of the standard normal: log Phi(z) and phi(z)/Phi(z), with a continued fraction in the tail.
    const LN_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);
    function mills(t) {
      // Q(t)/phi(t) for t > 0 by the continued fraction 1/(t + 1/(t + 2/(t + 3/(t + ...))))
      let x = t;
      for (let k = 120; k >= 1; k--) x = t + k / x;
      return 1 / x;
    }
    function logPhi(z) {
      if (z > -3) return Math.log(normCdf(z));
      const t = -z;
      return -0.5 * t * t - LN_SQRT_2PI + Math.log(mills(t));
    }
    function phiOverPhi(z) {
      if (z > -3) return Math.exp(-0.5 * z * z - LN_SQRT_2PI) / normCdf(z);
      return 1 / mills(-z);
    }
    const softplus = (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
    const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

    /** Log likelihood, score and observed information at theta (standardised scale). */
    function evaluate(Z, y, theta, link) {
      const n = Z.length;
      const p = theta.length;
      const g = new Float64Array(p);
      const H = zeros(p, p);
      let ll = 0;
      for (let i = 0; i < n; i++) {
        const z = Z[i];
        let eta = 0;
        for (let a = 0; a < p; a++) eta += z[a] * theta[a];
        let gi;
        let hi;
        if (link === "logit") {
          const pr = sigmoid(eta);
          ll += y[i] * eta - softplus(eta);
          gi = y[i] - pr;
          hi = pr * (1 - pr);
        } else {
          const q = 2 * y[i] - 1;
          ll += logPhi(q * eta);
          const lam = q * phiOverPhi(q * eta);
          gi = lam;
          hi = lam * (lam + eta);
        }
        for (let a = 0; a < p; a++) {
          g[a] += gi * z[a];
          for (let b = 0; b <= a; b++) H[a][b] += hi * z[a] * z[b];
        }
      }
      for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) H[a][b] = H[b][a];
      return { ll, g, H };
    }

    function binary({ y, X, names, link }) {
      let n1 = 0;
      for (const v of y) {
        if (v !== 0 && v !== 1) {
          const seen = [...new Set(y)].slice(0, 6).join(", ");
          throw new Error(
            `${link === "logit" ? "Logit" : "Probit"} needs a dependent variable coded 0/1; found values such as ${seen}.`,
          );
        }
        n1 += v;
      }
      const n = y.length;
      const n0 = n - n1;
      if (n1 === 0 || n0 === 0)
        throw new Error("The dependent variable has only one value (all 0 or all 1).");
      const { Z, A, p, k } = prepare(X, names);
      const ybar = n1 / n;
      let theta = new Float64Array(p);
      theta[k] = link === "logit" ? Math.log(ybar / (1 - ybar)) : normInv(ybar);
      let cur = evaluate(Z, y, theta, link);
      let converged = false;
      let iterations = 0;
      const warnings = [];
      for (let iter = 1; iter <= 100; iter++) {
        iterations = iter;
        const ch = cholesky(cur.H, 1e-12);
        if (ch.bad !== undefined) {
          warnings.push("The information matrix became singular; the estimates may be unreliable.");
          break;
        }
        const delta = cholSolve(ch.L, cur.g);
        let step = 1;
        let next;
        let cand;
        for (let h = 0; h < 40; h++) {
          cand = theta.map((t, a) => t + step * delta[a]);
          next = evaluate(Z, y, cand, link);
          if (next.ll >= cur.ll - 1e-12) break;
          step /= 2;
        }
        let maxStep = 0;
        for (let a = 0; a < p; a++) maxStep = Math.max(maxStep, Math.abs(cand[a] - theta[a]));
        const dLL = Math.abs(next.ll - cur.ll);
        theta = cand;
        cur = next;
        if (maxStep < 1e-9 || dLL < 1e-13 * (1 + Math.abs(cur.ll))) {
          converged = true;
          break;
        }
      }
      if (cur.ll > -1e-5) {
        converged = false;
        warnings.push(
          "Perfect prediction: some combination of the variables separates the 0s from the 1s, " +
            "so the coefficients are not finite. Remove or combine variables.",
        );
      } else if (!converged) {
        warnings.push("The estimation did not converge in 100 iterations; treat the results with caution.");
      }
      const ch = cholesky(cur.H, 1e-14);
      if (ch.bad !== undefined)
        throw new Error("The information matrix is singular, so standard errors cannot be computed.");
      const V = cholInverse(ch.L);
      const crit = normInv(0.975);
      const pvalue = (z) => 2 * (1 - normCdf(Math.abs(z)));
      const coefficients = table(names, theta, V, A, pvalue, crit);
      if (link === "logit") {
        for (const c of coefficients) {
          c.oddsRatio = Math.exp(c.coef);
          c.orSe = Math.exp(c.coef) * c.se;
          c.orLo = Math.exp(c.lo);
          c.orHi = Math.exp(c.hi);
        }
      }
      const yHat = new Float64Array(n);
      let correct = 0;
      for (let i = 0; i < n; i++) {
        let eta = 0;
        for (let a = 0; a < p; a++) eta += Z[i][a] * theta[a];
        yHat[i] = link === "logit" ? sigmoid(eta) : normCdf(eta);
        if ((yHat[i] >= 0.5 ? 1 : 0) === y[i]) correct++;
      }
      const ll0 = n1 * Math.log(n1 / n) + n0 * Math.log(n0 / n);
      const lr = 2 * (cur.ll - ll0);
      return {
        model: link,
        n,
        n1,
        n0,
        p,
        dfModel: k,
        logLik: cur.ll,
        logLikNull: ll0,
        lrChi2: lr,
        lrP: 1 - jStat.chisquare.cdf(lr, k),
        pseudoR2: 1 - cur.ll / ll0,
        aic: -2 * cur.ll + 2 * p,
        bic: -2 * cur.ll + p * Math.log(n),
        accuracy: correct / n,
        iterations,
        converged,
        coefficients,
        statName: "z",
        y: Array.from(y),
        yHat: Array.from(yHat),
        residuals: Array.from(y, (v, i) => v - yHat[i]),
        warnings,
      };
    }

    /** Fit the model chosen in the interface. `options.robust` applies to OLS only. */
    function fit(model, y, X, names, options = {}) {
      if (model === "ols") return ols({ y, X, names, robust: !!options.robust });
      if (model === "logit" || model === "probit") return binary({ y, X, names, link: model });
      throw new Error(`Unknown model "${model}".`);
    }

    return { fit, ols, binary, logPhi, phiOverPhi };
  }

  return { create, toNumber, classifyColumns, designMatrix };
});
