/* app.js: interface for the Speedy Regression Tool. The statistics live in regression.js. */
(() => {
  "use strict";

  const core = RegressionCore.create(jStat);
  const { toNumber, classifyColumns, designMatrix } = RegressionCore;
  const $ = (id) => document.getElementById(id);
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

  const state = { rows: [], headers: [], numeric: [], text: [], result: null, fileName: "data" };

  // ------------------------------------------------------------------------------ formatting
  const fmt = (x, d = 4) => (Number.isFinite(x) ? x.toFixed(d) : "–");
  const fmtP = (p) => (Number.isFinite(p) ? (p < 0.0005 ? "0.000" : p.toFixed(3)) : "–");
  /** Fixed decimals for ordinary sizes, scientific notation for very large or very small coefficients. */
  const fmtCoef = (x) => {
    if (!Number.isFinite(x)) return "–";
    if (x === 0) return "0";
    const a = Math.abs(x);
    return a >= 1e5 || a < 1e-3 ? x.toExponential(3) : x.toFixed(4);
  };
  const fmtCell = (v) => (v === null || v === undefined || v === "" ? "·" : esc(v));

  function say(text, kind = "danger") {
    $("message").innerHTML = text ? `<div class="alert alert-${kind}">${esc(text)}</div>` : "";
  }

  // ------------------------------------------------------------------------------ theme
  function setTheme(theme) {
    document.documentElement.setAttribute("data-bs-theme", theme);
    if (state.result) drawPlot();
  }
  setTheme(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  $("darkModeToggle").addEventListener("click", () => {
    setTheme(document.documentElement.getAttribute("data-bs-theme") === "dark" ? "light" : "dark");
  });

  // ------------------------------------------------------------------------------ loading data
  function loadCsv(source, name) {
    say("");
    Papa.parse(source, {
      header: true,
      skipEmptyLines: "greedy",
      complete: (results) => {
        const headers = (results.meta.fields || []).filter((h) => h !== "");
        if (results.data.length === 0 || headers.length === 0) {
          say("No data found. Check that the file is a CSV with a header row.");
          return;
        }
        state.rows = results.data;
        state.headers = headers;
        state.fileName = name.replace(/\.[^.]+$/, "");
        state.result = null;
        const { numeric, text } = classifyColumns(state.rows, headers);
        state.numeric = numeric;
        state.text = text;
        ["resultsCard", "plotCard"].forEach((id) => $(id).classList.add("d-none"));
        $("datasetOverviewCard").classList.remove("d-none");
        showOverview();
        if (numeric.length < 2) {
          ["varCard", "modelCard"].forEach((id) => $(id).classList.add("d-none"));
          say("At least two numeric columns are needed for a regression.");
          return;
        }
        setupVariables();
        if (results.errors.length > 0) {
          say(
            `The file was read, but ${results.errors.length} row(s) looked malformed (${results.errors[0].message}).`,
            "warning",
          );
        }
      },
      error: (err) => say(`Could not read the file: ${err.message}`),
    });
  }

  $("csvFile").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) loadCsv(file, file.name);
  });

  $("sampleBtn").addEventListener("click", () => {
    fetch("sample.csv")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => loadCsv(text, "sample.csv"))
      .catch(() =>
        say(
          "The sample file can only be loaded when this page is opened from a web server, not from a local file.",
        ),
      );
  });

  function setupVariables() {
    const options = state.numeric.map((h) => `<option value="${esc(h)}">${esc(h)}</option>`).join("");
    $("dependentVar").innerHTML = options;
    $("independentVarsContainer").innerHTML = "";
    addVariable();
    ["varCard", "modelCard"].forEach((id) => $(id).classList.remove("d-none"));
    $("excludedNote").textContent =
      state.text.length > 0 ? `Not offered (text or empty columns): ${state.text.join(", ")}.` : "";
    updateRobustVisibility();
  }

  function chosenX() {
    return Array.from(document.querySelectorAll("#independentVarsContainer select")).map((s) => s.value);
  }

  function addVariable() {
    const taken = new Set([$("dependentVar").value, ...chosenX()]);
    const free = state.numeric.find((h) => !taken.has(h));
    const wrap = document.createElement("div");
    wrap.className = "d-flex mb-2 align-items-center gap-2";
    const select = document.createElement("select");
    select.className = "form-select";
    select.setAttribute("aria-label", "Independent variable");
    select.innerHTML = state.numeric.map((h) => `<option value="${esc(h)}">${esc(h)}</option>`).join("");
    select.value = free !== undefined ? free : state.numeric[0];
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn btn-outline-danger btn-sm";
    remove.textContent = "×";
    remove.setAttribute("aria-label", "Remove this variable");
    remove.addEventListener("click", () => wrap.remove());
    wrap.append(select, remove);
    $("independentVarsContainer").appendChild(wrap);
  }
  $("addVarBtn").addEventListener("click", addVariable);

  function updateRobustVisibility() {
    $("robustWrap").classList.toggle("d-none", $("modelSelect").value !== "ols");
  }
  $("modelSelect").addEventListener("change", updateRobustVisibility);

  // ------------------------------------------------------------------------------ dataset overview
  function showOverview() {
    const { rows, headers, numeric } = state;
    $("firstRowsTable").querySelector("thead").innerHTML =
      "<tr>" + headers.map((h) => `<th scope="col">${esc(h)}</th>`).join("") + "</tr>";
    $("firstRowsTable").querySelector("tbody").innerHTML = rows
      .slice(0, 5)
      .map((r) => "<tr>" + headers.map((h) => `<td>${fmtCell(r[h])}</td>`).join("") + "</tr>")
      .join("");

    let desc = "";
    let miss = "";
    for (const h of headers) {
      const isNumeric = numeric.includes(h);
      let filled = 0;
      let sum = 0;
      let min = Infinity;
      let max = -Infinity;
      const vals = [];
      for (const r of rows) {
        if (isNumeric) {
          const v = toNumber(r[h]);
          if (v === null) continue;
          vals.push(v);
          sum += v;
          if (v < min) min = v;
          if (v > max) max = v;
        } else if (String(r[h] ?? "").trim() !== "") {
          filled++;
        }
      }
      if (isNumeric) {
        filled = vals.length;
        const mean = sum / filled;
        const sd = filled > 1 ? Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / (filled - 1)) : NaN;
        desc += `<tr><td>${esc(h)}</td><td class="num">${filled}</td><td class="num">${fmtCoef(mean)}</td><td class="num">${fmtCoef(sd)}</td><td class="num">${fmtCoef(min)}</td><td class="num">${fmtCoef(max)}</td></tr>`;
      }
      miss += `<tr><td>${esc(h)}</td><td>${isNumeric ? "numeric" : "text"}</td><td class="num">${filled}</td><td class="num">${rows.length - filled}</td></tr>`;
    }
    $("descStatsTable").querySelector("tbody").innerHTML = desc;
    $("missingTable").querySelector("tbody").innerHTML = miss;
    const complete = rows.filter((r) =>
      headers.every((h) =>
        numeric.includes(h) ? toNumber(r[h]) !== null : String(r[h] ?? "").trim() !== "",
      ),
    ).length;
    $("completeCase").innerHTML =
      `<b>Rows with no missing value in any column:</b> ${complete} of ${rows.length}`;
  }

  // ------------------------------------------------------------------------------ running a model
  function run() {
    say("");
    const yName = $("dependentVar").value;
    const xNames = chosenX();
    const model = $("modelSelect").value;
    if (xNames.length === 0) return say("Add at least one independent variable.");
    if (new Set([yName, ...xNames]).size !== xNames.length + 1) {
      return say(
        "Every variable must be different: the dependent variable cannot also be an independent variable, and none can be chosen twice.",
      );
    }
    let design;
    let res;
    try {
      design = designMatrix(state.rows, yName, xNames);
      res = core.fit(model, design.y, design.X, xNames, { robust: $("robustSE").checked });
    } catch (e) {
      state.result = null;
      ["resultsCard", "plotCard"].forEach((id) => $(id).classList.add("d-none"));
      return say(e.message);
    }
    state.result = { res, design, yName, xNames, model, robust: $("robustSE").checked && model === "ols" };
    renderResults();
    ["resultsCard", "plotCard"].forEach((id) => $(id).classList.remove("d-none"));
    drawPlot();
  }
  $("runBtn").addEventListener("click", run);

  function statList(res) {
    if (res.model === "ols") {
      return [
        ["Number of obs", res.n],
        [`F(${res.dfModel}, ${res.dfResid})`, fmt(res.fStat, 2)],
        ["Prob > F", fmtP(res.fP)],
        ["R-squared", fmt(res.r2)],
        ["Adj R-squared", fmt(res.adjR2)],
        ["Root MSE", fmtCoef(res.rootMSE)],
      ];
    }
    return [
      ["Number of obs", res.n],
      ["Outcome = 1 / = 0", `${res.n1} / ${res.n0}`],
      [`LR chi2(${res.dfModel})`, fmt(res.lrChi2, 2)],
      ["Prob > chi2", fmtP(res.lrP)],
      ["Log likelihood", fmt(res.logLik, 3)],
      ["Pseudo R2 (McFadden)", fmt(res.pseudoR2)],
      ["AIC", fmt(res.aic, 2)],
      ["BIC", fmt(res.bic, 2)],
      ["Correctly classified", `${fmt(100 * res.accuracy, 1)}%`],
      ["Iterations", res.iterations],
    ];
  }

  function renderResults() {
    const { res, design, yName, model, robust } = state.result;
    const names = { ols: "OLS", logit: "Logit", probit: "Probit" };
    $("resultTitle").textContent =
      `${names[model]}${robust ? " with robust (HC1) standard errors" : ""}: ${yName} on ${state.result.xNames.join(", ")}`;

    let notes = "";
    if (design.nDropped > 0) {
      const detail = Object.entries(design.missingBy)
        .filter(([, c]) => c > 0)
        .map(([name, c]) => `${esc(name)}: ${c}`)
        .join(", ");
      notes += `<div class="alert alert-info py-2">Used ${design.nUsed} of ${design.nTotal} rows. ${design.nDropped} left out because of missing values (${detail}).</div>`;
    }
    for (const w of res.warnings) notes += `<div class="alert alert-warning py-2">${esc(w)}</div>`;
    $("resultNotes").innerHTML = notes;

    $("statGrid").innerHTML = statList(res)
      .map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`)
      .join("");

    const stat = res.statName;
    const isLogit = model === "logit";
    $("coefTable").querySelector("thead").innerHTML =
      `<tr><th scope="col">Variable</th><th scope="col" class="num">Coef.</th><th scope="col" class="num">Std. err.</th><th scope="col" class="num">${stat}</th><th scope="col" class="num">P&gt;|${stat}|</th><th scope="col" class="num">[95% conf.</th><th scope="col" class="num">interval]</th>` +
      (isLogit ? '<th scope="col" class="num">Odds ratio</th>' : "") +
      "</tr>";
    $("coefTable").querySelector("tbody").innerHTML = res.coefficients
      .map(
        (c) =>
          `<tr><td>${esc(c.name)}</td><td class="num">${fmtCoef(c.coef)}</td><td class="num">${fmtCoef(c.se)}</td><td class="num">${fmt(c.stat, 2)}</td><td class="num">${fmtP(c.p)}</td><td class="num">${fmtCoef(c.lo)}</td><td class="num">${fmtCoef(c.hi)}</td>` +
          (isLogit ? `<td class="num">${fmtCoef(c.oddsRatio)}</td>` : "") +
          "</tr>",
      )
      .join("");
    $("coefNote").textContent =
      model === "ols"
        ? `Confidence intervals use the t distribution with ${res.dfResid} degrees of freedom. _cons is the intercept.`
        : "Maximum likelihood by Newton-Raphson with the observed information matrix; z statistics and normal confidence intervals. _cons is the intercept.";
  }

  $("downloadBtn").addEventListener("click", () => {
    if (!state.result) return;
    const { res, model, yName } = state.result;
    const head = ["variable", "coef", "std_err", res.statName, "p_value", "ci_low", "ci_high"];
    if (model === "logit") head.push("odds_ratio");
    const lines = [head.join(",")];
    for (const c of res.coefficients) {
      const row = [c.name, c.coef, c.se, c.stat, c.p, c.lo, c.hi];
      if (model === "logit") row.push(c.oddsRatio);
      lines.push(row.map((v) => (typeof v === "string" ? `"${v.replace(/"/g, '""')}"` : v)).join(","));
    }
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.fileName}_${model}_${yName}.csv`.replace(/[^\w.-]+/g, "_");
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  });

  // ------------------------------------------------------------------------------ plots
  const range = (arr) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of arr) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return [lo, hi];
  };

  function drawPlot() {
    if (!state.result) return;
    const { res } = state.result;
    const dark = document.documentElement.getAttribute("data-bs-theme") === "dark";
    const fg = dark ? "#f1f1f1" : "#212529";
    const grid = dark ? "#444" : "#e5e5e5";
    const layout = {
      margin: { t: 50, l: 60, r: 30, b: 55 },
      showlegend: false,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: fg },
      xaxis: { gridcolor: grid, zerolinecolor: grid },
      yaxis: { gridcolor: grid, zerolinecolor: grid },
    };
    const kind = $("plotSelect").value;
    const line = { color: "rgba(220,53,69,0.9)", dash: "dash" };
    let traces = [];
    if (kind === "actualFitted") {
      const [lo, hi] = range([...res.y, ...res.yHat]);
      traces = [
        {
          x: res.yHat,
          y: res.y,
          mode: "markers",
          type: "scatter",
          marker: { color: "rgba(13,110,253,0.6)", size: 7 },
        },
        { x: [lo, hi], y: [lo, hi], mode: "lines", type: "scatter", line },
      ];
      Object.assign(layout, { title: "Actual vs. fitted values" });
      layout.xaxis.title = "Fitted values";
      layout.yaxis.title = "Actual values";
    } else if (kind === "residualsFitted") {
      const [lo, hi] = range(res.yHat);
      traces = [
        {
          x: res.yHat,
          y: res.residuals,
          mode: "markers",
          type: "scatter",
          marker: { color: "rgba(255,193,7,0.8)", size: 7 },
        },
        { x: [lo, hi], y: [0, 0], mode: "lines", type: "scatter", line },
      ];
      layout.title = "Residuals vs. fitted values";
      layout.xaxis.title = "Fitted values";
      layout.yaxis.title = "Residuals";
    } else if (kind === "qq") {
      const sorted = Float64Array.from(res.residuals).sort();
      const n = sorted.length;
      const theo = Array.from(sorted, (_, i) => jStat.normal.inv((i + 0.5) / n, 0, 1));
      const q = (p) => sorted[Math.min(n - 1, Math.floor(p * n))];
      const slope = (q(0.75) - q(0.25)) / (jStat.normal.inv(0.75, 0, 1) - jStat.normal.inv(0.25, 0, 1));
      const icpt = q(0.25) - slope * jStat.normal.inv(0.25, 0, 1);
      traces = [
        {
          x: theo,
          y: Array.from(sorted),
          mode: "markers",
          type: "scatter",
          marker: { color: "rgba(25,135,84,0.7)", size: 6 },
        },
        {
          x: [theo[0], theo[n - 1]],
          y: [icpt + slope * theo[0], icpt + slope * theo[n - 1]],
          mode: "lines",
          type: "scatter",
          line,
        },
      ];
      layout.title = "Normal Q-Q plot of residuals";
      layout.xaxis.title = "Theoretical quantiles";
      layout.yaxis.title = "Sample quantiles (residuals)";
    } else {
      traces = [{ x: res.residuals, type: "histogram", marker: { color: "rgba(108,117,125,0.8)" } }];
      layout.title = "Histogram of residuals";
      layout.xaxis.title = "Residuals";
      layout.yaxis.title = "Frequency";
    }
    Plotly.react("plot", traces, layout, { responsive: true, displaylogo: false });
  }
  $("plotSelect").addEventListener("change", drawPlot);
})();
