# Speedy Regression Tool

OLS, logit and probit on any CSV file, in the browser. The file is read locally and never uploaded.

**Use it:** https://speedkelvyn.github.io/Speedy.github.io/tools/regression/ (press "Load sample data" to try it)

## What it does

- Reads a CSV, shows the first rows, descriptive statistics and a missing-value summary.
- **OLS** with classical or heteroskedasticity-robust (HC1, Stata's `, robust`) standard errors, F test, R², adjusted R², root MSE.
- **Logit and probit** by Newton-Raphson maximum likelihood: likelihood-ratio test, McFadden pseudo R², AIC, BIC, share correctly classified, odds ratios for logit.
- Warns about perfect prediction and non-convergence instead of printing nonsense; names the variable behind perfect multicollinearity.
- Four diagnostic plots (actual vs fitted, residuals vs fitted, Q-Q, histogram), light and dark themes, CSV download of the coefficient table.

## Checked against statsmodels

`tests/test_regression_js.py` runs the core in Node and compares coefficients, standard errors, t/z statistics, p-values, confidence intervals, F and LR tests, log likelihood, pseudo R², AIC and BIC with statsmodels (OLS, robust OLS, logit, probit), including badly scaled regressors, probit observations far in the tail, perfect separation and 50,000 rows. The page itself was driven in headless Chromium against the same references.

```bash
npm install jstat@1.9.6          # the only dependency of the core
NODE_PATH=$PWD/node_modules pytest -q tests
```

## What changed from the first version

The first version (`regression_tool.html`, kept in a course folder) had these problems, all fixed here:

| Problem | Effect |
|---|---|
| Coefficients were labelled in the wrong order: the intercept sat in the first row but was printed under the first variable's name, and the last variable under `_cons` | Every table was mislabelled |
| Blank cells were turned into 0 (`Number(null)`) | Missing data entered the regression as zeros; the "Missing" count always said 0 |
| Logit and probit used 100 fixed gradient steps of size 0.01 | Fine on small, well-scaled data, but a regressor ten times larger gave a coefficient of −41.8 instead of −0.11 |
| Probit standard errors used the logit weights | About 1.8 times too large on the sample data |
| An n by n weight matrix was built for the standard errors | Froze the browser on a few thousand rows |
| Confidence intervals used 1.96 for OLS | Too narrow in small samples |
| Perfect collinearity, text columns, a non-binary outcome or a repeated variable crashed silently or gave garbage | Now a clear message |
| `plotly-latest` and `jstat@latest` | Unpinned; now pinned with integrity hashes |
| Header names were inserted as HTML | Escaped |

## Files

| File | Contents |
|---|---|
| `index.html`, `app.js` | The page |
| `regression.js` | Numerical core: reading values, complete-case design, OLS, logit, probit (no DOM access) |
| `sample.csv`, `make_sample.py` | Sample data from the recession-risk study and the script that builds it |
| `tests/` | Comparison with statsmodels |
