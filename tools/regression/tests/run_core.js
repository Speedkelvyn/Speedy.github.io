// Runs one job from tests/test_regression_js.py: reads JSON on stdin, prints JSON on stdout.
const { create, designMatrix, classifyColumns, toNumber } = require("../regression.js");
const jStat = require("jstat");

const core = create(jStat);
let input = "";
process.stdin.on("data", (c) => (input += c));
process.stdin.on("end", () => {
  const job = JSON.parse(input);
  try {
    let out;
    if (job.task === "fit") {
      const t0 = Date.now();
      out = core.fit(job.model, job.y, job.X, job.names, job.options || {});
      out.seconds = (Date.now() - t0) / 1000;
    } else if (job.task === "design") {
      out = designMatrix(job.rows, job.yName, job.xNames);
    } else if (job.task === "classify") {
      out = classifyColumns(job.rows, job.headers);
    } else if (job.task === "toNumber") {
      out = job.values.map((v) => {
        const x = toNumber(v);
        return Number.isNaN(x) ? "text" : x;
      });
    } else if (job.task === "tails") {
      out = job.z.map((z) => [core.logPhi(z), core.phiOverPhi(z)]);
    }
    process.stdout.write(JSON.stringify({ ok: true, out }));
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: e.message }));
  }
});
