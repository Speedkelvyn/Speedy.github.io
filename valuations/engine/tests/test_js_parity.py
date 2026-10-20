"""The browser engine (web/model.js) and the Python engine must agree."""

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ENGINE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
import run_valuation  # noqa: E402
import valuation_v2 as v  # noqa: E402

NODE = shutil.which("node")
JS = """
const M = require(process.argv[1]);
const I = JSON.parse(require("fs").readFileSync(0, "utf8"));
const R = M.runAll(I, "Base", { n: 200, seed: 7 });
const s = {};
for (const k of Object.keys(R.scen3)) s[k] = R.scen3[k].perShare;
console.log(JSON.stringify({ wacc: R.W.wacc, ke: R.W.ke, base: R.base.perShare, scen: s, comps: R.rel.blended }));
"""


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
@pytest.mark.parametrize("company", ["dell", "sndk"])
def test_python_and_javascript_engines_agree(company):
    inp = run_valuation.load(company)
    out = subprocess.run(
        [NODE, "-e", JS, str(ENGINE / "web" / "model.js")],
        input=json.dumps(inp),
        capture_output=True,
        text=True,
        check=True,
    )
    js = json.loads(out.stdout)
    py = v.run_all(inp, "Base", mc_n=200)
    assert py["W"]["wacc"] == pytest.approx(js["wacc"], rel=1e-12)
    assert py["W"]["ke"] == pytest.approx(js["ke"], rel=1e-12)
    assert py["base"]["perShare"] == pytest.approx(js["base"], rel=1e-12)
    for k, val in js["scen"].items():
        assert py["scen3"][k]["perShare"] == pytest.approx(val, rel=1e-12)
    assert py["rel"]["blended"] == pytest.approx(js["comps"], rel=1e-12)
