"""The Python engine must reproduce the figures shown on the web pages."""

import math
import pathlib
import sys

import pytest

ENGINE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
import run_valuation  # noqa: E402
import valuation_v2 as v  # noqa: E402

# Base-case figures published on the site (computed by the page's JavaScript engine)
PAGE = {
    "dell": {
        "dcf": 284.73015228872623,
        "bear": 145.5897076500473,
        "bull": 411.8835552760559,
        "wacc": 0.11581621085873567,
        "ke": 0.1214470451705196,
        "comps": 369.40631186509967,
        "call": "OVERVALUED",
        "fair": 311.65,
    },
    "sndk": {
        "dcf": 1476.017628458122,
        "bear": 627.4729269294703,
        "bull": 2367.264769152028,
        "wacc": 0.14584937103055423,
        "ke": 0.14593335919016975,
        "comps": 1153.3493304093302,
        "call": "FAIRLY VALUED",
        "fair": 1315.59,
    },
}


@pytest.fixture(scope="module", params=sorted(PAGE))
def case(request):
    inp = run_valuation.load(request.param)
    return request.param, inp, v.run_all(inp, "Base", mc_n=1500)


def test_inputs_pass_checks(case):
    _, _, R = case
    assert R["checks"]["ok"]


def test_dcf_and_wacc_match_page_exactly(case):
    name, _, R = case
    P = PAGE[name]
    assert R["base"]["perShare"] == pytest.approx(P["dcf"], rel=1e-9)
    assert R["scen3"]["Bear"]["perShare"] == pytest.approx(P["bear"], rel=1e-9)
    assert R["scen3"]["Bull"]["perShare"] == pytest.approx(P["bull"], rel=1e-9)
    assert R["W"]["wacc"] == pytest.approx(P["wacc"], rel=1e-9)
    assert R["W"]["ke"] == pytest.approx(P["ke"], rel=1e-9)
    assert R["rel"]["blended"] == pytest.approx(P["comps"], rel=1e-9)


def test_verdict_matches_page(case):
    name, _, R = case
    assert R["verdict"]["call"] == PAGE[name]["call"]
    # Monte Carlo sampler differs from the page's, so allow a small gap on the blend
    assert R["verdict"]["fair"] == pytest.approx(PAGE[name]["fair"], rel=0.01)


def test_cli_runs(capsys):
    assert run_valuation.main(["sndk", "--mc", "500"]) == 0
    out = capsys.readouterr().out
    assert "VERDICT" in out and "Sandisk" in out


def test_missing_input_blocks_valuation():
    inp = run_valuation.load("dell")
    inp["capital"]["rf"] = None
    R = v.run_all(inp)
    assert not R["checks"]["ok"]
    assert math.isnan(R["verdict"].get("fair", float("nan"))) or R["verdict"]["call"] in ("NOT VALUED", None)
