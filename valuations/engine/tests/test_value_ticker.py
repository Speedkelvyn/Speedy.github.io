"""value_ticker.py: the one-command pipeline (fetch, build inputs, value, write page and workbook)."""

import copy
import json
import pathlib
import re
import sys

import pytest

ENGINE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
import run_valuation  # noqa: E402
import value_ticker  # noqa: E402


def fake_feeds(ticker, peers=None):
    """What fetch_yahoo.fetch returns, with empty feeds: the fake builder below ignores them."""
    return {
        "ticker": ticker, "today": "2026-10-21", "tnx": 4.0, "summary": {"seen_peers": peers}, "annual": {},
        "quarterly": {}, "stock": [], "index": [], "peers": {},
    }  # fmt: skip


@pytest.fixture
def inputs():
    return run_valuation.load("dell")


def builder_for(inp):
    seen = {}

    def build(feeds):
        seen["feeds"] = feeds
        return copy.deepcopy(inp)

    build.seen = seen
    return build


def page_data(path: pathlib.Path) -> dict:
    html = path.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="valuation-data">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_one_command_writes_inputs_snapshot_page_and_workbook(tmp_path, inputs, capsys):
    b = builder_for(inputs)
    _, R, files = value_ticker.value("dell", out=tmp_path, mc=300, fetcher=fake_feeds, builder=b)
    assert R["checks"]["ok"] and set(files) == {"snapshot", "inputs", "page", "workbook"}
    assert all(p.exists() and p.parent == tmp_path for p in files.values())
    assert files["inputs"].name == "DELL_inputs.json" and json.loads(files["inputs"].read_text()) == inputs
    assert files["snapshot"].name == "DELL_yahoo_2026-10-21.json"
    assert json.loads(files["snapshot"].read_text())["format"] == "compact-v1"
    assert page_data(files["page"]) == inputs  # the page carries the same inputs
    assert "VERDICT" in capsys.readouterr().out
    import openpyxl

    ws = openpyxl.load_workbook(files["workbook"]).worksheets
    formulas = sum(
        1
        for s in ws
        for row in s.iter_rows()
        for c in row
        if isinstance(c.value, str) and c.value.startswith("=")
    )
    assert formulas > 100  # live formulas, not pasted numbers
    assert b.seen["feeds"]["ticker"] == "DELL"


def test_flags_skip_the_page_and_the_workbook(tmp_path, inputs):
    _, _, files = value_ticker.value(
        "dell", out=tmp_path, mc=200, excel=False, page=False, fetcher=fake_feeds, builder=builder_for(inputs)
    )
    assert set(files) == {"snapshot", "inputs"}


def test_peers_are_passed_to_the_fetcher_and_cleaned(tmp_path, inputs):
    b = builder_for(inputs)
    value_ticker.value(
        "dell",
        peers=["hpq", "NTAP"],
        out=tmp_path,
        mc=200,
        excel=False,
        page=False,
        fetcher=fake_feeds,
        builder=b,
    )
    assert b.seen["feeds"]["summary"]["seen_peers"] == ["HPQ", "NTAP"]


def test_rerun_from_a_saved_snapshot_needs_no_network(tmp_path, inputs):
    snap = tmp_path / "DELL_yahoo_2026-10-21.json"
    snap.write_text(
        json.dumps({"format": "compact-v1", **{k: v for k, v in fake_feeds("DELL").items() if k != "built"}})
    )
    b = builder_for(inputs)

    def no_network(*_):
        raise AssertionError("fetch must not be called")

    _, R, files = value_ticker.value(
        snapshot=snap, out=tmp_path / "out", mc=200, fetcher=no_network, builder=b, excel=False
    )
    assert R["checks"]["ok"] and "snapshot" not in files  # the snapshot already exists
    assert b.seen["feeds"]["ticker"] == "DELL" and b.seen["feeds"]["annual"] == {}


def test_missing_inputs_exit_with_status_2_but_still_write_files(tmp_path, inputs, monkeypatch, capsys):
    broken = copy.deepcopy(inputs)
    broken["capital"]["rf"] = None
    monkeypatch.setattr(value_ticker.RP, "build", lambda feeds: broken)
    monkeypatch.setitem(sys.modules, "fetch_yahoo", type(sys)("fetch_yahoo"))
    sys.modules["fetch_yahoo"].fetch = fake_feeds
    sys.modules["fetch_yahoo"].save_snapshot = lambda feeds, path: pathlib.Path(path).write_text("{}")
    assert value_ticker.main(["dell", "--out", str(tmp_path), "--mc", "200"]) == 2
    out = capsys.readouterr().out
    assert "NOT VALUED" in out and "fix the items listed above" in out
    assert (tmp_path / "DELL_inputs.json").exists() and (tmp_path / "index.html").exists()
    assert not (tmp_path / "DELL_valuation.xlsx").exists()


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", "TOOLONGSYMBOL1", "x y", ".hidden"])
def test_unsafe_tickers_are_rejected(bad):
    with pytest.raises(SystemExit):
        value_ticker.clean_ticker(bad)


def test_class_shares_and_default_folder():
    assert value_ticker.clean_ticker("brk-b") == "BRK-B"
    assert value_ticker.default_folder("BRK-B").name == "brk-b"
    assert value_ticker.default_folder("BF.B").name == "bf-b"
    assert (
        value_ticker.default_folder("AAPL").parent == ENGINE.parent
    )  # valuations/aapl, next to dell and sndk


def test_need_a_ticker_or_a_snapshot():
    with pytest.raises(SystemExit):
        value_ticker.value()
    with pytest.raises(SystemExit):
        value_ticker.value("AAPL", snapshot=pathlib.Path("x.json"))
