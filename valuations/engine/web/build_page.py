"""build_page.py: build an interactive valuation page from an inputs file.

    python build_page.py ../../dell/dell_inputs.json ../../dell/index.html
    python build_page.py ../../sndk/SNDK_inputs.json ../../sndk/index.html
    python build_page.py AAPL_inputs.json aapl.html --inline     # one self-contained file

By default the page loads the engine (model.js) and the research module (research.js)
from this folder, so every company page on the site shares one copy of the code.
With --inline both are embedded, which gives a single file you can open anywhere.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ENGINE_FILES = ("model.js", "research.js")
SITE_LINKS = (
    '<nav class="site-links" aria-label="Site">\n'
    '  <a href="https://github.com/Speedkelvyn/Speedy.github.io/tree/main/valuations/engine"'
    ' target="_blank" rel="noopener">Python engine ↗</a>\n'
    '  <a href="../../">← Portfolio</a>\n'
    "</nav>"
)


def build(inputs: dict, out: pathlib.Path, inline: bool = False) -> str:
    meta = inputs["meta"]
    if inline:
        scripts = "\n".join(
            f"<script>\n{(HERE / f).read_text(encoding='utf-8')}\n</script>" for f in ENGINE_FILES
        )
        links = ""
    else:
        rel = pathlib.PurePosixPath(os.path.relpath(HERE, out.resolve().parent).replace(os.sep, "/"))
        scripts = "\n".join(f'<script src="{rel / f}"></script>' for f in ENGINE_FILES)
        links = SITE_LINKS
    data = json.dumps(inputs, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    page = (
        (HERE / "template.html")
        .read_text(encoding="utf-8")
        .replace("{{TITLE}}", html.escape(f"{meta['name']} ({meta['ticker']}) · Valuation"))
        .replace(
            "{{DESCRIPTION}}",
            html.escape(
                f"Interactive DCF valuation of {meta['name']} ({meta['ticker']}) as of {meta['valuation_date']}."
            ),
        )
        .replace("{{ENGINE_SCRIPTS}}", scripts)
        .replace("{{SITE_LINKS}}", links)
        .replace("{{DATA}}", data)
    )
    out.write_text(page, encoding="utf-8")
    return page


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--inline", action="store_true", help="embed the engine for a single self-contained file")
    a = ap.parse_args(argv)
    page = build(json.loads(a.inputs.read_text(encoding="utf-8")), a.out, a.inline)
    print(f"wrote {a.out} ({len(page) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
