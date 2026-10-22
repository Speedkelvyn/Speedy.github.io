"""build_single.py: write regression_tool.html, one self-contained file (still loads its libraries from a CDN).

python build_single.py [output.html]

Inlines regression.js, app.js and sample.csv, so the file works when opened from disk, e.g. from a course folder.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "regression_tool.html"
page = (HERE / "index.html").read_text(encoding="utf-8")
sample = (HERE / "sample.csv").read_text(encoding="utf-8").replace("</", "<\\/")
for name in ("regression.js", "app.js"):
    tag = f'<script src="{name}"></script>'
    assert page.count(tag) == 1, tag
    page = page.replace(tag, f"<script>\n{(HERE / name).read_text(encoding='utf-8')}\n</script>")
page = page.replace("</body>", f'<script type="text/plain" id="sampleData">\n{sample}</script>\n</body>')
page = page.replace('href="../../"', 'href="https://speedkelvyn.github.io/Speedy.github.io/"')
out.write_text(page, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
