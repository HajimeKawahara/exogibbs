"""Keep the standalone viewer snapshot aligned with the packaged catalog."""

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "documents/_static/package-graph.html"
CATALOG = ROOT / "src/exogibbs/data/mceg/index.json"


class _Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_package_graph_matches_the_complete_bundled_catalog():
    html = HTML.read_text()
    match = re.search(
        r'<script id="mceg-snapshot" type="application/json">(.*?)</script>',
        html, re.DOTALL,
    )
    assert match is not None
    snapshot = json.loads(match[1])
    graph = snapshot["graph"]
    assert graph["format"] == "mceg_graph_v1"
    assert snapshot["lang"] == "en"
    assert not snapshot["demo"]

    catalog = json.loads(CATALOG.read_bytes())
    sources = {s["path"]: s for s in graph["sources"]}
    release = re.search(r'^release = "([^"]+)"',
                        (ROOT / "documents/conf.py").read_text(), re.MULTILINE)[1]
    assert sources["index.json"]["sha256"] == hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    assert sources["index.json"]["distribution"] == {"name": "exogibbs", "version": release}
    records = {
        f"/models/{i}/records/{j}": record
        for i, model in enumerate(catalog["models"])
        for j, record in enumerate(model["records"])
    }
    assert set(sources) == {"index.json", *(r["path"] for r in records.values())}
    assert {s["origin"]["pointer"] for s in graph["selections"]} == set(records)
    for selection in graph["selections"]:
        record = records[selection["origin"]["pointer"]]
        assert selection["record"]["path"] == record["path"]
        assert selection["record"]["sha256"] == record["sha256"]
        assert sources[record["path"]]["sha256"] == record["sha256"]
        assert hashlib.sha256((CATALOG.parent / record["path"]).read_bytes()).hexdigest() == record["sha256"]

    elements = _Page(html).elements
    assert any(tag == "script" and attrs.get("id") == "mceg-script" for tag, attrs in elements)
    assert any(tag == "style" and attrs.get("id") == "mceg-style" for tag, attrs in elements)
    assert not any(tag == "script" and "src" in attrs for tag, attrs in elements)
    assert not any(tag == "link" and attrs.get("rel") == "stylesheet" for tag, attrs in elements)


def test_sphinx_copies_the_viewer_and_resolves_the_embedded_page(tmp_path):
    pytest.importorskip("sphinx")
    source, output = tmp_path / "source", tmp_path / "html"
    (source / "_static").mkdir(parents=True)
    (source / "mceg/_generated").mkdir(parents=True)
    (source / "conf.py").write_text('project = "Viewer test"\nhtml_static_path = ["_static"]\n')
    (source / "index.rst").write_text("Viewer test\n===========\n\n.. toctree::\n\n   mceg/package_graph\n")
    (source / "mceg/_generated/catalog.rst").write_text(":orphan:\n\nCatalog\n=======\n")
    shutil.copyfile(ROOT / "documents/mceg/package_graph.rst", source / "mceg/package_graph.rst")
    for name in ("package-graph.html", "package-graph-LICENSE.txt"):
        shutil.copyfile(ROOT / "documents/_static" / name, source / "_static" / name)
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-W", "--keep-going", "-q",
         str(source), str(output)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    page = output / "mceg/package_graph.html"
    elements = _Page(page.read_text()).elements
    frames = [attrs for tag, attrs in elements if tag == "iframe"]
    assert len(frames) == 1
    assert frames[0]["title"]
    target = page.parent / frames[0]["src"]
    assert target.read_bytes() == HTML.read_bytes()
    assert any(tag == "a" and attrs.get("href") == frames[0]["src"] for tag, attrs in elements)
