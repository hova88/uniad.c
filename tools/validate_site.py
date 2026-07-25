#!/usr/bin/env python3
import json, pathlib, re, subprocess
root = pathlib.Path(__file__).resolve().parents[1]
html = (root / "docs/index.html").read_text()
assert "<title>" in html and "<noscript>" in html and 'id="article"' in html
for link in re.findall(r'(?:href|src)="([^"#]+)"', html):
    if link.startswith(("http:", "https:")): continue
    assert (root / "docs" / link).exists(), f"broken local link: {link}"
json.loads((root / "docs/assets/demo-result.json").read_text())
json.loads((root / "docs/assets/operator-inventory.json").read_text())
node = subprocess.run(["node", "--check", root / "docs/assets/site.js"],
                      capture_output=True, text=True)
assert node.returncode == 0, node.stderr
assert "http://" not in (root / "docs/assets/site.js").read_text()
print("site validation: ok")
