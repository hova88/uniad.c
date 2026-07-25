#!/usr/bin/env python3
import json, pathlib, re, subprocess
root = pathlib.Path(__file__).resolve().parents[1]
html = (root / "docs/index.html").read_text()
assert "<title>" in html and "<noscript>" in html and 'id="article"' in html
required_sections = {
    "problem", "system", "bev", "queries", "map", "motion", "occupancy",
    "planning", "training", "evidence", "runtime", "limits", "references",
}
missing = sorted(section for section in required_sections
                 if f'id="{section}"' not in html)
assert not missing, f"missing article sections: {', '.join(missing)}"
for link in re.findall(r'(?:href|src)="([^"#]+)"', html):
    if link.startswith(("http:", "https:")): continue
    assert (root / "docs" / link).exists(), f"broken local link: {link}"
result = json.loads((root / "docs/assets/demo-result.json").read_text())
assert all(len(item.get("points", [])) >= 2 for item in result["map"]), \
    "canonical map entries must expose point arrays"
assert len(result["tracks"]) == 8 and len(result["ego_plan"]) == 6
assert all(len(item) == 64 for item in result["occupancy"])
json.loads((root / "docs/assets/operator-inventory.json").read_text())
study = (root / "evidence/uniad-architecture-study.md").read_text()
for topic in ("TrackFormer", "MapFormer", "MotionFormer", "OccFormer", "Planner"):
    assert topic in study, f"architecture study missing topic: {topic}"
css = (root / "docs/assets/site.css").read_text()
assert "@import" not in css and "url(http" not in css
script = (root / "docs/assets/site.js").read_text()
for hook in ("canonicalMapPoints", "canonicalBounds", "data-canonical-layer",
             "interface-aperture", "alignment-control", "track-time",
             "map-endpoint", "remove-context", "occupancy-world",
             "planner-canvas", "data-training-stage", "data-evidence-edge",
             "contract-wipe", "data-proof-rung"):
    assert hook in html or hook in script, f"missing interaction hook: {hook}"
for legacy_hook in ("data-system-step", "data-life=", "data-interaction=",
                    "collision-toggle", "data-contract="):
    assert legacy_hook not in html, f"legacy detached-control pattern remains: {legacy_hook}"
design_study = (root / "evidence/interaction-design-study.md").read_text()
for principle in ("One scene, one gesture", "Preserve a baseline",
                  "Progressive disclosure", "Input parity"):
    assert principle in design_study, f"interaction study missing principle: {principle}"
assert not re.search(r"[\u3400-\u9fff]", html + script), \
    "article and interactive copy must remain English"
node = subprocess.run(["node", "--check", root / "docs/assets/site.js"],
                      capture_output=True, text=True)
assert node.returncode == 0, node.stderr
assert "http://" not in script
print("site validation: ok")
