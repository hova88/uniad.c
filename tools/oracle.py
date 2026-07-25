#!/usr/bin/env python3
"""Independent PyTorch oracle for the tiny synthetic graph.

This intentionally re-expresses the math rather than binding to the C runtime.
It proves only tiny-graph equivalence, never upstream UniAD equivalence.
"""
from __future__ import annotations
import argparse, json, math, pathlib, struct, subprocess, sys
import torch

CAMERAS, TRACKS, MODES, PRED, PLAN, OCC = 6, 8, 3, 4, 6, 3

def fixtures():
    w = torch.tensor([
        math.sin((i + 1) * .37) * .45 + .55 for i in range(64)
    ], dtype=torch.float32)
    frames = []
    for frame in range(2):
        v = []
        for i in range(CAMERAS * 3 * 8 * 8):
            cam, rem = divmod(i, 3 * 8 * 8)
            ch, p = divmod(rem, 64)
            v.append(((p * 13 + cam * 17 + ch * 29 + frame * 11) % 101) / 50.0 - 1.0)
        frames.append(torch.tensor(v, dtype=torch.float32).reshape(CAMERAS, 3, 8, 8))
    return w, frames

def graph(weights, frames):
    previous = None
    final = None
    for frame_index, camera in enumerate(frames):
        bev = torch.empty((8, 8, 16), dtype=torch.float32)
        for y in range(8):
            for x in range(8):
                for d in range(16):
                    total = torch.tensor(0.0, dtype=torch.float32)
                    for cam in range(CAMERAS):
                        for ch in range(3):
                            total += camera[cam, ch, (y + cam) & 7, (x + d + cam) & 7] * weights[(cam * 7 + ch * 3 + d) & 63]
                    bev[y, x, d] = torch.tanh(total / 18.0)
        if previous is not None:
            warped = bev.clone()
            for y in range(8):
                for x in range(8):
                    px = x - 1
                    if px >= 0:
                        warped[y, x] = .65 * bev[y, x] + .35 * previous[y, px]
            bev = warped
        spatial = torch.empty_like(bev)
        for y in range(8):
            for x in range(8):
                value = 2 * bev[y, x]
                count = 2
                if x: value = value + bev[y, x - 1]; count += 1
                if x < 7: value = value + bev[y, x + 1]; count += 1
                if y: value = value + bev[y - 1, x]; count += 1
                if y < 7: value = value + bev[y + 1, x]; count += 1
                spatial[y, x] = value / count
        previous = spatial
        score = torch.empty(64)
        for q in range(64):
            total = torch.tensor(0.0)
            flat = spatial.reshape(64, 16)
            for d in range(16):
                total += flat[q, d] * weights[(q + d) & 63]
            score[q] = torch.sigmoid(total / 8)
        order = sorted(range(64), key=lambda q: (-float(score[q]), q))[:TRACKS]
        tracks = [{"id": q, "x": float(q % 8) - 3.5,
                   "y": float(q // 8) - 3.5, "score": float(score[q])} for q in order]
        maps = [{"points": [[-4., -3. + 2 * i], [4., -3. + 2 * i]],
                 "score": float(score[i * 8 + 4])} for i in range(4)]
        motions = []
        for tr in tracks:
            modes = []
            for mode in range(MODES):
                dx = (.15 + .05 * mode) * ((tr["id"] % 3) - 1)
                dy = .12 + .04 * mode
                modes.append({"score": .5 - .1 * mode + .05 * tr["score"],
                    "trajectory": [[tr["x"] + dx * (s + 1),
                                    tr["y"] + dy * (s + 1)] for s in range(PRED)]})
            motions.append({"track_id": tr["id"], "modes": modes})
        occupancy = [[int(float(score[y * 8 + x]) > .50 + .03 * h)
                      for y in range(8) for x in range(8)] for h in range(OCC)]
        command = "left" if frame_index else "straight"
        turn = -.12 if command == "left" else 0.
        plan = [[turn * (s + 1), .7 * (s + 1)] for s in range(PLAN)]
        collision = 0.
        for s, point in enumerate(plan):
            for motion in motions:
                target = motion["modes"][0]["trajectory"][min(s, PRED - 1)]
                risk = math.exp(-((point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2))
                collision = max(collision, risk)
        final = {"schema": "uniad.c/result-v1", "profile": "tiny-synthetic-v1",
            "scene": "synthetic-scene-001", "frame_index": frame_index,
            "coordinate_frame": "ego", "units": {"distance": "meter", "time": "step"},
            "command": command, "tracks": tracks, "map": maps, "motion": motions,
            "occupancy": occupancy, "ego_plan": plan, "collision_score": collision}
    return final

def compare(a, b, path="$", atol=1e-5, rtol=1e-4):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for key in a: compare(a[key], b[key], f"{path}.{key}", atol, rtol)
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, f"{path}[{i}]", atol, rtol)
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
        assert abs(float(a) - float(b)) <= atol + rtol * abs(float(a)), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", metavar="CLI")
    p.add_argument("--asset-dir", default="build/oracle-demo")
    p.add_argument("--write-expected")
    args = p.parse_args()
    expected = graph(*fixtures())
    if args.write_expected:
        pathlib.Path(args.write_expected).write_text(json.dumps(expected, indent=2) + "\n")
    if args.compare:
        asset = pathlib.Path(args.asset_dir)
        asset.mkdir(parents=True, exist_ok=True)
        subprocess.run([args.compare, "generate-demo", str(asset)], check=True,
                       stdout=subprocess.DEVNULL)
        actual = json.loads(subprocess.check_output(
            [args.compare, "demo", "--dir", str(asset)], text=True))
        compare(expected, actual)
        print("oracle: two-frame tiny graph matches (atol=1e-5, rtol=1e-4)")
    elif not args.write_expected:
        json.dump(expected, sys.stdout, separators=(",", ":")); print()

if __name__ == "__main__":
    main()
