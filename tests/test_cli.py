#!/usr/bin/env python3
import json, pathlib, subprocess, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ROOT / "build" / "uniad"

with tempfile.TemporaryDirectory() as directory:
    subprocess.run([CLI, "generate-demo", directory], check=True, capture_output=True)
    model = json.loads(subprocess.check_output([CLI, "inspect-model", f"{directory}/demo.uaw"]))
    assert model["profile"] == "tiny-synthetic-v1"
    result = json.loads(subprocess.check_output([CLI, "demo", "--dir", directory]))
    assert result["schema"] == "uniad.c/result-v1"
    assert len(result["tracks"]) == 8 and len(result["ego_plan"]) == 6
    bench = json.loads(subprocess.check_output(
        [CLI, "benchmark", "--dir", directory, "--warmup", "0", "--runs", "2"]))
    assert bench["evidence"] == "synthetic" and bench["owned_host_bytes"] <= 256 * 1024 * 1024
    corrupt = pathlib.Path(directory) / "bad.uaw"
    data = bytearray((pathlib.Path(directory) / "demo.uaw").read_bytes())
    data[-1] ^= 1; corrupt.write_bytes(data)
    assert subprocess.run([CLI, "inspect-model", corrupt], capture_output=True).returncode != 0

p = subprocess.run([CLI, "infer", "--profile", "production"], text=True, capture_output=True)
assert p.returncode and "metadata-only" in p.stderr
p = subprocess.run([CLI, "demo", "--backend", "cuda"], text=True, capture_output=True)
assert p.returncode and "backend unavailable" in p.stderr
print("cli tests: ok")
