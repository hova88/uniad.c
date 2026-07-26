#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
CLI = Path(os.environ.get("UA_TEST_CLI", ROOT / "build/cpu/uniad"))

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    checkpoint = root / "model.pth"
    config = root / "config.py"
    model = root / "model.uaw2"
    manifest = root / "manifest.json"
    config.write_text("model = dict(type='fixture')\n")
    torch.save({"state_dict": {
        "conv.weight": torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
        "bn.running_mean": torch.tensor([1.0, 2.0]),
        "bn.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }}, checkpoint)
    subprocess.run([
        "python3", ROOT / "tools/convert_uaw2.py", checkpoint, model,
        "--config", config, "--manifest", manifest,
    ], check=True, capture_output=True)
    report = json.loads(manifest.read_text())
    assert report["tensor_count"] == 3
    assert all(item["offset"] % 256 == 0 for item in report["tensors"])
    inspected = json.loads(subprocess.check_output([CLI, "inspect-model", model]))
    assert inspected["container"] == "UAW2"
    assert inspected["tensor_count"] == 3
    tensor = json.loads(subprocess.check_output([
        CLI, "inspect-model", model, "--tensor", "conv.weight"
    ]))
    assert tensor["dtype"] == 1
    assert tensor["rank"] == 4
    assert tensor["shape"] == [2, 3, 2, 2]
    assert tensor["nbytes"] == 48
    assert tensor["byte_offset"] % 256 == 0
    missing = subprocess.run([
        CLI, "inspect-model", model, "--tensor", "missing.weight"
    ], capture_output=True)
    assert missing.returncode != 0
    cpu = subprocess.run([
        CLI, "infer", "--model", model, "--frame", root / "missing.uaf",
        "--backend", "cpu",
    ], text=True, capture_output=True)
    assert cpu.returncode and "operator graph unavailable" in cpu.stderr

    truncated = root / "truncated.uaw2"
    truncated.write_bytes(model.read_bytes()[:-1])
    assert subprocess.run([CLI, "inspect-model", truncated],
                          capture_output=True).returncode != 0
    corrupt = root / "corrupt.uaw2"
    payload = bytearray(model.read_bytes())
    payload[-1] ^= 1
    corrupt.write_bytes(payload)
    assert subprocess.run([CLI, "inspect-model", corrupt],
                          capture_output=True).returncode != 0

print("uaw2 tests: ok")
