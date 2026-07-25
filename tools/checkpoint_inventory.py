#!/usr/bin/env python3
"""Safely inventory a future PyTorch checkpoint; never exports production UAW."""
import argparse, hashlib, json, pathlib
import torch

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--output")
args = p.parse_args()
path = pathlib.Path(args.checkpoint)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
# weights_only prevents arbitrary pickle globals in supported PyTorch versions.
obj = torch.load(path, map_location="cpu", weights_only=True)
state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
if not isinstance(state, dict):
    raise SystemExit("checkpoint does not contain a tensor mapping")
items = []
for name in sorted(state):
    value = state[name]
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous().numpy().tobytes()
        items.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                      "sha256": hashlib.sha256(raw).hexdigest()})
report = {"schema": "uniad.c/checkpoint-inventory-v1", "file_sha256": digest,
          "tensor_count": len(items), "tensors": items,
          "production_export_ready": False,
          "reason": "required upstream-to-runtime mapping is not implemented"}
text = json.dumps(report, indent=2) + "\n"
if args.output: pathlib.Path(args.output).write_text(text)
else: print(text, end="")
