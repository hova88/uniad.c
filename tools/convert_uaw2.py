#!/usr/bin/env python3
"""Convert an official UniAD checkpoint into a bounds-checkable UAW2 file.

PyTorch is an offline-only dependency. The runtime container stores every
checkpoint tensor with explicit dtype, shape, aligned offset and SHA-256.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch


ALIGN = 256
ENDIAN = 0x01020304
PROFILE = 2
HEADER = struct.Struct("<4sIIIIIQQQ32s32s32s112s")
ENTRY = struct.Struct("<128sIIII8QQQ32s8s")
DTYPES = {np.dtype("float16"): 1, np.dtype("float32"): 2,
          np.dtype("int64"): 3, np.dtype("int32"): 4, np.dtype("uint8"): 5}
SENSITIVE = ("running_mean", "running_var", "reference_points", "level_embeds",
             "cams_embeds", "can_bus_mlp", "motion_anchor")


def align(value):
    return (value + ALIGN - 1) // ALIGN * ALIGN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).digest()
    config_sha = hashlib.sha256(config_path.read_bytes()).digest()
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = obj.get("state_dict", obj)
    if not isinstance(state, dict):
        raise SystemExit("checkpoint does not contain a state_dict")

    tensors = []
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        if value.ndim > 8 or len(name.encode()) >= 128:
            raise SystemExit(f"unsupported tensor metadata: {name}")
        value = value.detach().cpu().contiguous()
        if value.is_floating_point():
            dtype = torch.float32 if any(k in name for k in SENSITIVE) else torch.float16
            value = value.to(dtype)
        array = value.numpy()
        if array.dtype not in DTYPES:
            raise SystemExit(f"unsupported dtype {array.dtype}: {name}")
        raw = array.tobytes(order="C")
        tensors.append((name, list(array.shape), DTYPES[array.dtype], raw))

    directory_offset = HEADER.size
    data_offset = align(directory_offset + len(tensors) * ENTRY.size)
    offset = data_offset
    entries = []
    manifest = []
    for name, shape, dtype, raw in tensors:
        offset = align(offset)
        dims = shape + [0] * (8 - len(shape))
        digest = hashlib.sha256(raw).digest()
        entries.append(ENTRY.pack(
            name.encode() + b"\0" * (128 - len(name.encode())),
            dtype, len(shape), 0, 0, *dims, offset, len(raw), digest, b"\0" * 8))
        manifest.append({
            "name": name, "shape": shape, "dtype": dtype, "offset": offset,
            "nbytes": len(raw), "sha256": digest.hex(),
        })
        offset += len(raw)
    file_size = offset
    directory = b"".join(entries)
    header = HEADER.pack(
        b"UAW2", 2, ENDIAN, PROFILE, len(tensors), HEADER.size,
        directory_offset, data_offset, file_size, config_sha, checkpoint_sha,
        hashlib.sha256(directory).digest(), b"\0" * 112)

    output = Path(args.output)
    with output.open("wb") as handle:
        handle.write(header)
        handle.write(directory)
        handle.write(b"\0" * (data_offset - handle.tell()))
        for item, (_, _, _, raw) in zip(manifest, tensors):
            handle.write(b"\0" * (item["offset"] - handle.tell()))
            handle.write(raw)
    assert output.stat().st_size == file_size

    report = {
        "schema": "uniad.c/uaw2-manifest-v1",
        "container": "UAW2",
        "profile": "production-nuscenes-stage2",
        "checkpoint_sha256": checkpoint_sha.hex(),
        "config_sha256": config_sha.hex(),
        "tensor_count": len(tensors),
        "file_size": file_size,
        "alignment": ALIGN,
        "floating_policy": "fp16-default-sensitive-fp32",
        "bn_folded": False,
        "layout": "PyTorch OIHW/OI; runtime kernels consume these layouts",
        "tensors": manifest,
    }
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(
        output.suffix + ".json")
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "container", "tensor_count", "file_size", "checkpoint_sha256",
        "config_sha256")}))


if __name__ == "__main__":
    main()
