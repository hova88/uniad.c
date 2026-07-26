#!/usr/bin/env python3
"""Build an auditable UniAD operator/shape and saved-tensor inventory.

The input JSONL files are emitted by UniAD/tools/trace_uniad_flow.py.  This
script intentionally performs no model execution: it turns immutable oracle
records into a compact production-planning artifact.
"""

import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def read_jsonl(path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: {exc}") from exc


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def tensor_signature(tensor):
    return {
        key: tensor.get(key)
        for key in ("shape", "stride", "dtype", "layout", "contiguous")
    }


def stage_for(module):
    name = module.lower()
    if name == "<root>":
        return "root"
    rules = (
        ("planning", ("planning_head",)),
        ("occupancy", ("occ_head",)),
        ("motion", ("motion_head",)),
        ("mapping", ("seg_head", "map_head")),
        ("bev_temporal", ("temporal_self_attention",)),
        ("bev_spatial", ("spatial_cross_attention",)),
        ("bev_encoder", ("pts_bbox_head.transformer.encoder", "bevformer")),
        ("tracking", ("pts_bbox_head", "track_head", "memory_bank",
                      "query_interact", "queryinteraction")),
        ("image_backbone", ("img_backbone", "grid_mask")),
        ("image_neck", ("img_neck",)),
    )
    for stage, needles in rules:
        if any(needle in name for needle in needles):
            return stage
    return "support"


def operator_family(operator):
    leaf = operator.rsplit(".", 1)[-1]
    aliases = {
        "Conv2d": "conv2d",
        "ConvModule": "conv2d_epilogue",
        "Linear": "linear",
        "LayerNorm": "layer_norm",
        "ReLU": "activation",
        "GELU": "activation",
        "Dropout": "dropout",
        "Softmax": "softmax",
        "MultiheadAttention": "multihead_attention",
    }
    if leaf in aliases:
        return aliases[leaf]
    lowered = leaf.lower()
    for needle, family in (
        ("deform", "deformable_attention_or_conv"),
        ("attention", "attention"),
        ("norm", "normalization"),
        ("conv", "convolution"),
        ("linear", "linear"),
        ("pool", "pooling"),
        ("interpolate", "resize"),
    ):
        if needle in lowered:
            return family
    return leaf


def summarize_calls(paths):
    signatures = {}
    stage_totals = collections.defaultdict(
        lambda: {"calls": 0, "input_bytes": 0, "output_bytes": 0,
                 "cuda_event_ms": []})
    family_signatures = collections.defaultdict(set)
    mode_counts = {}
    for mode, path in paths:
        count = 0
        for record in read_jsonl(path):
            if record.get("kind") != "module":
                continue
            count += 1
            inputs = record.get("inputs", [])
            outputs = record.get("outputs", [])
            canonical = {
                "operator": record["operator"],
                "inputs": [tensor_signature(item) for item in inputs],
                "outputs": [tensor_signature(item) for item in outputs],
            }
            key = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
            entry = signatures.setdefault(key, {
                **canonical,
                "family": operator_family(record["operator"]),
                "stages": set(),
                "modules": set(),
                "source": record.get("source"),
                "line": record.get("line"),
                "calls": {"infer": 0, "train": 0},
                "cuda_event_ms": {"infer": [], "train": []},
                "input_bytes_per_call": sum(x.get("nbytes", 0) for x in inputs),
                "output_bytes_per_call": sum(x.get("nbytes", 0) for x in outputs),
                "requires_grad": any(x.get("requires_grad", False)
                                     for x in inputs + outputs),
            })
            stage = stage_for(record["module"])
            entry["stages"].add(stage)
            entry["modules"].add(record["module"])
            entry["calls"][mode] += 1
            elapsed = record.get("cuda_elapsed_ms")
            if elapsed is not None:
                entry["cuda_event_ms"][mode].append(elapsed)
            total = stage_totals[(mode, stage)]
            total["calls"] += 1
            total["input_bytes"] += entry["input_bytes_per_call"]
            total["output_bytes"] += entry["output_bytes_per_call"]
            if elapsed is not None:
                total["cuda_event_ms"].append(elapsed)
        mode_counts[mode] = count

    output = []
    for serial, (key, entry) in enumerate(
            sorted(signatures.items(), key=lambda item: item[0])):
        entry["signature_id"] = f"op-{serial:04d}"
        entry["stages"] = sorted(entry["stages"])
        entry["modules"] = sorted(entry["modules"])
        for mode in ("infer", "train"):
            values = entry["cuda_event_ms"][mode]
            entry["cuda_event_ms"][mode] = {
                "samples": len(values),
                "sum": sum(values),
                "p50": percentile(values, .50),
                "p95": percentile(values, .95),
                "max": max(values) if values else None,
            }
        family_signatures[entry["family"]].add(entry["signature_id"])
        output.append(entry)

    stages = {}
    for (mode, stage), value in sorted(stage_totals.items()):
        stages.setdefault(stage, {})[mode] = {
            "calls": value["calls"],
            "input_bytes": value["input_bytes"],
            "output_bytes": value["output_bytes"],
            "cuda_event_ms_sum": sum(value["cuda_event_ms"]),
        }
    families = [
        {"family": family, "unique_signatures": len(ids),
         "signature_ids": sorted(ids)}
        for family, ids in sorted(family_signatures.items())
    ]
    return output, families, stages, mode_counts


def summarize_lifetimes(path):
    packs = {}
    live = {}
    peak_live_bytes = 0
    peak_live_tensors = 0
    lifetimes = []
    sequence = 0
    for record in read_jsonl(path):
        if record.get("kind") not in ("saved_tensor_pack", "saved_tensor_unpack"):
            sequence += 1
            continue
        tensor_id = record.get("tensor_id")
        if record["kind"] == "saved_tensor_pack":
            size = record["tensor"].get("nbytes", 0)
            packs[tensor_id] = (sequence, size, record["tensor"])
            live[tensor_id] = size
            peak_live_bytes = max(peak_live_bytes, sum(live.values()))
            peak_live_tensors = max(peak_live_tensors, len(live))
        elif tensor_id in packs:
            start, size, tensor = packs[tensor_id]
            lifetimes.append({
                "tensor_id": tensor_id,
                "pack_event": start,
                "unpack_event": sequence,
                "event_distance": sequence - start,
                "nbytes": size,
                "shape": tensor.get("shape"),
                "dtype": tensor.get("dtype"),
            })
            live.pop(tensor_id, None)
        sequence += 1
    distances = [item["event_distance"] for item in lifetimes]
    return {
        "matched": len(lifetimes),
        "unmatched_packs": len(live),
        "peak_live_bytes_upper_bound": peak_live_bytes,
        "peak_live_tensors": peak_live_tensors,
        "event_distance": {
            "p50": percentile(distances, .50),
            "p95": percentile(distances, .95),
            "max": max(distances) if distances else None,
        },
        "longest_lived": sorted(
            lifetimes, key=lambda item: item["event_distance"], reverse=True)[:100],
        "largest_saved": sorted(
            lifetimes, key=lambda item: item["nbytes"], reverse=True)[:100],
        "interpretation": (
            "Upper bound from saved-tensor hook events only; it is not CUDA "
            "allocator peak memory and excludes parameters/workspaces."
        ),
    }


def summarize_checkpoint(paths):
    by_key = {}
    for mode, path in paths:
        for item in json.loads(path.read_text()):
            entry = by_key.setdefault(item["checkpoint_key"], {
                "checkpoint_key": item["checkpoint_key"],
                "shape": item["shape"],
                "dtype": item["dtype"],
                "consumers": {},
            })
            entry["consumers"][mode] = {
                "runtime_module": item.get("runtime_module"),
                "consumer_operator": item.get("consumer_operator"),
            }
    entries = [by_key[key] for key in sorted(by_key)]
    observed = sum(
        any(v.get("runtime_module") for v in item["consumers"].values())
        for item in entries)
    return {
        "unique_checkpoint_keys": len(entries),
        "keys_with_observed_runtime_consumer": observed,
        "keys_without_observed_runtime_consumer": len(entries) - observed,
        "entries": entries,
    }


def write_markdown(inventory, path):
    summary = inventory["summary"]
    life = inventory["training_saved_tensor_lifetimes"]
    lines = [
        "# UniAD 生产算子与 Tensor 生命周期清单",
        "",
        "## 证据边界",
        "",
        "本清单由真实 nuScenes mini 六相机样本的 PyTorch oracle 追踪生成。"
        "`cuda_elapsed_ms` 是嵌套 module CUDA Event 跨度，会重复计时；由于当前机器 "
        "CUPTI 返回 `CUPTI_ERROR_INVALID_DEVICE`，它不是 kernel 时间，也不可相加后"
        "当作端到端延迟。清单证明调用、shape、dtype、布局和 autograd 保存关系，"
        "不证明纯 CUDA 图等价或数据集精度。",
        "",
        "## 总览",
        "",
        f"- 推理 module 调用：{summary['module_calls']['infer']}",
        f"- 训练 module 调用：{summary['module_calls']['train']}",
        f"- 唯一 operator/shape/layout 签名：{summary['unique_signatures']}",
        f"- 算子族：{summary['operator_families']}",
        f"- checkpoint keys：{summary['checkpoint_keys']}",
        f"- 有观测消费者的 checkpoint keys："
        f"{summary['checkpoint_keys_with_observed_consumer']}",
        f"- saved tensor 匹配生命周期：{life['matched']}",
        f"- saved tensor 事件估算峰值：{life['peak_live_bytes_upper_bound'] / 2**30:.3f} GiB "
        f"（不含参数、workspace、allocator 碎片）",
        "",
        "## 阶段调用",
        "",
        "| 阶段 | 推理调用 | 训练调用 | 推理输出 GiB（累计流量） | 训练输出 GiB（累计流量） |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, modes in inventory["stage_totals"].items():
        infer = modes.get("infer", {})
        train = modes.get("train", {})
        lines.append(
            f"| {stage} | {infer.get('calls', 0)} | {train.get('calls', 0)} | "
            f"{infer.get('output_bytes', 0) / 2**30:.3f} | "
            f"{train.get('output_bytes', 0) / 2**30:.3f} |")
    lines += [
        "",
        "## 算子族",
        "",
        "| 算子族 | 唯一签名数 |",
        "|---|---:|",
    ]
    for family in inventory["operator_families"]:
        lines.append(f"| {family['family']} | {family['unique_signatures']} |")
    lines += [
        "",
        "## 机器可读细节",
        "",
        "同目录 `production-operator-inventory.json` 包含每个签名的完整输入/输出 "
        "shape、stride、dtype、layout、字节数、梯度需求、源码位置、module 集合、"
        "训练/推理调用数和 CUDA Event 分布；还包含最长/最大的 saved tensor 生命周期"
        "以及逐 checkpoint key 的运行时消费者。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer-calls", type=Path, required=True)
    parser.add_argument("--train-calls", type=Path, required=True)
    parser.add_argument("--infer-map", type=Path, required=True)
    parser.add_argument("--train-map", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--web-json", type=Path)
    args = parser.parse_args()

    signatures, families, stages, counts = summarize_calls(
        (("infer", args.infer_calls), ("train", args.train_calls)))
    checkpoint = summarize_checkpoint(
        (("infer", args.infer_map), ("train", args.train_map)))
    inventory = {
        "schema": "uniad-production-operator-inventory-v1",
        "summary": {
            "module_calls": counts,
            "unique_signatures": len(signatures),
            "operator_families": len(families),
            "checkpoint_keys": checkpoint["unique_checkpoint_keys"],
            "checkpoint_keys_with_observed_consumer":
                checkpoint["keys_with_observed_runtime_consumer"],
        },
        "measurement_notes": {
            "cuda_elapsed_ms": (
                "Nested module CUDA Event span, not kernel time; values overlap."
            ),
            "cupti": (
                "Kernel activity unavailable: CUPTI_ERROR_INVALID_DEVICE on "
                "the evidence machine."
            ),
            "claim_layers": [
                "operator-boundary oracle",
                "end-to-end graph equivalence (not yet achieved by uniad.c)",
                "dataset metrics (not evaluated on mini)",
            ],
        },
        "stage_totals": stages,
        "operator_families": families,
        "signatures": signatures,
        "training_saved_tensor_lifetimes":
            summarize_lifetimes(args.train_calls),
        "checkpoint_consumers": checkpoint,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"
    args.json.write_text(encoded, encoding="utf-8")
    write_markdown(inventory, args.markdown)
    if args.web_json:
        args.web_json.parent.mkdir(parents=True, exist_ok=True)
        args.web_json.write_text(encoded, encoding="utf-8")
    print(json.dumps(inventory["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
