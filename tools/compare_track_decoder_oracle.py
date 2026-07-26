#!/usr/bin/env python3
"""Compare dumped CUDA TrackFormer windows with the PyTorch oracle JSON."""

import argparse
import json
from pathlib import Path


ATOL = 5e-3
RTOL = 1e-2


def close(actual, expected):
    return abs(actual - expected) <= ATOL + RTOL * abs(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pytorch_oracle")
    parser.add_argument("cuda_boundaries")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    oracle = json.loads(Path(args.pytorch_oracle).read_text())
    cuda = {}
    for line in Path(args.cuda_boundaries).read_text().splitlines():
        fields = line.split()
        if fields and (
                fields[0].startswith("production.track.decoder.layer") or
                fields[0].startswith("production.track.outputs.") or
                fields[0].startswith("production.track.decode.") or
                fields[0] == "production.track.query_interaction"):
            cuda[fields[0]] = [float(value) for value in fields[1:]]

    comparisons = []
    all_passed = True
    for record in oracle["layers"]:
        layer = record["layer"]
        for boundary, oracle_key in (
                ("norm2", "state_window"),
                ("regression", "regression_window"),
                ("reference", "reference_window")):
            name = (
                f"production.track.decoder.layer{layer}.{boundary}")
            actual = cuda.get(name)
            expected = record[oracle_key]
            if actual is None:
                raise ValueError(f"missing CUDA boundary: {name}")
            if len(actual) != len(expected):
                raise ValueError(
                    f"{name}: CUDA has {len(actual)} values, "
                    f"oracle has {len(expected)}")
            failures = [
                index for index, pair in enumerate(zip(actual, expected))
                if not close(*pair)]
            maximum = max(
                abs(left - right)
                for left, right in zip(actual, expected))
            passed = not failures
            all_passed &= passed
            comparisons.append({
                "name": name,
                "passed": passed,
                "passed_values": len(actual) - len(failures),
                "total_values": len(actual),
                "first_failure": failures[0] if failures else None,
                "max_abs_error": maximum,
                "cuda_window": actual,
                "pytorch_window": expected,
            })
    final_oracle = oracle.get(
        "direct_final_outputs", oracle["final_outputs"])
    for boundary, oracle_key in (
            ("class_logits", "class_logits_window"),
            ("box", "box_window"),
            ("past_trajectory", "past_trajectory_window")):
        name = f"production.track.outputs.{boundary}"
        actual = cuda.get(name)
        expected = final_oracle[oracle_key]
        if actual is None:
            raise ValueError(f"missing CUDA boundary: {name}")
        failures = [
            index for index, pair in enumerate(zip(actual, expected))
            if not close(*pair)]
        maximum = max(
            abs(left - right) for left, right in zip(actual, expected))
        passed = len(actual) == len(expected) and not failures
        all_passed &= passed
        comparisons.append({
            "name": name,
            "passed": passed,
            "passed_values": len(actual) - len(failures),
            "total_values": len(actual),
            "first_failure": failures[0] if failures else None,
            "max_abs_error": maximum,
            "cuda_window": actual,
            "pytorch_window": expected,
        })
    for boundary, oracle_key, discrete in (
            ("scores", "score_window", False),
            ("classes", "class_window", True),
            ("selected_count", "selected_count", True)):
        name = f"production.track.decode.{boundary}"
        actual = cuda.get(name)
        expected_value = final_oracle[oracle_key]
        expected = (
            [float(expected_value)] if boundary == "selected_count"
            else expected_value)
        if actual is None:
            raise ValueError(f"missing CUDA boundary: {name}")
        if discrete:
            failures = [
                index for index, pair in enumerate(zip(actual, expected))
                if int(pair[0]) != int(pair[1])]
        else:
            failures = [
                index for index, pair in enumerate(zip(actual, expected))
                if abs(pair[0] - pair[1]) >
                1e-4 + 1e-3 * abs(pair[1])]
        maximum = max(
            abs(left - right) for left, right in zip(actual, expected))
        passed = len(actual) == len(expected) and not failures
        all_passed &= passed
        comparisons.append({
            "name": name,
            "passed": passed,
            "passed_values": len(actual) - len(failures),
            "total_values": len(actual),
            "first_failure": failures[0] if failures else None,
            "max_abs_error": maximum,
            "cuda_window": actual,
            "pytorch_window": expected,
            "comparison": "exact integer" if discrete
                          else {"atol": 1e-4, "rtol": 1e-3},
        })
    if "direct_query_interaction" in oracle:
        name = "production.track.query_interaction"
        actual = cuda.get(name)
        expected = oracle["direct_query_interaction"]["output_window"]
        if actual is None:
            raise ValueError(f"missing CUDA boundary: {name}")
        failures = [
            index for index, pair in enumerate(zip(actual, expected))
            if not close(*pair)]
        maximum = max(
            abs(left - right) for left, right in zip(actual, expected))
        passed = len(actual) == len(expected) and not failures
        all_passed &= passed
        comparisons.append({
            "name": name,
            "passed": passed,
            "passed_values": len(actual) - len(failures),
            "total_values": len(actual),
            "first_failure": failures[0] if failures else None,
            "max_abs_error": maximum,
            "cuda_window": actual,
            "pytorch_window": expected,
        })

    gate = {
        "schema": "uniad.c/numerical-gate-v1",
        "candidate": (
            "track-decoder-heads-filter-and-query-interaction"),
        "oracle": str(Path(args.pytorch_oracle).resolve()),
        "cuda_dump": str(Path(args.cuda_boundaries).resolve()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "passed": all_passed,
        "comparisons": comparisons,
        "claim_limit": (
            "Decoder/refinement windows use a shared complete BEV boundary; "
            "final-head windows use shared hashed decoder state/regression/"
            "reference boundaries. This does not prove upstream visual, "
            "full-tensor, tracking-state/decode, or end-to-end equivalence."
        ),
    }
    Path(args.output).write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(
        f"track decoder gate: {'passed' if all_passed else 'failed'} "
        f"({len(comparisons)} boundaries)")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
