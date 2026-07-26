# Model contract

## Pinned production v2 contract

The `production-nuscenes-stage2` contract is tied to upstream commit
`609ee083ea51c3521c323f1279dfc4cee0e60467`.

| Surface | Contract |
|---|---|
| Cameras | six BGR views |
| Preprocess | configured normalization, size-divisor padding |
| Metadata | camera calibration, ego pose/motion, timestamp, CAN bus, command |
| Temporal BEV | `200 × 200 × 256` |
| Tracking | 900 object queries |
| Heads | tracking, vector map, multimodal motion, future occupancy, ego planning |
| Motion | six modes, 12 future steps |
| Planning | six future steps |
| Frame / units | ego-centric metric coordinates |

`tools/convert_uaw2.py` now exports the official 2459 checkpoint tensors and
the runtime validates and permanently uploads that payload on the CUDA backend.
The public v2 input accepts six decoded strided uint8 BGR image views,
calibration, ego pose, 18-value CAN bus, timestamp, scene token and navigation
command. The fixed-capacity result holds 300 tracks, 300 map vectors, `6×12`
motion per track, `5×200×200` occupancy and a six-step ego plan.

For this pinned profile each decoded view is 1600×900. The CUDA entry uses 2D
copies honoring `row_stride_bytes`, subtracts BGR means
`[103.530,116.280,123.675]`, preserves BGR order (`to_rgb=False`), converts to
planar FP16, and pads height to 928. The `6×3×928×1600` result remains in the
activation arena.

Calibration, ego pose, CAN bus, timestamp, navigation command and a stable scene
hash occupy a fixed 760-byte device packet. A changed scene token clears
previous BEV and track memory on the same CUDA stream before the new frame is
processed; explicit context reset clears the same state and scene identity.

Production CPU creation returns `unsupported`; there is no silent fallback.
The official operator graph is still gated and `ua_infer_production` returns
`unsupported` after input validation. Container/residency support must not be
reported as graph equivalence.

The loaded model retains the checksum-validated UAW2 directory for its entire
lifetime. Learned dispatch must resolve an exact checkpoint key with
`ua_model_find_tensor`, validate dtype/rank/shape, then add the returned
payload-relative offset to the resident CUDA weight base. Hard-coded file
offsets are outside the production contract.

## Runnable synthetic profile

| Surface | `tiny-synthetic-v1` |
|---|---|
| Input | six planar `3 × 8 × 8` FP32 camera tensors |
| State | one prior `8 × 8 × 16` FP32 BEV, scene-keyed |
| Queries | 64 candidates; stable top 8 |
| Map | four line elements |
| Motion | three modes, four steps, eight tracks |
| Occupancy | three `8 × 8` binary horizons |
| Planning | six ego points plus collision score |
| Limits | fixed compile-time capacities; no dynamic growth |

This graph is pedagogically faithful to representation flow, not numerically
equivalent to upstream UniAD.
