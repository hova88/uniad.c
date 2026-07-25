# Model contract

## Pinned production metadata (not executable)

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

The runtime rejects inference for this profile because there is no compatible
exported checkpoint and no proven operator mapping.

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
