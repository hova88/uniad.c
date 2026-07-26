# Deformable attention production contract

This document freezes the common CUDA core and the four UniAD adapters. It is
derived from the pinned UniAD modules and the real mini traces; it is not an
assumed generic attention layout.

## Common multi-scale sampling core

| Tensor | Logical layout | Storage / accumulation |
|---|---|---|
| projected value | `[B, Σ(H_lW_l), heads, channels/head]` | FP16 / read as FP32 |
| spatial shapes | `[levels, 2]` in `(height,width)` order | uint32 |
| level starts | `[levels]`, exact exclusive prefix of `H_lW_l` | `size_t` |
| sampling locations | `[B,Q,heads,levels,points,2]` in normalized `(x,y)` | FP32 |
| attention weights | `[B,Q,heads,levels,points]` | FP32 |
| output | `[B,Q,heads×channels/head]` | FP32 baseline |

For normalized coordinate `(x_n,y_n)` at level `(H,W)`, the exact
`grid_sample(..., mode=bilinear, padding_mode=zeros, align_corners=False)`
conversion is:

```text
x = x_n * W - 0.5
y = y_n * H - 0.5
```

The four neighboring samples outside `[0,W)×[0,H)` contribute zero. Each output
channel uses the value from the same head and accumulates all `level×point`
products in FP32. The CUDA baseline uses one thread per output channel and
therefore has deterministic loop order.

The fixture covers two levels `(2×3,1×2)`, exact prefix starts `(0,6)`, two
batches, two heads, four channels/head, fractional and outside coordinates, and
rejects inconsistent starts. Its FP16-value gate is `atol=5e-3, rtol=1e-2`.

## UniAD adapters

### BEV spatial cross-attention

`MSDeformableAttention3D` uses 8 heads, 4 image levels and 8 total points per
head. Projected 3D anchors are already camera-plane normalized. Offsets are
divided by `(width,height)`, then reshaped so the Z anchors are folded into the
point dimension. Camera rebatching produces a variable visible-query count
(9,675 in the traced inference frame); the adapter must retain the camera/query
scatter map and average only cameras that observed each BEV query.

### BEV temporal self-attention

The queue is exactly two: aligned previous BEV and current query. Query features
are concatenated before predicting offsets and weights. Queue is folded into
batch for the common core, then the two sampled outputs are reshaped and averaged
before output projection and residual addition. This averaging is outside the
common sampler and must not be replaced with an unnormalized sum.

### Track and map decoder attention

`CustomMSDeformableAttention` uses 8 heads, 4 points and the 200×200 BEV as one
level in the traced stage. For 2D references, offsets are divided by
`(width,height)`. A 4D reference box instead scales offset by box `(w,h)`,
`0.5/num_points`. Value projection occurs before sampling; output projection,
dropout-disabled inference and identity residual occur after it. Observed query
families include 300 map queries and 901/921/931 tracking queries.

### Motion BEV interaction

Motion flattens `agent×mode`, predicts offsets/weights for selected future steps,
converts agent-relative trajectories to ego coordinates by adding detected
centers, normalizes them with `bev_range`, folds `query×step` into the common
core, then flattens `step×embedding` before output projection. Observed query
shapes are `[1,21,6,256]` inference and `[1,33,6,256]` training.

## Claim boundary and next gates

The common sampler is implemented and fixture-verified. The learned value,
offset, attention-weight and output projections already have Linear baselines,
and CUDA location generation now covers both 2D point and 4D reference-box
formulae with exact `(width,height)` normalization. Camera result scatter now
averages by each query's actual observation count, and temporal queue fusion
implements the exact previous/current mean. The adapters are not yet wired to
production context tensors. Camera visibility-list construction, temporal queue
packing, motion coordinate conversion, learned projections and residuals still
need adapter-level PyTorch boundary fixtures before graph equivalence can be
claimed. No latency result is recorded for fixture wrappers because they include
allocation and transfers.
