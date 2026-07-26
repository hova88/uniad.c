# BEVFormer encoder layer 0 CUDA contract

This is an executable-prefix contract, not a complete BEVFormer equivalence
claim. It pins the complete first released encoder layer for the first frame
(`prev_bev == NULL`).

## TemporalSelfAttention

- Input query: `[1,40000,256]` FP16, released BEV query plus the FP32 CAN-bus
  MLP result.
- Position: `[1,40000,256]` FP16 learned column/row encoding.
- Reference points: `[2,40000,1,2]`; the first-frame path duplicates current
  2D references for the two temporal queues.
- Value: current query projected once and logically shared by both queues.
- Offset projection: `[query, query + position]`, `512 → 128`.
- Attention projection: `512 → 64`; softmax is applied independently to each
  head and queue over four points.
- Sampling: eight heads, one `200×200` level, four points, align-corners-false
  bilinear interpolation with zero outside the grid, FP32 accumulation.
- Queue fusion: deterministic mean over previous/current queue slots.
- Output: released `256 → 256` projection plus the unpositioned input query
  residual, stored as `[1,40000,256]` FP16.

The runtime avoids materializing the 512-wide concatenated query: the Linear
kernel reads the two 256-wide sources directly. Dropout is the identity in eval
mode.

## Norm0

`encoder.layers.0.norms.0` consumes the temporal result. One CUDA thread owns
each 256-wide row, computes mean and variance in fixed-order FP32, applies the
released FP16 affine vectors and stores FP16. This is the correctness baseline;
a parallel block reduction is a later measured candidate.

## SpatialCrossAttention and Norm1

The spatial path projects the physical `[6,30825,256]` visual sequence once.
It then processes cameras in order using the stable visible-query index lists,
reusing one `[40000,512]` offset buffer, one `[40000,256]` attention buffer,
and one `[40000,256]` sampled buffer. Each query uses eight heads, four image
levels and eight points. The eight points are paired with four pillar depths in
the official `point % 4` order. Sampling is align-corners-false with FP32
accumulation.

Per-camera results scatter into `[40000,256]` FP32 slots. A deterministic
camera-observation count divides the slots before the released output
projection and Norm1. No atomics are used.

## FFN and Norm2

The released `256 → 512 → 256` FFN uses ReLU, eval-mode identity dropout and
an input residual. Its hidden and output tensors remain FP16 while every dot
product accumulates in FP32. Norm2 uses the same fixed-order FP32 statistics as
Norm0/Norm1. Its output is the complete layer0 encoder result.

## Evidence and limits

`temporal-attention-layer0-gate.json` and
`encoder-layer0-norm0-gate.json` record 32/32 temporal boundaries.
`encoder-layer0-spatial-ffn-gate.json` records 32/32 at spatial attention,
Norm1, FFN and Norm2. All use `atol=5e-3, rtol=1e-2`.

The gates do not establish full-tensor parity, continuous-frame previous-BEV
alignment, encoder layers 1–5, decoded output, or nuScenes accuracy. Previous
BEV is therefore still not committed as valid production state.
