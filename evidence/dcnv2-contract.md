# Modulated DCNv2 production contract

The traced ResNet-101 stages use MMCV `ModulatedDeformConv2dPack` at:

- layer 3: `N×256×58×100 → N×256×58×100`, 23 blocks;
- layer 4: `N×512×29×50 → N×512×29×50`, 3 blocks;
- `N=6` inference and `N=6/12` training calls.

Each pack first applies a learned 3×3 `conv_offset` producing 27 NCHW channels.
MMCV chunks these into offset and mask, applies sigmoid to the 9 mask channels,
then invokes modulated deform convolution. Checkpoint evidence includes
`conv_offset.weight [27,C,3,3]`, `conv_offset.bias [27]`, and deform weight
`[C,C,3,3]`.

## Frozen core layout

- input: FP16 NCHW;
- weights: FP16 OIHW;
- offsets: FP32 `[N,2KH×KW,OH,OW]`;
- masks after sigmoid: FP32 `[N,KH×KW,OH,OW]`;
- output: FP16 baseline, with FP32 accumulation.

For kernel point `k=ky×KW+kx`, channels `2k` and `2k+1` contain
`(delta_y, delta_x)`. MMCV's `torch.chunk(out, 3)` separates the first 18
offset channels from the last 9 mask channels; concatenating the two
9-channel offset chunks preserves their original interleaved order. The sampled
coordinate is:

```text
y = oy*stride_y - pad_y + ky*dilation_y + delta_y
x = ox*stride_x - pad_x + kx*dilation_x + delta_x
```

Bilinear neighbors outside the input contribute zero. The sampled feature is
multiplied by `mask[k,oy,ox]`, then by the OIHW weight.

## Oracle

The fixture uses `N=1,Cin=2,H=4,W=5,Cout=3,K=3,stride=1,pad=1,dilation=1`
with fractional offsets and nonuniform masks. Offset values include an explicit
per-channel term; this is essential because the previous periodic fixture made
all 18 channel planes identical and therefore could not distinguish split from
interleaved channel order. Its C scalar reference is checked elementwise.

The first twelve values from the pinned MMCV CUDA operator are embedded in
`tests/test_runtime.c` and checked at `1e-6` against the scalar formula before
the FP16 CUDA result is checked at `atol=5e-3, rtol=1e-2`. The channel-varying
fixture rejects the split-half interpretation and independently fixes both
channel order and coordinate semantics.

## Production adapter

Layer3 now runs the 27-channel offset/mask Conv2d, applies sigmoid to the last
nine channels, then launches the core without an im2col buffer. Production
indexes the packed `[N,27,H,W]` tensor explicitly; it must not reuse the
isolated wrapper's two-pointer split ABI because mask planes are interleaved
between batches. The correctness wrapper still deliberately allocates and
transfers isolated split inputs and is not a latency measurement.
