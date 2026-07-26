# ResNet-101 bottleneck production contract

## Layer 1 executable layout

Layer 1 has three Caffe-style bottlenecks at `6×232×400`:

- block 0: `64 → 64 → 64 → 256`, plus `64 → 256` projection;
- blocks 1–2: `256 → 64 → 64 → 256`, identity residual.

Conv weights and BN affine tensors are FP16; running statistics and
accumulation are FP32. Conv1/Conv2 are followed by BN+ReLU. Conv3 is fused with
BN, residual addition and final ReLU.

The stem output is relocated to the 71,270,400-byte arena tail. Two 64-channel
scratch tensors occupy offsets 0 and 71,270,400. Block-0 output starts at
142,540,800. For blocks 1–2, final Conv3 reads the second scratch tensor and
overwrites the residual/output in place. No other thread reads a different
residual element, so this alias is race-free.

After layer1 completes the old stem boundary is invalidated. Debug access returns
`UA_ERR_PROFILE` rather than exposing recycled bytes.

## Strict numerical gate

The graph executes with real UAW2 weights and produces `6×256×232×400`.
Against the released-checkpoint FP32 PyTorch boundary, 31 of the first 32
zero-image samples pass the default FP16 gate. Flat index 2 does not:

```text
CUDA scalar FP16-storage baseline: 0.882324219
PyTorch FP32 oracle:              0.858774781
absolute error:                   0.023549438
allowed error:                    0.013587748
```

Layer1 is therefore an executable graph baseline, not a promoted
graph-equivalent candidate. `resnet-layer1-strict-gate.json` preserves this
negative evidence. The first 192 bytes of the shared GPU diagnostic area
preserve selected prefixes without intermediate D2H. Nonzero windows from
block0 and block1 each pass
32/32; the failure first appears in block2. The next comparison can therefore
focus on block2 FP32 storage, scalar FP16 storage and SM89 Tensor Core
accumulation without changing the tolerance.

## Layer 2 executable layout

Layer 2 consumes `6×256×232×400` and executes four Caffe-style bottlenecks:
block 0 places stride 2 on Conv1 and on the projection branch, then blocks 1–3
use identity residuals at `6×512×116×200`.

No relocation is needed. The Layer1 input remains at arena offset 142,540,800
through the block-0 projection. Layer2 output occupies offset 0 through
142,540,800. Two `6×128×116×200` scratch tensors occupy offsets 427,622,400
and 463,257,600, ending at 498,892,800. These four live intervals do not
overlap. After block 0, identity blocks update the Layer2 output in place.

The selected nonzero diagnostic windows for blocks 0, 1 and 2 pass 32/32.
Block 3 is the first failing boundary: only 1/32 values in its selected window
passes. The final output prefix passes 3/32; flat index 3 is
`0.018539429` versus the FP32 oracle `0.004144375`. This is executable negative
evidence, not a numerical-equivalence claim. Exact counts and the unchanged
tolerance are stored in `resnet-layer2-strict-gate.json`.

## Layer 3 DCNv2 execution

Layer 3 executes 23 Bottlenecks and produces `6×1024×58×100`. Every Conv2 is
the released DCNv2 pack: a learned 3×3 Conv generates packed
`[N,27,H,W]` FP32 offset/mask values, the last nine channels receive sigmoid,
and the deformable gather-convolution consumes the packed batch stride
directly. BN/ReLU and residual epilogues retain the same mixed-precision policy.

An early adapter incorrectly treated the packed buffer as separately contiguous
`[N,18,H,W]` and `[N,9,H,W]` arrays. With `N=6`, its mask pointer crossed batch
boundaries; block11 was the first nonfinite block and the final window contained
Inf. That candidate is rejected in the ledger. Explicit packed-27 indexing
eliminates all nonfinite values across 23 blocks.

The corrected block0 window at offset 5697 and block22/final window at offset
195 both pass 32/32 against PyTorch/MMCV FP32 oracles. This is selected-window
evidence, not full-tensor equivalence; `resnet-layer3-strict-gate.json` records
the exact samples and rejected ABI.

## Layer 4 and retained FPN inputs

Layer 4 executes three packed-27 DCNv2 Bottlenecks and produces
`6×2048×29×50`. The arena simultaneously retains:

- Layer2 `6×512×116×200` at offset 0;
- Layer3 `6×1024×58×100` at offset 181,934,400;
- Layer4 `6×2048×29×50` at offset 253,204,800.

This is required by the FPN contract. Layer4 scratch lives between Layer2 and
Layer3, so none of the three outputs is recomputed or overwritten.

All three Layer4 blocks remain finite. Block0's selected window passes 32/32.
Block2/final passes 30/32; the two failures are `1.68652344` versus
`1.65022302` and `0.618164062` versus `0.634063601`. The complete backbone is
therefore executable, but Layer4 is not promoted as graph-equivalent.
