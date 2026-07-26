# Production ResNet-101 stem contract

## Exact graph

```text
6×3×928×1600 FP16 BGR-normalized
  → Conv2d [64,3,7,7], stride 2, pad 3, no bias
  → 6×64×464×800
  → BatchNorm(eval, eps=1e-5) → ReLU
  → MaxPool 3×3, stride 2, pad 1, floor output
  → 6×64×232×400 FP16
```

UAW2 mixed storage is intentional: Conv weight, BN gamma and BN beta are
FP16; BN running mean and running variance are FP32. Convolution and
normalization accumulate in FP32 and activation boundaries are FP16.

The arena simultaneously holds normalized input (53,452,800 bytes), stem
activation (285,081,600 bytes), and pooled output (71,270,400 bytes), totaling
409,804,800 bytes. No allocation occurs inside the production dispatcher.

## Verification

`UA_TEST_PRODUCTION_MODEL=build/uniad_base_e2e.uaw2
build/cuda/test_runtime` loads and checks all 2,459 tensors, creates the real
context, uploads six 1600×900 images and metadata, resolves the five stem keys,
and executes the device chain.

For a deterministic all-zero uint8 BGR input, PyTorch applies the pinned
checkpoint to the same normalized/padded tensor. Its first 32 pooled values are
embedded in the C test. The named boundary `production.resnet_stem` copies only
the requested FP16 prefix as FP32; every value passes
`atol=5e-3, rtol=1e-2`. Unknown boundary names fail explicitly.

The whole test process measured 1.21 seconds and 366,996 KiB host max RSS. This
includes model I/O/checksums, context allocation and all other fixtures; it is
not stem latency and is not a production performance claim. The terminal status
now occurs after the executable layer1 baseline. The strict stem boundary itself
remains verified; layer1 has a separately recorded negative numerical gate.
