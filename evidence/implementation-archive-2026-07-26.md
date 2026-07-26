# UniAD production runtime implementation archive — 2026-07-26

This archive records what was executed, what artifacts were produced, and which
claims remain blocked. Paths are relative to `uniad.c` unless prefixed with
`../UniAD`.

## Environment and fixed contract

- GPU: NVIDIA RTX 4060 Ti 16GB, Ada SM89.
- CUDA production build: CUDA 12.4, `nvcc -arch=sm_89`.
- Runtime dependency boundary: libc, libm, CUDA Runtime and CUDA headers only;
  no cuBLAS, cuDNN, TensorRT, or host framework in production.
- Oracle/converter boundary: Python and PyTorch are offline evidence tools.
- Production CPU inference is unsupported and never falls back silently.
- The existing `tiny-synthetic-v1` API and two-frame oracle remain intact.

## Upstream training and oracle evidence

The sibling UniAD tree contains the exact data preparation, smoke configuration,
hook, and traces:

- `../UniAD/docs/evidence/stage2_mini_smoke_2026-07-26.json`: one real
  six-camera Stage 2 FP32 optimizer step; loss 271.3976135; all gradients finite;
  Track, Map, Motion, Occupancy and Planning parameters all changed; peak
  allocated 11,818,971,136 bytes and reserved 12,677,283,840 bytes.
- `../UniAD/docs/evidence/stage2_mini_smoke_fp16_experimental_2026-07-26.json`:
  negative evidence; experimental FP16 produced nonfinite tracking gradients and
  did not pass the strict gate.
- `../UniAD/evidence/pytorch_oracle_mini/`: official E2E checkpoint inference
  boundary, 2,230 module calls and 429 unique fixtures.
- `../UniAD/evidence/pytorch_train_mini/`: training boundary, 5,974 module calls,
  6,821 saved-tensor packs, 5,555 unpacks and 609 unique fixtures.
- CUPTI kernel activity was unavailable with `CUPTI_ERROR_INVALID_DEVICE`.
  Module CUDA Events remain useful spans but overlap and are not kernel timings.

## Operator inventory generation

Command:

```sh
python tools/build_operator_inventory.py \
  --infer-calls ../UniAD/evidence/pytorch_oracle_mini/calls.jsonl \
  --train-calls ../UniAD/evidence/pytorch_train_mini/calls.jsonl \
  --infer-map ../UniAD/evidence/pytorch_oracle_mini/checkpoint_runtime_map.json \
  --train-map ../UniAD/evidence/pytorch_train_mini/checkpoint_runtime_map.json \
  --json evidence/production-operator-inventory.json \
  --markdown evidence/production-operator-inventory.zh-CN.md \
  --web-json docs/assets/production-operator-inventory.json
```

Result:

- 8,204 total module calls (2,230 inference and 5,974 training);
- 710 unique operator/shape/stride/dtype/layout signatures;
- 64 coarse operator families;
- 2,459 unique checkpoint keys, 2,451 with an observed runtime consumer;
- 5,226 matched saved-tensor lifetimes;
- 14.220 GiB saved-reference upper bound, explicitly not allocator peak memory.

The JSON preserves full signatures, stage aggregates, source location,
requires-grad flags, per-mode call counts, CUDA Event distributions, 100 longest
and largest saved tensors, and all checkpoint consumer entries. The Chinese
Markdown is the human audit index; the same JSON is copied into the website.

## UAW2 and residency baseline

`tools/convert_uaw2.py` converted `uniad_base_e2e.pth` into a 264,909,568-byte
UAW2 payload containing all 2,459 tensors. FP16 is the default storage type;
numerically sensitive constants remain FP32 and integer tensors remain integer.
Every entry is 256-byte aligned and checksummed. The manifest records:

- checkpoint SHA256:
  `4dcdc34c09f964e377f608dbb764f003844a9ccc62b4a0469f94375302ed76d0`;
- config SHA256:
  `2d45be9b330f8e82456203e2d6374053bd7b80171c5316412dfa88b32e187b8c`;
- BN folding: false (not falsely claimed by the converter).

The CUDA production context permanently owns the payload, a 200×200×256 FP16
previous BEV, six raw-camera staging buffers, bounded track memory, a
nonblocking stream and a 512 MiB arena.
The full allocation/copy/reset path ran on the target GPU. Production inference
still returns an explicit unsupported-graph error.

The loader retains the validated UAW2 directory instead of discarding it after
checksum verification. `ua_model_find_tensor` and `inspect-model --tensor`
resolve exact checkpoint names to dtype, rank, shape, payload-relative offset
and byte length. CPU and CUDA tests cover a converted FP16 Conv weight and a
missing-key failure. Learned dispatch must use this contract rather than guess
raw file offsets.

The production entry now executes its first data stage before returning the
incomplete-graph gate: six 1600×900 strided BGR views are copied with
`cudaMemcpy2DAsync`, normalized with the pinned BGR means, converted to planar
FP16 and padded to 1600×928 inside the resident arena. The CUDA test allocates
the full context, validates exact 25,920,760-byte image-plus-metadata transfer
accounting, tests a valid device-weight subrange, and rejects bad image
dimensions and out-of-range weight spans. Calibration, pose, CAN bus, timestamp,
command and scene hash use a fixed 760-byte device ABI. A scene change and
explicit reset both clear previous BEV and track memory on the context stream.

The released-weight ResNet stem is now a real graph boundary. It resolves Conv,
BN affine and BN running-stat keys, validates their mixed UAW2 dtypes/shapes,
executes `Conv7×7/s2 → BN(eval)/ReLU → MaxPool3×3/s2`, and leaves
`6×64×232×400` FP16 in the arena. A read-only named debug boundary converts
only the requested prefix to FP32. For zero BGR input, its first 32 values match
the PyTorch checkpoint oracle at the FP16 gate. The full real-model test process
took 1.21 seconds with 366,996 KiB host max RSS; this includes setup and all
fixtures and is not a stem-only benchmark.

Layer1 now executes three real Caffe-style Bottleneck blocks. Block0 fuses the
projection branch into the final Conv3/BN/residual/ReLU kernel. Blocks1–2 write
their identity residual result in place. The arena relocates the stem input to
its tail and uses two 71,270,400-byte scratch intervals plus a
285,081,600-byte output. The real test takes 1.71 seconds, but the strict
PyTorch FP32 boundary gate is negative: 31/32 prefix samples pass and flat index
2 differs by 0.023549438. This baseline remains executable for graph
construction but is not recorded as graph-equivalent; the JSON negative
evidence and unchanged tolerance are retained.

Layer2 now executes four real Bottleneck blocks. Its downsampling block keeps
the Layer1 input live at arena offset 142,540,800, writes the
`6×512×116×200` output at offset 0, and uses two 35,635,200-byte scratch
regions above the Layer1 input. The highest touched arena byte is 498,892,800.
Blocks1–3 reuse the output as an in-place identity residual.

Seven 32-value windows are retained in a 448-byte device diagnostic allocation.
Layer1 block0/block1 pass and block2 contains its known single failure. Layer2
block0–2 selected windows pass 32/32, while block3 passes only 1/32 and the
final prefix passes 3/32. The real through-Layer2 test takes 2.39 seconds with
366,928 KiB host max RSS; it includes model loading and all fixtures and is not
a stage-only latency measurement. No intermediate boundary is downloaded
unless the debug API explicitly requests it.

Layer3 now executes all 23 released DCNv2 Bottlenecks. A first adapter reused a
split-pointer fixture ABI over packed `[N,27,H,W]` data and produced its first
Inf at block11. The corrected kernel uses explicit packed batch strides; all
blocks are finite and selected block0 and block22/final windows pass 32/32.
Layer4 executes its three DCNv2 blocks while retaining Layer2/3/4 outputs for
FPN. Its block0 window passes 32/32 and block2/final passes 30/32, so the
complete backbone is executable but not declared graph-equivalent. The
through-backbone process takes 26.99 seconds with 367,216 KiB host max RSS;
this is a scalar correctness baseline rather than a speed candidate.

The real FPN now consumes those three retained features, executes three lateral
1×1 convolutions, nearest-neighbor top-down additions, three 3×3 output
convolutions, and the stride-2 fourth level. Its four output shapes are
`6×256×{116×200,58×100,29×50,15×25}`. Prefix gates pass 30/32, 23/32,
32/32, and 16/32 respectively. FPN is executable and device-resident, but the
unchanged negative gates prevent an equivalence claim.
The through-FPN real-model regression takes 27.14 seconds with 367,208 KiB
host max RSS. It includes checkpoint loading and the complete fixture suite;
it is not a stage-only or warm-latency benchmark.

The BEVFormer input adapter now consumes all four FPN outputs, performs the
official camera-first spatial flatten/concatenation, and adds released FP32
camera/level embeddings before FP16 storage. The physical result is
`[6,30825,256]`; per-level first-token windows pass 31/32, 27/32, 25/32, and
29/32. Spatial shapes and level starts are fixed in a dedicated contract and
machine-readable gate.

Released BEV queries, the FP32 CAN-bus MLP/LayerNorm, and learned row/column
positions now execute on CUDA. Queries and positions are both stored as
`[40000,256]` FP16 encoder inputs. Their first 32 values each pass the official
FP32 PyTorch gate 32/32. Previous-BEV allocation/reset exists, while ego shift,
rotation, and temporal attention remain explicitly incomplete.

BEVFormer reference geometry now runs on CUDA: 2D temporal references, four
3D samples per pillar, rigid camera inverse, intrinsic projection, 1600×900
normalization and visibility masks. A nondegenerate identity-extrinsic fixture
locks the selected camera coordinate `(0.4375,0.3888889)` and visible flag.
The production API now rejects degenerate intrinsics and non-rigid transforms
instead of allowing undefined projection behavior.
Visibility is compacted into deterministic ascending query lists without
atomics. The fixture yields exactly 32 queries per camera and locks all camera0
indices.

The first released BEVFormer encoder layer now executes through its first-frame
TemporalSelfAttention and `norms.0`. The temporal path resolves all eight
official projection tensors, duplicates current BEV semantics when
`prev_bev == NULL`, computes two-queue/eight-head/four-point deformable
sampling with FP32 accumulation, averages the queues, applies the output
projection and input residual, then runs fixed-order FP32 LayerNorm statistics
with the released affine vectors. The selected PyTorch windows pass 32/32 at
both boundaries. The whole real visual-prefix regression takes 28.08 seconds;
this is not a stage-only or warm latency claim. Spatial cross-attention and the
continuous-frame temporal branch remain incomplete, so previous BEV is not yet
committed as valid encoder state.

Layer0 is now complete through SpatialCrossAttention, Norm1, the released
`256→512→256` FFN/residual, and Norm2. Spatial attention reuses scratch one
camera at a time, consumes the deterministic visibility lists, samples four
FPN levels with eight heads/eight points and four depth anchors, and accumulates
camera slots in FP32. Spatial, Norm1, FFN and Norm2 selected windows each pass
32/32 against a direct official-model PyTorch oracle. The expanded whole-prefix
run takes 27.79 seconds with 367,728 KiB host max RSS. Encoder layers 1–5 and
continuous-frame state are still absent; the production API continues to
return unsupported rather than emit a partial result.

The same strictly shape-checked resolver now executes encoder layers 1–5.
All six layer endpoints pass 32/32 official PyTorch windows and 36 internal
boundaries are snapshotted before arena reuse. The six-layer real prefix takes
29.80 seconds with 367,760 KiB host max RSS.

The CUDA context now also owns a separate aligned-previous-BEV buffer, scene
validity bit, CAN/ego translation shift and nearest-neighbor center-rotation
kernel. A commit function refuses inputs until all six encoder layers complete.
It is deliberately not called while the public inference transaction still
returns unsupported, so a failed call does not silently advance temporal state.

TrackFormer preparation splits the released 901×512 embedding into position
and content, and generates 901 normalized 3D reference points. Decoder layer0
then executes packed-QKV eight-head self-attention, three norms, single-level
four-point deformable BEV attention and its 256→512→256 FFN. All nine query
preparation/layer boundaries pass 32/32 selected windows. Reference refinement
At that checkpoint, decoder layers 1–5 were the next implementation step; the
following archived milestone supersedes that status.

### Six-layer TrackFormer decoder and refinement gate

The layer-0 template now executes all six released decoder layers. A strict
resolver consumes 22 decoder tensors and six regression-branch tensors per
layer. Each layer performs packed-QKV eight-head self-attention, one-level
four-point deformable BEV cross-attention, the 256→512→256 FFN and three
LayerNorm operations. Its released 256→256→10 regression branch then updates
`x/y/z` references with the same clamped inverse-sigmoid rule as
`projects/mmdet3d_plugin/uniad/modules/decoder.py`.

The arena retains all six complete `[901,256]` layer states, `[901,10]`
regression outputs and `[901,3]` refined-reference tensors below byte
4,000,000; large attention/MLP scratch is reused. Layer `n+1` is rejected
unless both decoder layer `n` and its reference refinement completed.

`tools/oracle_track_decoder.py` feeds the exact hashed full
`[40000,1,256]` CUDA BEV boundary into the released PyTorch decoder, isolating
this gate from known visual-prefix drift. The machine-readable
`track-decoder-six-layer-pytorch-oracle.json` and
`track-decoder-six-layer-gate.json` record 18 independent 32-value comparisons:
Norm2 state, raw regression and refined reference for every layer. All
`18×32/32` values pass the FP16 gate (`atol=5e-3`, `rtol=1e-2`); maximum
absolute error is `0.008960247` at layer-4 Norm2. The real complete-prefix
test passes in 30.85 seconds with 367,984 KiB host max RSS.

This is selected-window decoder/refinement evidence, not full-tensor or
end-to-end equivalence. Classification, decoded box and past-trajectory
outputs, track-ID/query/memory lifecycle, the other four task heads,
transactional previous-BEV commit and production result decode remain
incomplete, so the public production entry still returns
`UA_ERR_UNSUPPORTED_PROFILE`.

### Final TrackFormer output heads

The inference-only final decoder state now feeds the released layer-5
classification and past-trajectory branches. Classification executes two
`Linear→LayerNorm→ReLU` blocks followed by the 10-logit projection.
Past/future history executes two `Linear→ReLU` blocks followed by 16 offsets
(`8×2`). The cached layer-5 regression and layer-4 reference produce metric
10D boxes with FP32 inverse-sigmoid, sigmoid and the official
`[-51.2,-51.2,-5,51.2,51.2,3]` range transform.

The oracle tool accepts separately hashed full decoder-state, regression and
reference boundaries, so the head gate does not inherit accepted decoder
drift. Class logits, boxes and history offsets each pass 32/32 at the FP16
gate. Together with decoder/refinement, the machine-readable gate now records
21 passing boundaries. `tests/test_runtime.c` embeds the three official head
windows and the real UAW2 run passes in 30.69 seconds with 368,244 KiB host
max RSS.

### Track score and activation filter

The final 900 actor-query logits now remain on device for FP32 sigmoid,
stable lowest-class tie breaking and maximum-class scoring. A deterministic
query-order compactor applies the released base-E2E inference
`score_thresh=0.4` and
enforces the public result capacity of 300 without host synchronization or
normal-path D2H. Query 900 remains reserved for SDC and is not treated as an
actor.

The generic CUDA fixture covers equal logits, scores below and above the
threshold, capacity truncation and NaN rejection. Against the shared-head
PyTorch oracle, 32 FP32 scores pass at `atol=1e-4, rtol=1e-3`, 32 class IDs
match exactly, and the zero-image fixture's selected count is exactly zero.
The combined Track gate now contains 24 passing boundaries. The complete real
prefix including this filter passes in 30.73 seconds with 368,236 KiB host
max RSS.

Track-ID assignment, miss counters, query interaction, memory-bank update and
result decode are not claimed yet.

### Cross-frame Track state contract and tracker baseline

Source/config audit established that `base_e2e.py` overrides the detector
defaults: the production thresholds are 0.4 for new IDs and 0.35 for active
retention, with five-frame miss tolerance. QueryInteraction concatenates 901
fresh queries with every active actor, so the correct bounded decoder capacity
is 1,201 rather than the first-frame-only 901.

`track-state-contract.md` records the exact inference order, tensor shapes,
state-reset behavior and candidate/commit transaction. The CUDA context now
allocates separate 4,392,000-byte committed and candidate slabs. Each slab can
hold query/content embeddings, output embeddings, FP32 references/boxes/scores,
IDs, miss counters, four memory slots and their masks/save periods for 1,201
queries. Scene reset clears both slabs.

The first RuntimeTrackerBase kernel is an isolated correctness baseline. It
allocates global IDs in stable query order, preserves reserved negative IDs,
resets or increments miss counters, retires an ID on its fifth miss and
compacts active indices without exceeding capacity. Exact tests cover the
0.4/0.35 edges, reserved SDC, capacity, multiple new IDs, five consecutive
misses and NaN rejection.

This kernel is not wired into the unsupported public frame transaction yet:
the decoder must first be generalized from 901 to 1,201 queries and
MemoryBank/QueryInteraction must pass continuous-frame oracle gates.

The MemoryBank insertion baseline is now implemented separately. It reproduces
the inference ordering exactly: select only `(save_period == 0) &&
(score > 0)`, decrement positive periods, reset saved periods to three, shift
the four-slot padding mask left, project the current output embedding through
released-layout FP16 Linear with FP32 accumulation, shift the bank and append
the projection. Its generic fixture proves delayed saving, mask/bank shifts,
period countdown, projection values and NaN rejection.

QueryInteractionModule now has a released-weight CUDA baseline for up to 300
active tracks. The packed eight-head self-attention uses
`query_pos + output_embedding` for Q/K and `output_embedding` for V, followed
by its residual/Norm1, 256→256 FFN/residual/Norm2 and the separate
query-content FFN/residual/NormFeat. `update_query_pos=False` is taken from the
released config, so position-update weights are not consumed on this path.
A 32-active-query diagnostic uses the exact hashed C decoder state and
FP16-rounded initial query embedding as shared PyTorch inputs; its selected
window passes 32/32, bringing the combined Track gate to 25 boundaries.

During this gate a tiny/production macro collision was found: `UA_MAX_TRACKS`
is 8 for the legacy tiny result, while production capacity is
`UA_PROD_MAX_TRACKS=300`. All production score compaction, 1,201-query state
capacity, debug limits and QIM bounds now use the production constant. The
zero-candidate fixture could not reveal this error; the nonzero 32-query QIM
gate now prevents regression.

## First production CUDA correctness baselines

`src/cuda_backend.cu` now provides device kernels and isolated test wrappers for:

1. strided interleaved uint8 BGR → planar FP16 normalize and padding;
2. FP16 Linear with `[out,in]` weights, optional bias and FP32 accumulation;
3. FP16 LayerNorm storage with fixed-order FP32 mean/variance;
4. NCHW/OIHW FP16 Conv2d with bias, padding, stride and FP32 accumulation;
5. stable FP32 Softmax;
6. align-corners-false FP32 bilinear resize with zero boundary sampling;
7. HWC multi-channel FP32 deformable bilinear gather;
8. stable FP32 TopK with deterministic lower-index tie breaking.
9. FP16-value multi-scale deformable-attention sampling with normalized
   coordinates and FP32 accumulation.
10. MMCV-compatible modulated DCNv2 core with FP16 feature/weight, FP32
    offset/mask/accumulation, and fused bilinear gather-convolution.
11. FP32 deformable sampling-location generation for normalized 2D reference
    points and 4D reference boxes.
12. camera-visible query scatter with per-query observation averaging;
13. temporal previous/current queue mean.
14. mixed FP16-affine/FP32-stat BatchNorm inference fused with ReLU;
15. FP16 MaxPool with negative-infinity padding.
16. FP16-logit sigmoid/max-class scoring and deterministic threshold
    compaction with a fixed output capacity.
17. deterministic RuntimeTrackerBase ID allocation, miss counting,
    five-frame retirement and active-index compaction.
18. periodic FP16 MemoryBank projection/insertion with save-period and
    padding-mask shifts.
19. released-weight QueryInteraction self-MHA, residual FFN and query-content
    update for a variable active-query count up to 300.

`tests/test_runtime.c` compares them against direct formulae or the scalar C
operators. It covers row padding, output padding, Linear tails, production
LayerNorm dimension 256 with affine gamma/beta, Conv2d stride and boundary
padding, logits near 1000, zero dimensions, invalid stride, and NaN/Inf
rejection. Default gates are FP16 `atol=5e-3, rtol=1e-2` and FP32
`atol=1e-4, rtol=1e-3`.

These kernels are correctness baselines, not optimized kernels. LayerNorm and
Softmax intentionally use one CUDA thread per row to preserve a fixed reduction
order. Fixture wrappers include allocation and H2D/D2H, so their wall time is
not a production-kernel benchmark.

## Commands executed for the CUDA gate

```sh
make CUDA=ON test
```

This passed:

- CUDA runtime C tests, including the nineteen new operator gates;
- CUDA CLI tests;
- UAW2 structure, profile, checksum, corruption and truncation tests;
- two-frame tiny PyTorch oracle (`atol=1e-5, rtol=1e-4`);
- static website validation.

Final verification passed:

- `make test`;
- `make CUDA=ON test`;
- `make sanitize` with ASan/UBSan and leak detection;
- CUDA 12.4 / SM89 CMake configure, build and `ctest`;
- Python bytecode compilation for converter and inventory generator;
- `tools/validate_site.py`;
- `git diff --check`.

Key artifact SHA256 values at this milestone:

- operator inventory JSON:
  `d1dfb0b0f528126858c781301b1aa9bf379e8728bc4fe405870ddafa418f70cf`;
- inventory generator:
  `4f068f9c0fc1569f4f6e531eb42475e19fd0292c76835697009b24a02af6aa92`;
- CUDA backend:
  `b6c15ab934565e819e364d310082d2496918ea22c5c2b38658798f9cebf40c3d`;
- CUDA/runtime test:
  `5efb38b175f116df345ce51678b0c2867ed89b4ee8e28e01548eeaf7e5c3796d`;
- website HTML:
  `d8149799aa0bd0dcf479db852f2890b5ef716ce8f35fecd7c08d9e8da920bf9d`.

## Design disposition

The scalar C path remains the readable semantic oracle. Dense UniAD weights are
kept resident; streamed expert-style weight loading was rejected as a mismatch
for this graph. Correctness baselines precede SM89 Tensor Core tiling. Candidate
promotion requires parent, mechanism, numerical result, cold/warm latency,
memory, transfer volume and retain/reject reason in `candidate-ledger.md`.

The visual backbone and FPN now execute. Current graph blockers are the learned
temporal/spatial/decoder/motion deformable-attention adapters, remaining
transformer primitives, five task heads, query and memory state transitions,
GPU decode and fixed-iteration occupancy-aware planning optimization. Until
those pass raw stage-tensor and continuous-scene tests, no production E2E or
speed claim is made.
