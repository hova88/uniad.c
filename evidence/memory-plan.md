# Memory and residency plan

| Class | Tiny CPU ownership | Lifetime |
|---|---:|---|
| Resident weights/model | about 384 B | model |
| Temporal BEV | 4 KiB | context |
| Current + spatial BEV | 8 KiB stack scratch | inference |
| Frame cameras | 4.5 KiB | loaded frame |
| Canonical result | bounded struct | caller |

The measured owned-memory counters include model and context allocations.
Temporary stack storage and caller-owned frames/results are reported separately
by contract, not disguised as allocator ownership. The tiny gates are 256 MiB
host and 256 MiB device.

Dense demo weights should remain resident. There is no reason to copy a sparse
expert-streaming scheme into this dense graph. A complete CUDA implementation
would keep weights and intermediate BEV tensors device-resident and transfer only
inputs and compact results. The current CUDA file validates an explicit device
boundary but is not performance evidence for a complete accelerated graph.
# Production v2 measured allocation contract

The current CUDA residency baseline owns:

- converted UAW2 payload: 264,909,568 bytes for the pinned checkpoint/config;
- six raw-camera staging buffers: 25,920,000 bytes
  (`6×1600×900×3` uint8);
- fixed device metadata: 760 bytes;
- boundary diagnostic prefixes: 5,632 bytes (16 visual windows,
  36 encoder windows and 36 reserved track-decoder windows);
- per-stage nonfinite status: 4 bytes;
- previous BEV FP16: `200×200×256`, 20,480,000 bytes;
- aligned previous-BEV scratch: another 20,480,000 bytes;
- committed and candidate Track state: two 4,392,000-byte fixed-capacity
  slabs. Each supports 1,201 queries (`901` fresh plus at most `300` active),
  query/output embeddings, FP32 references/boxes/scores, IDs, miss counters,
  `4×256` FP16 memory, masks, save periods and scalar counters;
- reusable activation arena: 512 MiB.

This totals 877,450,876 bytes (about 0.817 GiB) plus CUDA allocator/runtime
reserve. It is an
allocation baseline, not the final liveness-derived arena size. Future kernels
must reuse this arena and may reduce it only after boundary fixtures show that
all overlapping lifetimes fit. Images/metadata are the only intended H2D
inputs and the compact production result is the only intended D2H output.

The production entry performs one `cudaMemcpy2DAsync` per camera so host row
padding is never treated as pixels. It transfers exactly 25,920,000 image bytes
plus a fixed 760-byte metadata packet, writes normalized/padded
`6×3×928×1600` FP16 into the arena, and leaves both resident. This boundary is
executable; later graph stages remain gated.

The verified stem arena interval is:

| Tensor | Bytes | Offset |
|---|---:|---:|
| normalized cameras `6×3×928×1600` FP16 | 53,452,800 | 0 |
| Conv/BN/ReLU `6×64×464×800` FP16 | 285,081,600 | 53,452,800 |
| pooled stem `6×64×232×400` FP16 | 71,270,400 | 338,534,400 |

The simultaneous stem footprint is 409,804,800 bytes, leaving 127,066,112 bytes
of the 512-MiB arena. Later ResNet stages therefore must recycle the normalized
input and pre-pool stem intervals after their last use; retaining every backbone
activation would violate the arena contract.

Layer1 relocates the 71,270,400-byte pooled stem to the arena tail, uses two
71,270,400-byte 64-channel scratch regions at the front, and places its
285,081,600-byte 256-channel output at offset 142,540,800. Block0 fuses its
projection residual into the final epilogue. Blocks1–2 overwrite that output
in-place only after reading the same residual element. Completing layer1
explicitly ends the stem boundary lifetime.

Layer2 keeps the 285,081,600-byte Layer1 input at offset 142,540,800 while its
block-0 projection is live. Its 142,540,800-byte output starts at offset 0.
Two 35,635,200-byte scratch intervals begin at offsets 427,622,400 and
463,257,600, so the maximum touched arena byte is 498,892,800. Identity blocks
1–3 then update the Layer2 output in place.

Layer3 keeps its block-0 Layer2 input at offset 0, places the first
17,817,600-byte Conv1 scratch at offset 142,540,800, the 3,758,400-byte packed
FP32 offset/mask tensor next, the second 17,817,600-byte scratch next, and the
71,270,400-byte output at offset 181,934,400. The maximum touched byte is
253,204,800. Identity blocks recycle the arena front for scratch.

Layer4 retains Layer2 at offset 0 and Layer3 at offset 181,934,400. Its
8,908,800-byte first scratch begins at 142,540,800, followed by a
939,600-byte packed FP32 offset/mask tensor and a second 8,908,800-byte
scratch. Layer4 output occupies 253,204,800 through 288,840,000. Thus all three
FPN inputs coexist while staying well below the 512-MiB arena boundary.

Twelve 32-value FP16 windows (768 bytes) survive outside the arena for backbone
numerical localization. A four-byte stage status reports the first nonfinite
Layer3 block; sentinel 23 means all blocks are finite. Boundary windows remain
on device until explicitly requested. The status copy is counted separately
from normal result D2H while this correctness baseline remains incomplete.
Four additional 32-value windows preserve FPN prefixes once spatial attention
reuses their arena interval, bringing the diagnostic allocation to 1,024 bytes.

FPN first writes three lateral tensors after the retained Layer4 output:
71,270,400, 17,817,600, and 4,454,400 bytes. The last lateral byte is
382,382,400. After all lateral convolutions finish, the old backbone intervals
are dead; four FPN outputs reuse the arena front and occupy 94,694,400 bytes in
total. Top-down nearest additions update lateral tensors in place. No FPN
boundary is downloaded unless requested through the debug API.

The BEVFormer input adapter preserves all four FPN outputs and writes the
equally sized `[6,30825,256]` FP16 flattened/embedded tensor at arena offset
94,694,400 through 189,388,800. Camera and level embeddings are read directly
from resident FP32 UAW2 storage. This layout leaves more than 347 MiB for BEV
queries, attention scratch, and encoder output.

BEV queries and learned positions each occupy 20,480,000 bytes at offsets
189,388,800 and 209,868,800. The normalized FP32 CAN embedding uses the next
1,024 bytes. The persistent previous-BEV remains outside the arena so scene
reset and continuous-frame state do not compete with encoder activations.

Reference geometry begins at aligned offset 230,350,080. Ref2d, ref3d,
six-camera FP32 projected coordinates, uint8 visibility, stable uint32 query
indices and six counts occupy 10,720,024 bytes and end at 241,070,104. They
coexist with the visual sequence, BEV
queries and positions and leave over 296 MiB of the arena unused.

The first-frame layer0 TemporalSelfAttention scratch starts at aligned byte
241,070,336. Projected values, 128 FP16 offsets/query, 64 FP16 attention
weights/query, sampled output and projected/residual output end at byte
317,870,336. The following `[40000,256]` FP16 Norm0 output occupies
317,870,336 through 338,350,336. Peak arena ownership remains the fixed 512 MiB;
no attention boundary is copied unless the read-only debug API requests it.
The temporal output and its scratch may be recycled only after Norm0, while
queries, visual features, camera references, visibility indices and Norm0 must
remain live for spatial cross-attention.

Spatial cross-attention projects the visual sequence into the now-dead FPN
interval at bytes 0 through 94,694,400. It reuses temporal scratch for one
camera's offsets and attention, places sampled values at 338,350,336, FP32
scatter slots at 358,830,336 and spatial output at 399,790,336. Norm1 ends at
440,750,336. Processing cameras sequentially avoids six simultaneous
`[40000,512]` offset tensors.

Layer0 FFN uses the final free arena interval: `[40000,512]` hidden values at
440,750,336, residual output at 481,710,336, and Norm2 at 502,190,336 through
522,670,336. The maximum touched byte leaves 14,200,576 bytes of the fixed
512-MiB arena. Four 32-value FPN snapshots are retained in the separately
accounted diagnostic buffer before their arena storage is reused.

Encoder layers 1–5 reuse exactly the same scratch intervals. Each of the six
layers saves six 32-value windows outside the arena before reuse. The final
Norm2 remains at bytes 502,190,336 through 522,670,336 as the decoder BEV.

Track query preparation reuses the arena front for `[901,256]` query-position
and query tensors plus `[901,3]` FP32 initial references. The six decoder
layers retain their complete `[901,256]` Norm2 states in
`[1,048,576, 3,816,448)`, their `[901,10]` regression outputs beginning at
byte 3,817,472, and six `[901,3]` FP32 refined-reference sets beginning at
byte 3,932,160. All retained sparse state ends before the decoder scratch at
byte 4,194,304. The final inference heads retain `[901,10]` FP16 class logits
at byte 4,000,000, `[901,16]` FP16 past/future-history offsets at byte
4,020,000 and `[901,10]` FP32 decoded box fields at byte 4,050,000; all end
at byte 4,086,040. Actor scores, class IDs, at most 300 selected query indices
and selected count occupy bytes 4,087,000 through 4,096,204 without
overlapping the decoder scratch. Packed QKV,
self-attention, three norms, projected BEV,
deformable offsets/weights, sampled values, regression MLP scratch and FFN
scratch are reused for every layer below byte 45,550,080 while the final
encoder BEV remains at the arena tail.

The real training trace contains 5,226 matched saved-tensor lifetimes. Summing
every live saved reference produces a 14.220 GiB upper bound, but this double
counts views and repeated saves and excludes allocator behavior, so it is not an
arena size. The independently measured PyTorch training peak was 11.857 GiB
reserved. Inference reserved 2.805 GiB. The production arena remains 512 MiB
until the complete inference-only tensor DAG supplies first/last-use intervals;
training saved-tensor lifetimes are diagnostic evidence, not a safe substitute
for that DAG.
