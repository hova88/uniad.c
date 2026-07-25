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
