# Production v2 continuation handoff

This branch is a reviewable checkpoint, not a completed production runtime.
Keep its pull request in draft and do not merge it as an end-to-end UniAD
implementation.

## What is ready

- The legacy `tiny-synthetic-v1` API, demo and CPU oracle remain intact.
- The versioned production ABI accepts six strided BGR views, calibration,
  ego/CAN/timestamp/scene metadata and fixed-capacity five-task results.
- The offline converter and strict UAW2 parser cover the released 2,459-tensor
  `uniad_base_e2e.pth` checkpoint, with hashes, alignment and corruption tests.
- CUDA owns persistent weights, image/metadata staging, BEV/query state and
  separate committed/candidate track-memory slabs.
- Nineteen isolated CUDA correctness baselines cover the visual prefix,
  BEVFormer encoder/decoder prefix and Track state helpers. The checked-in
  evidence records positive numerical gates and explicit negative gates.
- The website, English contracts and Chinese operator inventory describe the
  same implementation boundary.

## Current hard boundary

`ua_context_infer_v2` still returns `UA_ERR_UNSUPPORTED_PROFILE` for
`production-nuscenes-stage2-v2`. This is intentional. The code does not yet
implement or validate the full stateful Track/Map/Motion/Occ/Planning graph,
so this branch makes no claim about official accuracy or production speed.

In particular, the six-layer Track decoder diagnostic still uses the
901-query detection slab. It is not yet integrated with the 300 propagated
queries, continuous-frame memory attention, transactional state commit, or
the other four task heads.

## Resume plan

1. Generalize the Track decoder to the full 1,201-query layout, including
   velocity-aware reference updates, propagated-query merge and exact
   per-layer selection semantics.
2. Implement memory temporal attention and connect RuntimeTrackerBase,
   MemoryBank and QueryInteraction to candidate state. Add two-scene,
   continuous-frame oracle gates and commit state only after a successful
   frame.
3. Implement and gate the Map decoder/head and its raster/vector decoding.
4. Implement Motion, occupancy and occupancy-aware planning in dependency
   order, retaining raw boundary tensors and decoded-result parity checks.
5. Run multi-frame end-to-end PyTorch/CUDA comparisons. Only after numerical
   gates pass, optimize the retained kernels with residency, arena reuse,
   vectorization, fusion and CUDA Graphs, then compare warm p50/p95 and peak
   VRAM on the target RTX 4060 Ti 16 GB.
6. Treat full nuScenes trainval metrics as a separate gate when the complete
   dataset is available; mini data must not be used for an official accuracy
   claim.

## Re-entry checks

```sh
make test
make sanitize
make CUDA=ON test

env UA_TEST_PRODUCTION_MODEL=build/uniad_base_e2e.uaw2 \
    UA_TEST_QUERY_INTERACTION=1 \
    build/cuda/test_runtime
```

Read `LIMITATIONS.md`, `evidence/model-contract.md`,
`evidence/track-state-contract.md`, and
`evidence/implementation-archive-2026-07-26.md` before extending the public
inference transaction.
