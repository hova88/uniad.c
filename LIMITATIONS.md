# Limitations and claim boundaries

- The runnable `tiny-synthetic-v1` profile uses procedural 8×8 camera tensors,
  seeded scalar weights, and reduced 8×8 BEV / 16-query capacities.
- Its stages are pedagogical analogues. They are not the exact upstream
  operators, learned parameters, decoding, or preprocessing graph.
- No checkpoint or dataset is bundled or downloaded. No dataset metric is
  measured.
- `production-nuscenes-stage2` is metadata only. Inference returns
  `UA_ERR_UNSUPPORTED_PROFILE`.
- CUDA is an optional route for the full demo boundary. CUDA results are not
  claimed unless that build and its oracle checks run on the current host.
- nuPlan remains future work: the pinned upstream revision does not establish a
  usable config, checkpoint, and evaluation chain.
