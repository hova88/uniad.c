# Limitations and claim boundaries

- The runnable `tiny-synthetic-v1` profile uses procedural 8×8 camera tensors,
  seeded scalar weights, and reduced 8×8 BEV / 16-query capacities.
- Its stages are pedagogical analogues. They are not the exact upstream
  operators, learned parameters, decoding, or preprocessing graph.
- No checkpoint or dataset is bundled. No dataset metric is measured.
- `production-nuscenes-stage2-v2` has a real 2459-tensor UAW2 export, strict
  parser, v2 I/O ABI and persistent CUDA allocation. Its official operator
  dispatch is incomplete, so inference still returns
  `UA_ERR_UNSUPPORTED_PROFILE` rather than running the synthetic graph.
- The production profile accepts only CUDA. The CPU request is explicitly
  unsupported and no fallback occurs.
- nuPlan remains future work: the pinned upstream revision does not establish a
  usable config, checkpoint, and evaluation chain.
