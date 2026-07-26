# BEVFormer visual-input adapter contract

The released four-level FPN tensors have NCHW shapes
`6×256×{116×200,58×100,29×50,15×25}`. Official
`PerceptionTransformer.get_bev_features` flattens each spatial plane, permutes
camera before spatial tokens, adds a 256-value camera embedding and level
embedding, concatenates levels, then permutes to
`[camera, ΣHW, batch, channel]`.

Production batch is one and the six leading FPN images are the camera axis, so
the CUDA physical layout is `[6,30825,256]`. Spatial shapes are
`[[116,200],[58,100],[29,50],[15,25]]`; level starts are
`[0,23200,29000,30450]`. Both embedding tables are consumed as released FP32
UAW2 tensors. Addition accumulates in FP32 and stores FP16.

The adapter writes 94,694,400 bytes at arena offset 94,694,400, directly after
the four still-live FPN outputs. No allocation or transfer occurs. Camera-zero
32-value windows at the first token of each level pass 31/32, 27/32, 25/32,
and 29/32. The exact failure indices are machine-readable in
`bevformer-input-adapter-gate.json`. These are inherited numerical gates, not
evidence of a layout failure.
