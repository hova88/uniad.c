# BEV query, position, motion and state contract

The track head owns 40,000 released 256-dimensional BEV queries. Before the
encoder, each query receives the same CAN-bus embedding:

`Linear(18,128) → ReLU → Linear(128,256) → ReLU → LayerNorm(256)`.

All CAN weights, affine parameters, reductions and accumulations remain FP32.
The final sum with the released FP16 query table is stored FP16.

Learned position uses two `[200,128]` FP16 tables. For token `y*200+x`,
channels 0–127 are `col_embed[x]` and channels 128–255 are `row_embed[y]`.
CUDA writes the already-flattened `[40000,256]` form consumed by the encoder.

Queries occupy arena bytes 189,388,800–209,868,800; positions occupy
209,868,800–230,348,800. A temporary 256-value FP32 CAN result follows them.
Both official PyTorch FP32 32-value boundary gates pass completely.

The context already owns a separate persistent `200×200×256` previous-BEV
allocation and clears it on scene change/reset. Ego-frame shift, rotation and
the temporal attention consumer remain the next unimplemented state boundary;
the presence of allocated state is not claimed as temporal equivalence.
