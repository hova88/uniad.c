# BEV reference and six-camera projection contract

The encoder generates 40,000 normalized 2D temporal reference points and four
3D samples per 200×200 pillar. The point-cloud range is
`[-51.2,-51.2,-5,51.2,51.2,3]`. Reference tensors retain FP16 because they are
created from FP16 BEV queries; camera projection is FP32.

The public API supplies `camera_to_ego` and a 3×3 intrinsic for each camera.
CUDA applies the rigid inverse as `Rᵀ(p_ego-t)`, multiplies by the intrinsic,
divides by positive projected depth, then normalizes x by 1600 and y by 900.
Visibility requires finite coordinates, positive depth, and strict `(0,1)`
bounds on both axes. Output layouts exactly match the encoder after its
permutes: `[camera,query,depth,2]` and `[camera,query,depth]`.

Arena layout after query preparation:

- ref2d FP16: 160,000 bytes;
- ref3d FP16: 960,000 bytes;
- camera references FP32: 7,680,000 bytes;
- visibility uint8: 960,000 bytes.
- stable visible-query indices: 960,000 bytes;
- six uint32 counts: 24 bytes.

The last byte is 241,070,104. A single deterministic thread per camera scans
queries in ascending order, avoiding atomic ordering variance. The
nondegenerate calibration oracle and exact
selected projection are recorded in `bev-camera-geometry-gate.json`.
Degenerate intrinsics and non-rigid transforms are rejected before any camera
upload or graph execution.
