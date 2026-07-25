# UAW and UAF containers

Both containers are little-endian, version 1, and 64-byte payload aligned.
Readers check magic, version, endian tag, fixed capacities, offsets, arithmetic
bounds, NUL-terminated names, duplicate names, dtype/rank, truncation,
FNV-1a-64 payload checksums, and finite FP32 values.

`UAW1` has a fixed header and tensor directory. Each directory record carries
name, dtype, rank/dimensions, aligned offset, byte count, and checksum. Profile
identity and a deterministic seed live in the header.

`UAF1` carries profile, scene/frame identity, six-camera count, command,
timestamp, CAN-bus values, per-camera `3×3` intrinsics and `4×4`
camera-to-ego transforms, a `4×4` ego pose, ego-motion fields, declared result
capacities, and an aligned camera tensor payload. Synthetic calibration matrices
are deterministic identities plus camera offsets.

FNV detects accidental corruption; it is not a cryptographic signature.
