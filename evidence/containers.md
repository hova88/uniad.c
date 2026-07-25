# UAW and UAF containers

Both containers are little-endian, version 1, and 64-byte payload aligned.
Readers check magic, version, endian tag, fixed capacities, offsets, arithmetic
bounds, NUL-terminated names, duplicate names, dtype/rank, truncation,
FNV-1a-64 payload checksums, and finite FP32 values.

`UAW1` has a fixed header and tensor directory. Each directory record carries
name, dtype, rank/dimensions, aligned offset, byte count, and checksum. Profile
identity and a deterministic seed live in the header.

`UAF1` carries profile, scene/frame identity, six-camera count, command,
ego-motion fields, and an aligned camera tensor payload. The production contract
reserves calibration, timestamp, pose, and CAN-bus semantics, but the version-1
synthetic file materializes only fields used by its graph.

FNV detects accidental corruption; it is not a cryptographic signature.
