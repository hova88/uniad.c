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

## UAW2 production container

`UAW2` is little-endian and uses 256-byte tensor alignment. Its fixed header
contains profile/version, tensor count, directory/data/file bounds, the source
checkpoint SHA-256, config SHA-256 and directory SHA-256. Each fixed directory
entry contains a NUL-terminated 128-byte name, dtype, rank, up to eight
dimensions, aligned offset, byte count and tensor SHA-256.

The offline converter reads all 2459 tensors from `uniad_base_e2e.pth`.
Floating tensors are FP16 by default; named numerically sensitive constants
remain FP32 and integer buffers retain their integer dtype. The current
container keeps PyTorch OIHW/OI layouts because the planned implicit-GEMM
kernels consume those layouts directly. BN folding is explicitly `false` in
the manifest until graph-aware folding is oracle-tested.

The C reader validates bounds, arithmetic, duplicate names, directory checksum
and every tensor checksum before exposing the model. The CUDA context uploads
the payload once and owns previous BEV, track memory, a reusable activation
arena and a non-blocking stream.

After validation the runtime retains a compact in-memory directory for all
2,459 records. `ua_model_find_tensor` resolves a full checkpoint key to dtype,
rank, shape, payload-relative byte offset and byte length. The CLI exposes the
same read-only contract:

```sh
uniad inspect-model model.uaw2 --tensor img_backbone.conv1.weight
```

Lookup failure is explicit; graph code must never guess an offset or silently
substitute a similarly shaped tensor.
