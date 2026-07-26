# UniAD.c

UniAD.c is an auditable C11 **synthetic vertical slice** of the representation
flow in [OpenDriveLab UniAD](https://github.com/OpenDriveLab/UniAD). It runs a
small deterministic six-camera, two-frame graph on CPU, exposes a stable C API,
and validates its canonical result against an independent PyTorch oracle.

The v2 API and UAW2 converter now validate and upload the released R101
checkpoint, but the official operator graph is not yet implemented. The
project therefore does not claim numerical equivalence, nuScenes accuracy,
production latency, or nuPlan support.

```sh
make
./build/cpu/uniad generate-demo build/demo
./build/cpu/uniad demo --dir build/demo
python3 tools/oracle.py --compare ./build/cpu/uniad --asset-dir build/demo
make test
```

Useful commands:

```sh
./build/cpu/uniad doctor
./build/cpu/uniad inspect-model build/demo/demo.uaw
./build/cpu/uniad inspect-model --production
./build/cpu/uniad benchmark --dir build/demo --warmup 2 --runs 10
make CUDA=ON test

python3 tools/convert_uaw2.py /path/uniad_base_e2e.pth \
  build/uniad_base_e2e.uaw2 \
  --config /path/base_e2e.py \
  --manifest evidence/uniad_base_e2e.uaw2.json
./build/cuda/uniad inspect-model build/uniad_base_e2e.uaw2
```

The CPU runtime depends only on the C standard library. CUDA is an optional,
explicit backend boundary (`CUDA=ON`); requesting an unavailable backend fails
instead of falling back. See [the technical article](docs/index.html), the
[model contract](evidence/model-contract.md), and [limitations](LIMITATIONS.md).
The current draft checkpoint and ordered resume plan are recorded in
[CONTINUATION.md](CONTINUATION.md).

Live technical article: https://hova88.github.io/uniad.c/

Apache-2.0. This repository contains original clean-room code and documentation;
upstream provenance is recorded in [SOURCES.md](SOURCES.md).
