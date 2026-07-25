# UniAD.c

UniAD.c is an auditable C11 **synthetic vertical slice** of the representation
flow in [OpenDriveLab UniAD](https://github.com/OpenDriveLab/UniAD). It runs a
small deterministic six-camera, two-frame graph on CPU, exposes a stable C API,
and validates its canonical result against an independent PyTorch oracle.

This is not a port of the released R101 checkpoint. It does not claim numerical
equivalence, nuScenes accuracy, production latency, or nuPlan support. The
production profile is inspectable but intentionally cannot execute without a
future compatible export.

```sh
make
./build/uniad generate-demo build/demo
./build/uniad demo --dir build/demo
python3 tools/oracle.py --compare ./build/uniad --asset-dir build/demo
make test
```

Useful commands:

```sh
./build/uniad doctor
./build/uniad inspect-model build/demo/demo.uaw
./build/uniad benchmark --dir build/demo --warmup 2 --runs 10
./build/uniad infer --profile production
```

The CPU runtime depends only on the C standard library. CUDA is an optional,
explicit backend boundary (`CUDA=ON`); requesting an unavailable backend fails
instead of falling back. See [the technical article](docs/index.html), the
[model contract](evidence/model-contract.md), and [limitations](LIMITATIONS.md).

Apache-2.0. This repository contains original clean-room code and documentation;
upstream provenance is recorded in [SOURCES.md](SOURCES.md).
