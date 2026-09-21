# Contributing

EdgeVision is a measured engineering log as much as a runtime: every change that affects
speed or accuracy comes with a number, and every number comes with the script that produced
it. This page is the short version of how to work in the repository; `README.md` has the
architecture and results, `infra/README.md` the AWS GPU box.

## Set up (CPU, any OS)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock -r requirements-dev.txt -e .
python scripts/export_onnx.py && python scripts/export_onnx.py --nms   # downloads yolo26n.pt, exports the two graphs
pre-commit install                                                       # optional: the CI checks on every commit
```

`requirements.lock` is the exact environment that produced the CPU numbers; `requirements.txt`
is the unpinned list for humans and `pyproject.toml` the package contract. Add a dependency to
`requirements.txt` and `pyproject.toml`, then regenerate the lock with `uv pip freeze`.

## Check before pushing

The CI (`.github/workflows/ci.yml`) runs exactly these; `pre-commit run --all-files` covers the
static part locally.

| what | command |
|---|---|
| Python lint + format | `ruff check python scripts tests && ruff format --check python scripts tests` |
| Python types | `mypy` |
| Shell scripts | `shellcheck -S warning scripts/*.sh scripts/aws/*.sh` |
| Python tests + coverage (floor 80%) | `pytest --cov --cov-fail-under=80` |
| C++ GPU-free library, strict warnings + CTests | `cmake -S cpp -B cpp/build-cpu -DEDGEVISION_CPU_ONLY=ON && cmake --build cpp/build-cpu && ctest --test-dir cpp/build-cpu` |
| C++ format | `clang-format --dry-run --Werror cpp/src/* cpp/include/edgevision/* cpp/tests/*` |
| C++ static analysis (CPU sources) | `clang-tidy cpp/src/{tracker,metrics,names,cli,report}.cpp cpp/tests/test_*.cpp -- -std=c++17 -Icpp/include` |

Tests that need a GPU (`tests/test_tensorrt_detector.py`, `tests/test_cpp_runtime.py`, the
`letterbox` and `nv12` CTests) skip themselves on the CPU and run on the GPU box through the
`scripts/gpu_sprint*.sh` scripts.

## Working on the GPU parts

The full runtime (CUDA kernel, TensorRT, NVDEC) builds only inside `docker/Dockerfile.tensorrt`
on a machine with an NVIDIA GPU. The repository's own box is an EC2 g4dn.xlarge defined in
`infra/gpu-dev.yaml`:

```bash
scripts/aws/resume.sh            # launch the instance, push code + ONNX, rebuild the image
scripts/aws/run.sh 'docker run --rm --gpus all --ipc=host -v /opt/edgevision/repo:/workspace/edgevision edgevision:trt scripts/gpu_sprint8.sh'
scripts/aws/sync-down.sh         # engines + benchmarks/results back to the laptop
scripts/aws/standby.sh           # ALWAYS when done: removes the instance and its disk (zero cost)
```

A GPU change is complete when a `scripts/gpu_sprint*.sh` (or an existing one) reproduces its
measurement, the JSON report and log are in `benchmarks/results/`, and `benchmarks/README.md`
states the number and what it changed.

## Conventions

- **One hypothesis per change.** Say which bottleneck the change targets and which measurement
  moved; a rejected hypothesis is still a result (see INT8 and batching in `benchmarks/README.md`).
- **Python and C++ share contracts.** `Detection` fields, the metrics summary
  (`mean_ms/p50_ms/p95_ms/max_ms/samples`) and the benchmark JSON schema are the same on both
  sides and checked by parity tests. Change them together.
- **Static graphs own their input size.** ONNX graphs and TensorRT engines fix their shape at
  export; the detectors read it from the graph. Never hard-code 640 or 512 in a detector.
- **Config is data.** Runtime knobs go in `configs/app.yaml` or CLI flags, not in code.
- **Commits** are imperative, scoped (`cpp:`, `python:`, `ci:`, `docs:`, `infra:`), and explain
  the why in the body when it is not obvious from the diff. No co-author trailers.
- **Line endings** are LF everywhere (`.gitattributes`); scripts must keep running on the Linux
  box after being edited on Windows.
- **Formatting** is not discussed: `ruff format` for Python, `cpp/.clang-format` for C++.

## Releases

Tags follow `vMAJOR.MINOR.PATCH`; `CHANGELOG.md` gets an entry per release. `v0.1.0` marks
the end of Sprint 8 (C++ runtime with NVDEC and ByteTrack on the T4).
